from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
import streamlit as st

from medical_diffusion.models import BasicModel
from medical_diffusion.utils.train_utils import EMAModel
from medical_diffusion.utils.math_utils import kl_gaussians


# 定义边缘增强网络
class EdgeEnhancementNet(nn.Module):
    def __init__(self):
        super(EdgeEnhancementNet, self).__init__()
        # 修改为接受8个通道的输入
        self.conv1 = nn.Conv2d(in_channels=8, out_channels=1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(1, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 8, kernel_size=3, padding=1)  # 输出仍然是3个通道
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.conv4(x)
        return x


def edge_loss(pred, target):
    # 检查pred和target的形状
    print(f"pred shape: {pred.shape}, target shape: {target.shape}")

    # 使用插值调整pred的大小
    if pred.size()[2:] != target.size()[2:]:
        pred = F.interpolate(pred, size=(target.size(2), target.size(3)), mode='bilinear', align_corners=False)
        print(f"Resized pred shape: {pred.shape}")

    # 确保输入张量的形状一致
    assert pred.size() == target.size(), "pred and target must have the same shape"

    # 检查 NaN 和 inf
    if torch.isnan(pred).any() or torch.isinf(pred).any():
        print("pred contains NaN or inf values.")
    if torch.isnan(target).any() or torch.isinf(target).any():
        print("target contains NaN or inf values.")

    # 定义Sobel算子
    sobel_x = torch.tensor([[1, 0, -1],
                            [2, 0, -2],
                            [1, 0, -1]], dtype=torch.float32).to(pred.device).view(1, 1, 3, 3)

    sobel_y = torch.tensor([[1, 2, 1],
                            [0, 0, 0],
                            [-1, -2, -1]], dtype=torch.float32).to(pred.device).view(1, 1, 3, 3)

    pred_edges = []
    target_edges = []

    for i in range(pred.size(1)):
        pred_channel = pred[:, i:i + 1, :, :]
        target_channel = target[:, i:i + 1, :, :]

        pred_edges_x = F.conv2d(pred_channel, sobel_x, padding=1)
        pred_edges_y = F.conv2d(pred_channel, sobel_y, padding=1)
        target_edges_x = F.conv2d(target_channel, sobel_x, padding=1)
        target_edges_y = F.conv2d(target_channel, sobel_y, padding=1)

        # 增加小常数以避免平方根负值
        pred_edges_channel = torch.sqrt(pred_edges_x ** 2 + pred_edges_y ** 2 + 1e-6)
        target_edges_channel = torch.sqrt(target_edges_x ** 2 + target_edges_y ** 2 + 1e-6)

        pred_edges.append(pred_edges_channel)
        target_edges.append(target_edges_channel)

    pred_edges = torch.cat(pred_edges, dim=1)
    target_edges = torch.cat(target_edges, dim=1)

    # 返回损失
    return F.l1_loss(pred_edges, target_edges)



class DiffusionPipeline(BasicModel):
    def __init__(self,
                 noise_scheduler,
                 noise_estimator,
                 latent_embedder=None,
                 noise_scheduler_kwargs={},
                 noise_estimator_kwargs={},
                 latent_embedder_checkpoint='',
                 estimator_objective='x_T',
                 estimate_variance=False,
                 use_self_conditioning=False,
                 classifier_free_guidance_dropout=0.5,
                 num_samples=4,
                 do_input_centering=True,
                 clip_x0=True,
                 use_ema=False,
                 ema_kwargs={},
                 optimizer=torch.optim.AdamW,
                 optimizer_kwargs={'lr': 1e-4},
                 lr_scheduler=None,
                 lr_scheduler_kwargs={},
                 loss=torch.nn.L1Loss,
                 loss_kwargs={},
                 sample_every_n_steps=1000):

        super().__init__(optimizer, optimizer_kwargs, lr_scheduler, lr_scheduler_kwargs)
        self.loss_fct = loss(**loss_kwargs)
        self.sample_every_n_steps = sample_every_n_steps

        noise_estimator_kwargs['estimate_variance'] = estimate_variance
        noise_estimator_kwargs['use_self_conditioning'] = use_self_conditioning

        self.noise_scheduler = noise_scheduler(**noise_scheduler_kwargs)
        self.noise_estimator = noise_estimator(**noise_estimator_kwargs)

        with torch.no_grad():
            if latent_embedder is not None:
                self.latent_embedder = latent_embedder.load_from_checkpoint(latent_embedder_checkpoint)
                for param in self.latent_embedder.parameters():
                    param.requires_grad = False
            else:
                self.latent_embedder = None

        self.estimator_objective = estimator_objective
        self.use_self_conditioning = use_self_conditioning
        self.num_samples = num_samples
        self.classifier_free_guidance_dropout = classifier_free_guidance_dropout
        self.do_input_centering = do_input_centering
        self.estimate_variance = estimate_variance
        self.clip_x0 = clip_x0
        self.use_ema = use_ema

        # 初始化边缘增强网络
        self.edge_enhancer = EdgeEnhancementNet()

        if use_ema:
            self.ema_model = EMAModel(self.noise_estimator, **ema_kwargs)

    def _step(self, batch: dict, batch_idx: int, state: str, step: int, optimizer_idx: int):
        results = {}
        x_0 = batch['source']
        condition = batch.get('target', None)

        # 嵌入到潜在空间或归一化
        if self.latent_embedder is not None:
            self.latent_embedder.eval()
            with torch.no_grad():
                x_0 = self.latent_embedder.encode(x_0)

        if self.do_input_centering:
            x_0 = 2 * x_0 - 1  # [0, 1] -> [-1, 1]

        # 随机选择 t [0,T-1] 并计算 x_t（x_0 的噪声版本）
        x_t, x_T, t = self.noise_scheduler.sample(x_0)

        # 使用 EMA 模型
        if self.use_ema and (state != 'train'):
            noise_estimator = self.ema_model.averaged_model
        else:
            noise_estimator = self.noise_estimator

        # 重新估计 x_T 或 x_0
        self_cond = None
        if self.use_self_conditioning:
            with torch.no_grad():
                pred, pred_vertical = noise_estimator(x_t, t, condition, None)
                if self.estimate_variance:
                    pred, _ = pred.chunk(2, dim=1)  # 分离预测和方差估计
                if self.estimator_objective == "x_T":
                    self_cond = self.noise_scheduler.estimate_x_0(x_t, pred, t=t, clip_x0=self.clip_x0)
                elif self.estimator_objective == "x_0":
                    self_cond = self.noise_scheduler.estimate_x_T(x_t, pred, t=t, clip_x0=self.clip_x0)
                else:
                    raise NotImplementedError(f"Option estimator_target={self.estimator_objective} not supported.")

        # 分类器自由引导
        if torch.rand(1) < self.classifier_free_guidance_dropout:
            condition = None

        # 运行去噪
        pred, pred_vertical = noise_estimator(x_t, t, condition, self_cond)

        if self.estimate_variance:
            pred, pred_var = pred.chunk(2, dim=1)

        # 指定目标
        if self.estimator_objective == "x_T":
            target = x_T
        elif self.estimator_objective == "x_0":
            target = x_0
        else:
            raise NotImplementedError(f"Option estimator_target={self.estimator_objective} not supported.")

        # 计算损失
        loss = self.loss_fct(pred, target)

        # 使用边缘增强网络
        enhanced_pred = self.edge_enhancer(pred)

        # 计算边缘损失
        # loss_edge = edge_loss(enhanced_pred, target)
        # loss += loss_edge  # 将边缘损失添加到总损失中

        results['loss'] = loss
        # results['edge_loss'] = loss_edge  # 保存边缘损失到结果

        # 计算指标
        with torch.no_grad():
            results['L2'] = F.mse_loss(pred, target)
            # results['L1'] = F.l1_loss(pred, target)

        # 日志记录
        for metric_name, metric_val in results.items():
            self.log(f"{state}/{metric_name}", metric_val, batch_size=x_0.shape[0], on_step=True, on_epoch=True)

        # 图像保存
        if self.global_step != 0 and self.global_step % self.sample_every_n_steps == 0:
            path_out = Path(self.logger.log_dir) / 'images'
            path_out.mkdir(parents=True, exist_ok=True)
            save_image(enhanced_pred, path_out / f'pred_edge_enhanced_{self.global_step}.png', normalize=True)

        return loss

    # 其他方法保持不变...

    def forward(self, x_t, t, condition=None, self_cond=None, guidance_scale=1.0, cold_diffusion=False, un_cond=None):
        # Note: x_t expected to be in range ~ [-1, 1]
        if self.use_ema:
            noise_estimator = self.ema_model.averaged_model
        else:
            noise_estimator = self.noise_estimator

        # Concatenate inputs for guided and unguided diffusion as proposed by classifier-free-guidance
        if (condition is not None) and (guidance_scale != 1.0):
            # Model prediction
            pred_uncond, _ = noise_estimator(x_t, t, condition=un_cond, self_cond=self_cond)
            pred_cond, _ = noise_estimator(x_t, t, condition=condition, self_cond=self_cond)
            pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)

            if self.estimate_variance:
                pred_uncond, pred_var_uncond = pred_uncond.chunk(2, dim=1)
                pred_cond, pred_var_cond = pred_cond.chunk(2, dim=1)
                pred_var = pred_var_uncond + guidance_scale * (pred_var_cond - pred_var_uncond)
        else:
            pred, _ = noise_estimator(x_t, t, condition=condition, self_cond=self_cond)
            if self.estimate_variance:
                pred, pred_var = pred.chunk(2, dim=1)

        if self.estimate_variance:
            pred_var_scale = pred_var / 2 + 0.5  # [-1, 1] -> [0, 1]
            pred_var_value = pred_var
        else:
            pred_var_scale = 0
            pred_var_value = None

            # pred_var_scale = pred_var_scale.clamp(0, 1)

        if self.estimator_objective == 'x_0':
            x_t_prior, x_0 = self.noise_scheduler.estimate_x_t_prior_from_x_0(x_t, t, pred, clip_x0=self.clip_x0,
                                                                              var_scale=pred_var_scale,
                                                                              cold_diffusion=cold_diffusion)
            x_T = self.noise_scheduler.estimate_x_T(x_t, x_0=pred, t=t, clip_x0=self.clip_x0)
            self_cond = x_T
        elif self.estimator_objective == 'x_T':
            x_t_prior, x_0 = self.noise_scheduler.estimate_x_t_prior_from_x_T(x_t, t, pred, clip_x0=self.clip_x0,
                                                                              var_scale=pred_var_scale,
                                                                              cold_diffusion=cold_diffusion)
            x_T = pred
            self_cond = x_0
        else:
            raise ValueError("Unknown Objective")

        return x_t_prior, x_0, x_T, self_cond

    @torch.no_grad()
    def denoise(self, x_t, steps=None, condition=None, use_ddim=True, **kwargs):
        self_cond = None

        # ---------- run denoise loop ---------------
        if use_ddim:
            steps = self.noise_scheduler.timesteps if steps is None else steps
            timesteps_array = torch.linspace(0, self.noise_scheduler.T - 1, steps, dtype=torch.long,
                                             device=x_t.device)  # [0, 1, 2, ..., T-1] if steps = T
        else:
            timesteps_array = self.noise_scheduler.timesteps_array[
                slice(0, steps)]  # [0, ...,T-1] (target time not time of x_t)

        st_prog_bar = st.progress(0)
        for i, t in tqdm(enumerate(reversed(timesteps_array))):
            st_prog_bar.progress((i + 1) / len(timesteps_array))

            # UNet prediction
            x_t, x_0, x_T, self_cond = self(x_t, t.expand(x_t.shape[0]), condition, self_cond=self_cond, **kwargs)
            self_cond = self_cond if self.use_self_conditioning else None

            if use_ddim and (steps - i - 1 > 0):
                t_next = timesteps_array[steps - i - 2]
                alpha = self.noise_scheduler.alphas_cumprod[t]
                alpha_next = self.noise_scheduler.alphas_cumprod[t_next]
                sigma = kwargs.get('eta', 1) * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
                c = (1 - alpha_next - sigma ** 2).sqrt()
                noise = torch.randn_like(x_t)
                x_t = x_0 * alpha_next.sqrt() + c * x_T + sigma * noise

        # ------ Eventually decode from latent space into image space--------
        if self.latent_embedder is not None:
            x_t = self.latent_embedder.decode(x_t)

        return x_t  # Should be x_0 in final step (t=0)

    @torch.no_grad()
    def sample(self, num_samples, img_size, condition=None, **kwargs):
        template = torch.zeros((num_samples, *img_size), device=self.device)
        x_T = self.noise_scheduler.x_final(template)
        x_0 = self.denoise(x_T, condition=condition, **kwargs)
        return x_0

    @torch.no_grad()
    def interpolate(self, img1, img2, i=None, condition=None, lam=0.5, **kwargs):
        assert img1.shape == img2.shape, "Image 1 and 2 must have equal shape"

        t = self.noise_scheduler.T - 1 if i is None else i
        t = torch.full(img1.shape[:1], i, device=img1.device)

        img1_t = self.noise_scheduler.estimate_x_t(img1, t=t, clip_x0=self.clip_x0)
        img2_t = self.noise_scheduler.estimate_x_t(img2, t=t, clip_x0=self.clip_x0)

        img = (1 - lam) * img1_t + lam * img2_t
        img = self.denoise(img, i, condition, **kwargs)
        return img

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.ema_model.step(self.noise_estimator)

    def configure_optimizers(self):
        optimizer = self.optimizer(self.noise_estimator.parameters(), **self.optimizer_kwargs)
        if self.lr_scheduler is not None:
            lr_scheduler = {
                'scheduler': self.lr_scheduler(optimizer, **self.lr_scheduler_kwargs),
                'interval': 'step',
                'frequency': 1
            }
            return [optimizer], [lr_scheduler]
        else:
            return [optimizer]