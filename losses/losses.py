import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

# ---------------------------
# SSIM (simple, differentiable)
# ---------------------------
def _gaussian_window(window_size: int = 11, sigma: float = 1.5, channels: int = 3, device="cpu"):
    import math
    coords = torch.arange(window_size, device=device).float() - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    w1d = g.view(1, 1, window_size)
    w2d = (w1d.transpose(2, 1) @ w1d).view(1, 1, window_size, window_size)
    w2d = w2d.expand(channels, 1, window_size, window_size).contiguous()
    return w2d

def ssim(img1, img2, window_size=11, sigma=1.5, data_range=1.0):
    """
    img1,img2: (B,C,H,W) in [0,1]
    returns mean SSIM over batch
    """
    device = img1.device
    C = img1.shape[1]
    window = _gaussian_window(window_size, sigma, C, device=device)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=C)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=C)
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu12 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=C) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=C) - mu12

    K1, K2 = 0.01, 0.03
    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    ssim_map = ((2 * mu12 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2) + 1e-7)
    return ssim_map.mean()

# ---------------------------
# Gradient / Laplacian
# ---------------------------
def sobel_grad(x: torch.Tensor):
    """
    x: (B,C,H,W)
    returns gx, gy
    """
    device = x.device
    kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32, device=device).view(1,1,3,3)
    ky = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32, device=device).view(1,1,3,3)
    C = x.shape[1]
    kx = kx.repeat(C,1,1,1)
    ky = ky.repeat(C,1,1,1)
    gx = F.conv2d(x, kx, padding=1, groups=C)
    gy = F.conv2d(x, ky, padding=1, groups=C)
    return gx, gy

def laplacian(x: torch.Tensor):
    device = x.device
    k = torch.tensor([[0,1,0],[1,-4,1],[0,1,0]], dtype=torch.float32, device=device).view(1,1,3,3)
    C = x.shape[1]
    k = k.repeat(C,1,1,1)
    return F.conv2d(x, k, padding=1, groups=C)

# ---------------------------
# Stage-configurable Loss
# ---------------------------
class HighlightRemovalLoss(nn.Module):
    def __init__(
        self,
        w_l1=1.0,
        w_ssim=0.2,
        w_grad=0.0,
        w_lap=0.0,
        w_keep=0.0,
        dilate_keep=5,
        grad_g_thresh=0.5,
    ):
        super().__init__()
        self.w_l1 = w_l1
        self.w_ssim = w_ssim
        self.w_grad = w_grad
        self.w_lap = w_lap
        self.w_keep = w_keep
        self.dilate_keep = dilate_keep
        self.grad_g_thresh = grad_g_thresh

    @staticmethod
    def _dilate_binary(mask01: torch.Tensor, r: int):
        # very simple dilation using maxpool (works for binary in [0,1])
        k = 2*r + 1
        return F.max_pool2d(mask01, kernel_size=k, stride=1, padding=r)

    def forward(self, pred: torch.Tensor, gt: torch.Tensor, I_in: torch.Tensor, M: torch.Tensor, G: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        pred, gt, I_in: (B,3,H,W)
        M: (B,1,H,W) highlight mask 0/1
        G: (B,1,H,W) boundary weight map 0..1
        """
        losses = {}

        # L1
        l1 = F.l1_loss(pred, gt)
        losses["l1"] = l1

        # SSIM loss = 1 - SSIM
        s = ssim(pred, gt)
        losses["ssim"] = 1.0 - s

        # Boundary-weighted gradient loss
        if self.w_grad > 0:
            gx_p, gy_p = sobel_grad(pred)
            gx_g, gy_g = sobel_grad(gt)
            # weight map broadcast to 3 channels
            w = G.repeat(1, pred.shape[1], 1, 1)
            lgrad = (w * (gx_p - gx_g).abs()).mean() + (w * (gy_p - gy_g).abs()).mean()
            losses["grad_g"] = lgrad
        else:
            losses["grad_g"] = pred.new_tensor(0.0)

        # Boundary-weighted Laplacian loss
        if self.w_lap > 0:
            lp = laplacian(pred)
            lg = laplacian(gt)
            w = G.repeat(1, pred.shape[1], 1, 1)
            llap = (w * (lp - lg).abs()).mean()
            losses["lap_g"] = llap
        else:
            losses["lap_g"] = pred.new_tensor(0.0)

        # Keep non-highlight region unchanged (texture preservation)
        if self.w_keep > 0:
            keep_mask = 1.0 - self._dilate_binary(M, self.dilate_keep)
            w = keep_mask.repeat(1, pred.shape[1], 1, 1)
            lkeep = (w * (pred - I_in).abs()).mean()
            losses["keep"] = lkeep
        else:
            losses["keep"] = pred.new_tensor(0.0)

        total = (
            self.w_l1 * losses["l1"]
            + self.w_ssim * losses["ssim"]
            + self.w_grad * losses["grad_g"]
            + self.w_lap * losses["lap_g"]
            + self.w_keep * losses["keep"]
        )
        losses["total"] = total
        return losses

def make_loss_for_stage(stage: str) -> HighlightRemovalLoss:
    stage = stage.lower()
    if stage == "stage1":
        return HighlightRemovalLoss(w_l1=1.0, w_ssim=0.2, w_grad=0.0, w_lap=0.0, w_keep=0.0)
    if stage == "stage2":
        return HighlightRemovalLoss(w_l1=1.0, w_ssim=0.2, w_grad=0.5, w_lap=0.2, w_keep=0.0)
    if stage == "stage3":
        return HighlightRemovalLoss(w_l1=1.0, w_ssim=0.2, w_grad=0.5, w_lap=0.2, w_keep=0.5)
    raise ValueError(f"Unknown stage: {stage}. Use stage1/stage2/stage3")
