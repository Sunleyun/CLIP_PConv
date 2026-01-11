import torch
import torch.nn.functional as F
import math

@torch.no_grad()
def psnr(pred: torch.Tensor, gt: torch.Tensor, eps=1e-8) -> float:
    mse = F.mse_loss(pred, gt).item()
    return 10.0 * math.log10(1.0 / max(mse, eps))

@torch.no_grad()
def psnr_masked(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor, eps=1e-8) -> float:
    """
    mask: (B,1,H,W) in {0,1}. compute MSE only where mask==1.
    """
    w = mask.repeat(1, pred.shape[1], 1, 1)
    diff2 = (pred - gt) ** 2 * w
    denom = w.sum().item() + eps
    mse = diff2.sum().item() / denom
    return 10.0 * math.log10(1.0 / max(mse, eps))
