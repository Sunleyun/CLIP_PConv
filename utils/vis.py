import os
import torch
import torchvision
import numpy as np

def overlay_mask_rgb(img: torch.Tensor, mask: torch.Tensor, color=(1.0, 0.0, 0.0), alpha=0.45) -> torch.Tensor:
    """
    img: (3,H,W) in [0,1]
    mask: (1,H,W) in {0,1}
    """
    c = torch.tensor(color, device=img.device).view(3,1,1)
    m = mask.clamp(0,1)
    return img * (1 - alpha*m) + c * (alpha*m)

def heatmap_overlay(img: torch.Tensor, heat: torch.Tensor, alpha=0.45) -> torch.Tensor:
    """
    heat: (1,H,W) in [0,1]
    creates a simple 3ch heat overlay (yellow-ish) without matplotlib
    """
    h = heat.clamp(0,1)
    heat_rgb = torch.cat([h, h, torch.zeros_like(h)], dim=0)  # R+G
    return img*(1-alpha) + heat_rgb*alpha

@torch.no_grad()
def save_debug_grid(
    out_path: str,
    I_in: torch.Tensor,   # (B,3,H,W)
    I_gt: torch.Tensor,   # (B,3,H,W)
    I_pred: torch.Tensor, # (B,3,H,W)
    M: torch.Tensor,      # (B,1,H,W)
    G: torch.Tensor,      # (B,1,H,W)
    n: int = 4,
):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    B = I_in.shape[0]
    n = min(n, B)
    tiles = []
    for i in range(n):
        a = I_in[i]
        d = I_gt[i]
        p = I_pred[i]
        m = M[i]
        g = G[i]

        a_m = overlay_mask_rgb(a, m)
        p_m = overlay_mask_rgb(p, m)
        a_g = heatmap_overlay(a, g)

        err = (p - d).abs().mean(dim=0, keepdim=True).clamp(0,1)
        err_rgb = torch.cat([err, err, err], dim=0)

        tiles.extend([a, a_m, a_g, p, p_m, d, err_rgb])

    grid = torchvision.utils.make_grid(tiles, nrow=7, padding=2)
    torchvision.utils.save_image(grid, out_path)
