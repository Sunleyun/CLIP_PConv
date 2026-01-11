import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List

# ---------------------------
# Partial Convolution Layer
# ---------------------------
class PConv2d(nn.Module):
    """
    Partial Convolution from NVIDIA inpainting idea:
    - Input: (x, mask) where mask is 0/1
    - Convolution is normalized by valid pixel ratio
    - Output mask is updated (valid if any valid in kernel)
    """
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=bias)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # mask conv is fixed: sums valid pixels in each window
        self.register_buffer("weight_mask", torch.ones(1, 1, kernel_size, kernel_size))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B,C,H,W)
        mask: (B,1,H,W) with 0/1
        """
        assert mask.shape[1] == 1
        x_masked = x * mask

        out = self.conv(x_masked)

        with torch.no_grad():
            # Count valid pixels in each sliding window
            valid_count = F.conv2d(
                mask,
                self.weight_mask,
                bias=None,
                stride=self.stride,
                padding=self.padding,
            )
            # Update mask: valid if any pixel is valid
            out_mask = (valid_count > 0).float()

        # Normalize output to avoid intensity shift
        eps = 1e-6
        # valid_count ranges [0, k*k], normalize to scale up where few valids
        norm = (self.kernel_size * self.kernel_size) / (valid_count + eps)
        out = out * norm

        # Where no valid pixels, force output to 0
        out = out * out_mask

        return out, out_mask

# ---------------------------
# Blocks
# ---------------------------
class PConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, use_gn=True):
        super().__init__()
        self.pconv = PConv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=True)
        self.use_gn = use_gn
        self.gn = nn.GroupNorm(num_groups=min(32, out_ch), num_channels=out_ch) if use_gn else nn.Identity()
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x, mask):
        x, mask = self.pconv(x, mask)
        x = self.gn(x)
        x = self.act(x)
        return x, mask

class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, use_gn=True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.use_gn = use_gn
        self.gn = nn.GroupNorm(num_groups=min(32, out_ch), num_channels=out_ch) if use_gn else nn.Identity()
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.conv(x)
        x = self.gn(x)
        x = self.act(x)
        return x

# ---------------------------
# UNet with Partial Conv encoder
# ---------------------------
class PConvUNet(nn.Module):
    """
    Input channels: 5 = RGB(3) + M(1) + G(1)
    We will construct a UNet-like network:
      Encoder: partial conv blocks with downsampling
      Decoder: upsample + skip concat + conv
    Output: RGB in [0,1] using sigmoid
    """
    def __init__(self, in_ch=5, base=64, use_gn=True):
        super().__init__()
        # Encoder
        self.e1 = PConvBlock(in_ch, base, k=7, s=1, p=3, use_gn=use_gn)
        self.e2 = PConvBlock(base, base*2, k=5, s=2, p=2, use_gn=use_gn)
        self.e3 = PConvBlock(base*2, base*4, k=5, s=2, p=2, use_gn=use_gn)
        self.e4 = PConvBlock(base*4, base*8, k=3, s=2, p=1, use_gn=use_gn)
        self.e5 = PConvBlock(base*8, base*8, k=3, s=2, p=1, use_gn=use_gn)

        # Decoder
        self.u4 = UpBlock(base*8, base*8, use_gn=use_gn)
        self.u3 = UpBlock(base*16, base*4, use_gn=use_gn)
        self.u2 = UpBlock(base*8, base*2, use_gn=use_gn)
        self.u1 = UpBlock(base*4, base, use_gn=use_gn)

        self.out_conv = nn.Conv2d(base*2, 3, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        x: (B,5,H,W) includes RGB+M+G, but partial conv mask should be derived from M channel
        mask: (B,1,H,W) binary mask where 1 = valid, 0 = HOLE
              For inpainting-like tasks: we treat highlight region as "hole" to be restored.
              Thus valid = 1 - M_highlight
        """
        # Encoder with skip connections
        e1, m1 = self.e1(x, mask)
        e2, m2 = self.e2(e1, m1)
        e3, m3 = self.e3(e2, m2)
        e4, m4 = self.e4(e3, m3)
        e5, m5 = self.e5(e4, m4)

        # Decoder
        d4 = self.u4(e5)
        d4 = torch.cat([d4, e4], dim=1)

        d3 = self.u3(d4)
        d3 = torch.cat([d3, e3], dim=1)

        d2 = self.u2(d3)
        d2 = torch.cat([d2, e2], dim=1)

        d1 = self.u1(d2)
        d1 = torch.cat([d1, e1], dim=1)

        out = self.out_conv(d1)
        out = torch.sigmoid(out)
        return out
