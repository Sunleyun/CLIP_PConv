import os
import json
import lmdb
import cv2
import numpy as np
from typing import Dict, Tuple, Optional
from torch.utils.data import Dataset
import torch
from scipy.ndimage import binary_dilation, binary_erosion, distance_transform_edt

# ---------------------------
# Utils
# ---------------------------
def decode_png_bytes(png_bytes: bytes, flags=cv2.IMREAD_UNCHANGED) -> np.ndarray:
    arr = np.frombuffer(png_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, flags)
    if img is None:
        raise ValueError("cv2.imdecode failed")
    return img

def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3 and img.shape[2] == 3:
        return img[:, :, ::-1]
    return img

def pad_to_256_reflect(img: np.ndarray, is_mask: bool = False, out_size: int = 256) -> np.ndarray:
    h, w = img.shape[:2]
    if h > out_size or w > out_size:
        raise ValueError(f"Input too large: {h}x{w}, expected <= {out_size}")
    pad_h = out_size - h
    pad_w = out_size - w
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    if is_mask:
        border_type = cv2.BORDER_CONSTANT
        value = 0
        padded = cv2.copyMakeBorder(img, top, bottom, left, right, border_type, value=value)
    else:
        border_type = cv2.BORDER_REFLECT_101
        padded = cv2.copyMakeBorder(img, top, bottom, left, right, border_type)
    return padded

def make_boundary_band_weight(mask01: np.ndarray, r: int = 1, rp: int = 3, sigma: float = 3.0) -> np.ndarray:
    """
    mask01: HxW binary (0/1)
    returns G: HxW float in [0,1], high near boundary.
    """
    m = mask01.astype(bool)

    dil = binary_dilation(m, iterations=r)
    ero = binary_erosion(m, iterations=r)
    boundary = np.logical_and(dil, np.logical_not(ero))  # boundary band

    # dist to boundary: 0 at boundary, grows outward
    # distance_transform_edt computes distance to nearest zero; so invert boundary:
    dist = distance_transform_edt(~boundary).astype(np.float32)
    G = np.exp(-dist / float(sigma)).astype(np.float32)

    near = binary_dilation(m, iterations=rp)
    G = G * near.astype(np.float32)
    # normalize to [0,1] already, but clamp for safety
    G = np.clip(G, 0.0, 1.0)
    return G

# ---------------------------
# Dataset
# ---------------------------
class SHIQLmdbDataset(Dataset):
    """
    LMDB keys: "{oid}/A" (highlight), "{oid}/D" (diffuse/GT), "{oid}/T" (binary mask).
    meta.json contains: {"keys": [oid1, oid2, ...]}
    """
    def __init__(
        self,
        lmdb_dir: str,
        split: str = "train",
        out_size: int = 256,
        use_g: bool = True,
        g_params: Tuple[int, int, float] = (1, 3, 3.0),  # (r, rp, sigma)
        limit: Optional[int] = None,
    ):
        super().__init__()
        self.lmdb_dir = lmdb_dir
        self.split = split
        self.out_size = out_size
        self.use_g = use_g
        self.g_params = g_params

        meta_path = os.path.join(lmdb_dir, "meta.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"meta.json not found: {meta_path}")

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.keys = meta["keys"]
        if limit is not None:
            self.keys = self.keys[: int(limit)]

        # Open LMDB env lazily per worker
        self._env = None

    def _get_env(self):
        if self._env is None:
            self._env = lmdb.open(
                self.lmdb_dir,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
                max_readers=256,
            )
        return self._env

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        oid = self.keys[idx]
        env = self._get_env()
        with env.begin(write=False) as txn:
            a = txn.get(f"{oid}/A".encode("utf-8"))
            d = txn.get(f"{oid}/D".encode("utf-8"))
            t = txn.get(f"{oid}/T".encode("utf-8"))
            if a is None or d is None or t is None:
                raise KeyError(f"Missing key for oid={oid}")

        # Decode
        A = decode_png_bytes(a, flags=cv2.IMREAD_COLOR)  # BGR uint8
        D = decode_png_bytes(d, flags=cv2.IMREAD_COLOR)
        T = decode_png_bytes(t, flags=cv2.IMREAD_GRAYSCALE)  # uint8

        A = bgr_to_rgb(A)
        D = bgr_to_rgb(D)

        # Binary mask 0/1
        M = (T > 127).astype(np.uint8)

        # Pad to out_size
        if self.out_size is not None:
            A = pad_to_256_reflect(A, is_mask=False, out_size=self.out_size)
            D = pad_to_256_reflect(D, is_mask=False, out_size=self.out_size)
            M = pad_to_256_reflect(M, is_mask=True, out_size=self.out_size)

        # Build G
        if self.use_g:
            r, rp, sigma = self.g_params
            G = make_boundary_band_weight(M, r=r, rp=rp, sigma=sigma)
        else:
            G = np.zeros_like(M, dtype=np.float32)

        # To torch float [0,1]
        A_t = torch.from_numpy(A).permute(2, 0, 1).float() / 255.0
        D_t = torch.from_numpy(D).permute(2, 0, 1).float() / 255.0
        M_t = torch.from_numpy(M).unsqueeze(0).float()  # 0/1
        G_t = torch.from_numpy(G).unsqueeze(0).float()

        # Input: [I, M, G] or [I, M]
        x = torch.cat([A_t, M_t, G_t], dim=0)  # 3 + 1 + 1 = 5

        return {
            "oid": oid,
            "x": x,            # (5,H,W)
            "I_in": A_t,       # (3,H,W)
            "I_gt": D_t,       # (3,H,W)
            "M": M_t,          # (1,H,W)
            "G": G_t,          # (1,H,W)
        }
