"""
影像工具：sRGB / 亮度等。
"""

import torch
import numpy as np


def srgb_to_linear(v):
    """
    sRGB [0,1]（gamma 編碼）→ linear [0,1]。
    係數 0.2126/0.7152/0.0722 的 luma 公式必須用在 linear RGB，直接用在 sRGB 會高估暗部。
    """
    if isinstance(v, torch.Tensor):
        return torch.where(
            v <= 0.04045,
            v / 12.92,
            torch.pow((v + 0.055) / 1.055, 2.4)
        )
    v = np.clip(np.asarray(v, dtype=np.float64), 0, 1)
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4).astype(np.float32)


def mean_luma_linear(image_tensor):
    """
    從 [C,H,W]、值域 [0,1] 的 sRGB 影像計算整張的 mean luma（linear）。
    luma = 0.2126*R_lin + 0.7152*G_lin + 0.0722*B_lin，其中 R_lin 等為 sRGB→linear 後的數值。
    回傳標量 float。
    """
    R, G, B = image_tensor[0], image_tensor[1], image_tensor[2]
    Rl = srgb_to_linear(R)
    Gl = srgb_to_linear(G)
    Bl = srgb_to_linear(B)
    luma = 0.2126 * Rl + 0.7152 * Gl + 0.0722 * Bl
    return luma.mean().item()
