"""
CLAHE 預處理（dino_clahm 推論用）

用於大霧/低對比場景：BGR -> LAB -> 僅對 L channel 做 CLAHE -> 合併 -> RGB -> ImageNet Normalize -> Tensor。
供 DINOv2 類模型推論前使用。
"""

import cv2
import numpy as np
import torch
from torchvision import transforms

# DINOv2 常用 ImageNet normalize
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def create_clahe(clip_limit=2.0, tile_grid_size=(8, 8)):
    """建立 CLAHE 物件。"""
    return cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)


def bgr_to_rgb_clahe(
    bgr: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple = (8, 8),
) -> np.ndarray:
    """
    BGR 圖 -> LAB -> 僅對 L 做 CLAHE -> 合併 -> RGB（uint8）。

    Args:
        bgr: (H, W, 3) uint8，OpenCV BGR
        clip_limit: CLAHE clipLimit
        tile_grid_size: CLAHE tileGridSize

    Returns:
        rgb: (H, W, 3) uint8，RGB
    """
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    clahe = create_clahe(clip_limit=clip_limit, tile_grid_size=tile_grid_size)
    l_ch, a, b = cv2.split(lab)
    l_ch = clahe.apply(l_ch)
    lab = cv2.merge([l_ch, a, b])
    # OpenCV 僅有 LAB2BGR，再轉 BGR -> RGB
    bgr_out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    rgb = cv2.cvtColor(bgr_out, cv2.COLOR_BGR2RGB)
    return rgb


def bgr_to_tensor_clahm_normalized(
    bgr: np.ndarray,
    image_size: tuple,
    device: torch.device,
    clip_limit: float = 2.0,
    tile_grid_size: tuple = (8, 8),
    mean: list = None,
    std: list = None,
) -> torch.Tensor:
    """
    BGR 圖 -> CLAHE(L) -> RGB -> [0,1] -> Resize -> ImageNet Normalize -> Tensor。

    Args:
        bgr: (H, W, 3) uint8，OpenCV BGR
        image_size: (H, W) 目標尺寸，與訓練一致（如 (160, 160)）
        device: torch device
        clip_limit: CLAHE clipLimit
        tile_grid_size: CLAHE tileGridSize
        mean: Normalize mean，預設 ImageNet
        std: Normalize std，預設 ImageNet

    Returns:
        tensor: (1, 3, image_size[0], image_size[1])，已 normalize，在 device 上
    """
    if mean is None:
        mean = IMAGENET_MEAN
    if std is None:
        std = IMAGENET_STD
    rgb = bgr_to_rgb_clahe(bgr, clip_limit=clip_limit, tile_grid_size=tile_grid_size)
    # [0, 255] -> [0, 1] float
    rgb_float = rgb.astype(np.float32) / 255.0
    tensor = torch.from_numpy(rgb_float).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
    if tensor.shape[2:] != image_size:
        tensor = torch.nn.functional.interpolate(
            tensor, size=image_size, mode='bilinear', align_corners=False
        )
    normalize = transforms.Normalize(mean=mean, std=std)
    tensor = normalize(tensor).to(device)
    return tensor
