"""
用 DINOv2 找「視覺上最像天空」的區域，產出 SAM 的 point prompt。

流程：圖 → DINOv2 patch 特徵 → 以圖上方 patch 平均當 sky prototype →
      每 patch 與 prototype 的相似度 → 取最高者之中心當一點 → 回傳 (point, labels)。

用法：由 eval_sam2_failure_cases.py 呼叫，或獨立測試。
"""

import os
import sys
import numpy as np
import torch

# 專案根目錄
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# DINOv2 常用輸入尺寸（patch_size=14 時 518/14≈37）
DINO_INPUT_SIZE = 518
# 圖上方多少比例當「天空」取樣做 prototype（約上方 25%）
SKY_TOP_RATIO = 0.25
# ImageNet 常規 normalize
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

DINO_AVAILABLE = False
_dino_model = None


def _load_dino_model(device='cuda'):
    """載入 DINOv2 ViT-B/14（torch.hub）。"""
    global DINO_AVAILABLE, _dino_model
    if _dino_model is not None:
        return _dino_model
    try:
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14', trust_repo=True)
        model = model.to(device)
        model.eval()
        _dino_model = model
        DINO_AVAILABLE = True
        return model
    except Exception as e:
        raise RuntimeError(f"無法載入 DINOv2: {e}") from e


def _preprocess_for_dino(image_np, size=DINO_INPUT_SIZE):
    """image_np: (H,W,3) uint8. 回傳 (1,3,size,size) tensor，已 normalize。"""
    from PIL import Image
    h, w = image_np.shape[:2]
    img = Image.fromarray(image_np).resize((size, size), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)
    return tensor


def get_dino_patch_tokens(model, image_tensor, device):
    """
    image_tensor: (1, 3, 518, 518).
    回傳 patch tokens (1, N, D)，且 N = 37*37（ViT-B/14）。
    """
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        out = model.forward_features(image_tensor)
    # out 可能是 dict 或 tuple，依 hub 回傳
    if isinstance(out, dict):
        tokens = out.get("x_norm_patchtokens")
        if tokens is None:
            tokens = out.get("x_prenorm")
            if tokens is not None:
                tokens = tokens[:, 1:, :]
    else:
        tokens = out[:, 1:, :] if out.dim() == 3 else out
    return tokens


def get_sky_point_from_dino(image_np, device='cuda', dino_model=None):
    """
    用 DINOv2 找「最像天空」的 patch，回傳該 patch 中心在「原圖」上的座標 (x, y)。

    image_np: (H, W, 3) uint8，原圖。
    回傳: (x, y) 整數，原圖座標系，可直接當 SAM point。
    """
    if dino_model is None:
        dino_model = _load_dino_model(device)
    orig_h, orig_w = image_np.shape[:2]
    x = _preprocess_for_dino(image_np, DINO_INPUT_SIZE)
    tokens = get_dino_patch_tokens(dino_model, x, device)
    # tokens: (1, N, D)
    if tokens.dim() != 3:
        raise ValueError(f"Expected patch tokens (1, N, D), got {tokens.shape}")
    n = tokens.shape[1]
    # ViT-B/14 @ 518x518 -> 37*37 = 1369
    grid = int(np.sqrt(n) + 0.5)
    if grid * grid != n:
        grid = int(np.sqrt(n))
        if grid * grid != n:
            raise ValueError(f"Patch count {n} is not a perfect square.")
    # (1, N, D) -> (N, D)
    patches = tokens[0].cpu().numpy()
    # 上方 SKY_TOP_RATIO 的 patch 當 sky prototype
    top_rows = max(1, int(grid * SKY_TOP_RATIO))
    sky_indices = []
    for i in range(top_rows):
        for j in range(grid):
            sky_indices.append(i * grid + j)
    prototype = np.mean(patches[sky_indices], axis=0, dtype=np.float32)
    prototype = prototype / (np.linalg.norm(prototype) + 1e-8)
    # 每個 patch 與 prototype 的 cosine similarity
    patches_norm = patches / (np.linalg.norm(patches, axis=1, keepdims=True) + 1e-8)
    sim = patches_norm @ prototype
    best_idx = int(np.argmax(sim))
    best_i, best_j = best_idx // grid, best_idx % grid
    # patch 中心在 518 圖上的座標（patch_size=14）
    patch_size = DINO_INPUT_SIZE // grid
    cx_518 = best_j * patch_size + patch_size // 2
    cy_518 = best_i * patch_size + patch_size // 2
    # 縮放回原圖
    x_orig = int(cx_518 * orig_w / DINO_INPUT_SIZE)
    y_orig = int(cy_518 * orig_h / DINO_INPUT_SIZE)
    x_orig = max(0, min(orig_w - 1, x_orig))
    y_orig = max(0, min(orig_h - 1, y_orig))
    return x_orig, y_orig


def get_prompt_from_dino(image_np, device='cuda', dino_model=None):
    """
    回傳可傳給 SAM 的 (prompt_type, prompt_data)。
    一律回傳單點：視覺上最像天空的點。
    """
    x, y = get_sky_point_from_dino(image_np, device=device, dino_model=dino_model)
    points = np.array([[x, y]], dtype=np.int64)
    labels = np.array([1], dtype=np.int32)  # 前景
    return 'point', (points, labels)


if __name__ == '__main__':
    # 簡單測試：載入一張圖、輸出 DINO 給的 sky 點
    import argparse
    from PIL import Image

    parser = argparse.ArgumentParser()
    parser.add_argument('image', nargs='?', default=None)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    if not args.image or not os.path.exists(args.image):
        print("Usage: python dino_sky_prompt.py <path_to_image.jpg>")
        sys.exit(1)
    img = np.array(Image.open(args.image).convert('RGB'))
    x, y = get_sky_point_from_dino(img, device=args.device)
    print(f"DINO sky point (orig image coords): x={x}, y={y}")
