"""
CV Baseline 視覺化驗證
使用傳統 CV 方法對 toy dataset 進行天空分割，並輸出 overlay 視覺化結果

輸出（每個 sample 3 張圖）：
1. XXX_1_image.png      - 原始圖片
2. XXX_2_pred_mask.png  - CV baseline 預測的 mask（白=天空）
3. XXX_3_pred_overlay.png - 預測 mask 疊在原圖上（紅色=預測為天空）

用途：
- 肉眼觀察 CV baseline 把哪些地方當成天空
- 理解 rule-based 方法的優缺點
"""

import os
import sys
import numpy as np
from PIL import Image

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.dataset import get_dataloader
from src.models.cv_baseline import predict_sky_mask_cv, get_default_params


def create_overlay(image_np, mask_np, color=(255, 50, 50), alpha=0.5):
    """
    建立 overlay：將 mask 區域用半透明顏色疊在原圖上
    
    Args:
        image_np: numpy array [H, W, 3], uint8, range [0, 255]
        mask_np: numpy array [H, W], values {0, 1}
        color: RGB tuple for overlay color (default: red)
        alpha: transparency (0=transparent, 1=opaque)
    
    Returns:
        overlay_np: numpy array [H, W, 3], uint8
    """
    # 確保 image 是 float 方便計算
    image_float = image_np.astype(np.float32)
    
    # 建立純色 overlay
    overlay = np.zeros_like(image_float)
    overlay[:, :, 0] = color[0]
    overlay[:, :, 1] = color[1]
    overlay[:, :, 2] = color[2]
    
    # 將 mask 擴展成 3 通道
    mask_3ch = np.stack([mask_np] * 3, axis=-1).astype(np.float32)
    
    # 混合：只在 mask=1 的地方疊加顏色
    result = image_float * (1 - mask_3ch * alpha) + overlay * (mask_3ch * alpha)
    
    return result.clip(0, 255).astype(np.uint8)


def tensor_to_numpy_uint8(tensor):
    """
    將 torch tensor [C, H, W] 轉換為 numpy [H, W, 3] uint8
    """
    # [C, H, W] -> [H, W, C]
    arr = tensor.permute(1, 2, 0).numpy()
    # [0, 1] -> [0, 255]
    arr = (arr * 255).clip(0, 255).astype(np.uint8)
    return arr


def main():
    # === 設定 ===
    images_dir = 'data/toy/images'
    masks_dir = 'data/toy/masks'
    output_dir = 'outputs/cv_toy_vis'
    num_samples = 10  # 處理前 10 張
    
    os.makedirs(output_dir, exist_ok=True)
    
    print('=' * 60)
    print('  CV Baseline on Toy Dataset')
    print('=' * 60)
    print()
    
    # 顯示 CV baseline 參數
    params = get_default_params()
    print('[CV Baseline Parameters]')
    for k, v in params.items():
        print(f'  {k}: {v}')
    print()
    
    # === 載入 DataLoader ===
    print('[1] Loading DataLoader...')
    dataloader = get_dataloader(
        images_dir=images_dir,
        masks_dir=masks_dir,
        batch_size=num_samples,
        shuffle=False,
        transform=False  # 不做 augmentation
    )
    
    # 取得一個 batch
    images, gt_masks = next(iter(dataloader))
    print(f'    Loaded {images.shape[0]} images')
    print(f'    Image shape: {images.shape}')
    print(f'    GT mask shape: {gt_masks.shape}')
    print()
    
    # === 對每張圖執行 CV baseline ===
    print('[2] Running CV baseline...')
    print('-' * 60)
    
    for idx in range(num_samples):
        sample_id = f'{idx+1:03d}'
        
        # 取得單張 image tensor
        image_tensor = images[idx]  # [C, H, W]
        gt_mask_tensor = gt_masks[idx]  # [1, H, W]
        
        # 轉換成 numpy uint8 供顯示
        image_np = tensor_to_numpy_uint8(image_tensor)  # [H, W, 3]
        
        # === 執行 CV baseline ===
        pred_mask = predict_sky_mask_cv(image_tensor)  # [H, W], {0, 1}
        
        # === 建立 overlay ===
        # 用紅色標示預測為天空的區域
        pred_overlay = create_overlay(image_np, pred_mask, color=(255, 50, 50), alpha=0.5)
        
        # === 儲存結果 ===
        # 1. 原始圖片
        Image.fromarray(image_np).save(
            os.path.join(output_dir, f'{sample_id}_1_image.png')
        )
        
        # 2. 預測 mask（轉成 0/255 方便顯示）
        pred_mask_vis = (pred_mask * 255).astype(np.uint8)
        Image.fromarray(pred_mask_vis, mode='L').save(
            os.path.join(output_dir, f'{sample_id}_2_pred_mask.png')
        )
        
        # 3. 預測 overlay
        Image.fromarray(pred_overlay).save(
            os.path.join(output_dir, f'{sample_id}_3_pred_overlay.png')
        )
        
        # 統計
        sky_ratio = pred_mask.sum() / pred_mask.size * 100
        print(f'    Sample {sample_id}: pred sky ratio = {sky_ratio:.1f}%')
    
    print('-' * 60)
    print()
    
    # === 輸出摘要 ===
    print('[3] Output files:')
    print(f'    Directory: {output_dir}/')
    print()
    print('    For each sample XXX:')
    print('    -------------------------------------------------------')
    print('    XXX_1_image.png       : Original image')
    print('    XXX_2_pred_mask.png   : Predicted mask (white=sky)')
    print('    XXX_3_pred_overlay.png: Prediction overlay (red=predicted sky)')
    print()
    print('=' * 60)
    print()
    
    # === 觀察指南 ===
    print('=' * 60)
    print('  HOW TO INTERPRET THE RESULTS')
    print('=' * 60)
    print()
    print('[What to look for in pred_overlay]')
    print('  - RED area = pixels that CV baseline thinks is SKY')
    print('  - Non-red area = pixels classified as NON-SKY')
    print()
    print('[Expected behavior]')
    print('  - Blue sky regions should be marked RED')
    print('  - Ground/trees/buildings should NOT be red')
    print()
    print('[Known limitations of this method]')
    print('  1. Only detects BLUE sky (misses sunset, cloudy sky)')
    print('  2. May miss sky if too dark or too desaturated')
    print('  3. May false-positive on blue objects (water, blue walls)')
    print()
    print('=' * 60)


if __name__ == '__main__':
    main()
