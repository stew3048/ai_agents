"""
驗證 SkyFinder 資料的 image 和 mask 對齊
產生 overlay 圖片讓使用者肉眼確認

輸出：
- 原始圖片
- mask（黑白）
- overlay（天空區域用半透明藍色標示）
"""

import os
import numpy as np
from PIL import Image

# 設定
IMAGES_DIR = 'data/skyfinder_sample/images'
MASKS_DIR = 'data/skyfinder_sample/masks'
OUTPUT_DIR = 'outputs/skyfinder_alignment_check'

# 要檢查的樣本數
NUM_SAMPLES = 10


def create_overlay(image, mask, color=(0, 100, 255), alpha=0.5):
    """
    建立 overlay 圖片
    天空區域（mask=1）用半透明顏色標示
    """
    # 確保 image 是 RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # 確保 mask 是 L (grayscale)
    if mask.mode != 'L':
        mask = mask.convert('L')
    
    # 轉換為 numpy array
    img_array = np.array(image).astype(np.float32)
    mask_array = np.array(mask)
    
    # 建立 mask 的 boolean array（天空 = 白色 = 255 或 1）
    sky_mask = mask_array > 127
    
    # 建立 overlay
    overlay = img_array.copy()
    
    # 在天空區域加上半透明顏色
    overlay[sky_mask] = overlay[sky_mask] * (1 - alpha) + np.array(color) * alpha
    
    return Image.fromarray(overlay.clip(0, 255).astype(np.uint8))


def main():
    print('=' * 60)
    print('  SkyFinder Image-Mask Alignment Verification')
    print('=' * 60)
    print()
    
    # 建立輸出目錄
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 取得圖片列表
    image_files = sorted([f for f in os.listdir(IMAGES_DIR) if f.endswith(('.jpg', '.png'))])
    
    print(f'  Found {len(image_files)} images')
    print(f'  Checking first {NUM_SAMPLES} samples')
    print()
    
    # 檢查每張圖片
    for i, img_file in enumerate(image_files[:NUM_SAMPLES]):
        sample_id = img_file.split('.')[0]
        
        # 載入圖片
        img_path = os.path.join(IMAGES_DIR, img_file)
        image = Image.open(img_path)
        
        # 找對應的 mask
        mask_file = f'{sample_id}.png'
        mask_path = os.path.join(MASKS_DIR, mask_file)
        
        if not os.path.exists(mask_path):
            print(f'  [{sample_id}] Warning: Mask not found!')
            continue
        
        mask = Image.open(mask_path)
        
        # 檢查尺寸
        img_size = image.size
        mask_size = mask.size
        
        size_match = img_size == mask_size
        
        print(f'  [{sample_id}] Image: {img_size[0]}x{img_size[1]} | Mask: {mask_size[0]}x{mask_size[1]} | Match: {"YES" if size_match else "NO!"}')
        
        # 如果尺寸不同，調整 mask 到圖片尺寸
        if not size_match:
            print(f'           Resizing mask to match image...')
            mask = mask.resize(img_size, Image.NEAREST)
        
        # 建立 overlay
        overlay = create_overlay(image, mask, color=(0, 100, 255), alpha=0.4)
        
        # 儲存三張圖
        # 1. 原始圖片
        image.save(os.path.join(OUTPUT_DIR, f'{sample_id}_1_image.png'))
        
        # 2. Mask
        mask.save(os.path.join(OUTPUT_DIR, f'{sample_id}_2_mask.png'))
        
        # 3. Overlay
        overlay.save(os.path.join(OUTPUT_DIR, f'{sample_id}_3_overlay.png'))
    
    print()
    print('=' * 60)
    print('  Verification Complete!')
    print('=' * 60)
    print()
    print(f'  Output saved to: {OUTPUT_DIR}/')
    print()
    print('  How to verify alignment:')
    print('    1. Open the overlay images')
    print('    2. Blue area = sky region (from mask)')
    print('    3. Check if blue covers EXACTLY the sky')
    print()
    print('  Correct alignment:')
    print('    - Blue covers all sky pixels')
    print('    - Blue does NOT cover ground/buildings')
    print('    - Edges are sharp and match object boundaries')
    print()
    print('  Incorrect alignment:')
    print('    - Blue shifted left/right/up/down')
    print('    - Blue covers wrong areas')
    print('    - Edges don\'t match')
    print()
    print('=' * 60)


if __name__ == '__main__':
    main()
