"""
Resize 後的 overlay 驗收腳本
驗證 resize 後 image 與 mask 是否正確對齊
"""

import os
import numpy as np
from PIL import Image


def create_resized_overlay(image_path, mask_path, output_path, target_size=(256, 256), alpha=0.4):
    """
    將 image 和 mask resize 後產生 overlay
    
    參數:
        image_path: 原始圖片路徑
        mask_path: mask 路徑
        output_path: 輸出路徑
        target_size: 目標尺寸 (width, height)
        alpha: 遮罩透明度
    """
    # 讀取圖片
    image = Image.open(image_path).convert('RGB')
    mask = Image.open(mask_path).convert('L')
    
    original_image_size = image.size  # (width, height)
    original_mask_size = mask.size
    
    # Resize image - 使用 BILINEAR（與訓練時一致）
    image_resized = image.resize(target_size, Image.BILINEAR)
    
    # Resize mask - 必須使用 NEAREST（保持二進制值）
    mask_resized = mask.resize(target_size, Image.NEAREST)
    
    # 轉換為 numpy
    img_array = np.array(image_resized, dtype=np.float32)
    mask_array = np.array(mask_resized, dtype=np.float32) / 255.0
    
    # 建立 overlay
    overlay = img_array.copy()
    
    # 天空區域塗上綠色
    sky_mask = mask_array > 0.5
    overlay[sky_mask, 0] = overlay[sky_mask, 0] * (1 - alpha) + 0 * alpha
    overlay[sky_mask, 1] = overlay[sky_mask, 1] * (1 - alpha) + 255 * alpha
    overlay[sky_mask, 2] = overlay[sky_mask, 2] * (1 - alpha) + 0 * alpha
    
    # 畫邊界紅線
    for x in range(mask_array.shape[1]):
        for y in range(1, mask_array.shape[0]):
            if mask_array[y, x] != mask_array[y-1, x]:
                for dy in range(-2, 3):
                    if 0 <= y + dy < mask_array.shape[0]:
                        overlay[y + dy, x] = [255, 0, 0]
    
    # 儲存
    overlay_image = Image.fromarray(overlay.astype(np.uint8))
    overlay_image.save(output_path)
    
    return {
        'original_image_size': original_image_size,
        'original_mask_size': original_mask_size,
        'resized_size': target_size,
        'mask_unique_before': np.unique(np.array(mask)),
        'mask_unique_after': np.unique(np.array(mask_resized))
    }


def main():
    # 設定
    images_dir = 'data/train/images'
    masks_dir = 'data/train/masks'
    output_dir = 'data/train/resized_overlay'
    target_size = (256, 256)
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("Resize 後的 Overlay 驗收")
    print("=" * 60)
    print(f"目標尺寸: {target_size[0]} x {target_size[1]}")
    print(f"Image interpolation: BILINEAR")
    print(f"Mask interpolation: NEAREST (保持二進制)")
    print()
    
    # 取得所有圖片
    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.png')])
    
    print(f"處理 {len(image_files)} 張圖片...")
    print()
    
    for i, image_file in enumerate(image_files):
        image_path = os.path.join(images_dir, image_file)
        mask_path = os.path.join(masks_dir, image_file)
        output_path = os.path.join(output_dir, image_file)
        
        if not os.path.exists(mask_path):
            print(f"[SKIP] {image_file} - 找不到 mask")
            continue
        
        info = create_resized_overlay(image_path, mask_path, output_path, target_size)
        
        print(f"[{i+1}/{len(image_files)}] {image_file}")
        print(f"      Original image: {info['original_image_size'][0]} x {info['original_image_size'][1]}")
        print(f"      Original mask:  {info['original_mask_size'][0]} x {info['original_mask_size'][1]}")
        print(f"      Resized to:     {info['resized_size'][0]} x {info['resized_size'][1]}")
        print(f"      Mask values before resize: {set(info['mask_unique_before'])}")
        print(f"      Mask values after resize:  {set(info['mask_unique_after'])}")
        
        # 檢查 mask 值是否仍為二進制
        if not set(info['mask_unique_after']).issubset({0, 255}):
            print(f"      [WARN] Mask 值在 resize 後不是二進制！")
        else:
            print(f"      [OK]")
        print()
    
    print("=" * 60)
    print(f"[OK] 完成！Resized overlay 儲存在: {output_dir}")
    print("=" * 60)
    print()
    print("=" * 60)
    print("如何判斷對齊是否正確")
    print("=" * 60)
    print()
    print("【正確對齊的特徵】")
    print("  1. 紅色邊界線精準貼合天空與地面的交界")
    print("  2. 綠色遮罩完整覆蓋天空區域，不會超出或不足")
    print("  3. 雲朵區域也被綠色覆蓋（因為雲是天空的一部分）")
    print("  4. 地面區域完全保持原色，沒有任何綠色")
    print()
    print("【錯誤對齊的特徵（3種常見問題）】")
    print()
    print("  1. [水平/垂直偏移]")
    print("     - 現象：紅線與實際邊界有明顯的水平或垂直距離")
    print("     - 原因：image 和 mask 來自不同圖片，或處理時發生偏移")
    print()
    print("  2. [縮放比例不一致]")
    print("     - 現象：mask 覆蓋範圍比例與圖片不符")
    print("            例如：圖片天空佔 50%，但綠色只覆蓋 30%")
    print("     - 原因：image 和 mask 原始尺寸不同，resize 時比例錯誤")
    print()
    print("  3. [邊界鋸齒/破碎]")
    print("     - 現象：邊界線呈現明顯鋸齒狀，或 mask 有破碎的孔洞")
    print("     - 原因：mask resize 時使用了 BILINEAR 而非 NEAREST")
    print("            導致中間值（如 127）出現，二進制化後產生錯誤")
    print()


if __name__ == '__main__':
    main()
