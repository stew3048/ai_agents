"""
建立圖片與 mask 的疊加圖
用於視覺化確認 mask 是否正確
"""

import os
import numpy as np
from PIL import Image


def create_overlay(image_path, mask_path, output_path, alpha=0.5):
    """
    將圖片與 mask 疊加
    
    參數:
        image_path: 原始圖片路徑
        mask_path: mask 路徑
        output_path: 輸出路徑
        alpha: mask 透明度 (0-1)
    """
    # 讀取圖片
    image = Image.open(image_path).convert('RGB')
    mask = Image.open(mask_path).convert('L')
    
    # 確保尺寸一致
    if image.size != mask.size:
        mask = mask.resize(image.size, Image.NEAREST)
    
    # 轉換為 numpy 陣列
    img_array = np.array(image, dtype=np.float32)
    mask_array = np.array(mask, dtype=np.float32) / 255.0  # 正規化到 0-1
    
    # 建立遮罩層
    overlay = img_array.copy()
    
    # 天空區域（mask > 0.5）塗上綠色半透明
    sky_mask = mask_array > 0.5
    overlay[sky_mask, 0] = overlay[sky_mask, 0] * (1 - alpha) + 0 * alpha    # R
    overlay[sky_mask, 1] = overlay[sky_mask, 1] * (1 - alpha) + 255 * alpha  # G
    overlay[sky_mask, 2] = overlay[sky_mask, 2] * (1 - alpha) + 0 * alpha    # B
    
    # 在天空與地面的邊界畫一條紅線（更容易看出分界）
    # 找出邊界
    for x in range(mask_array.shape[1]):
        for y in range(1, mask_array.shape[0]):
            if mask_array[y, x] != mask_array[y-1, x]:
                # 這是邊界，畫紅線（上下各2像素）
                for dy in range(-2, 3):
                    if 0 <= y + dy < mask_array.shape[0]:
                        overlay[y + dy, x] = [255, 0, 0]  # 紅色邊界線
    
    # 轉換回 PIL Image
    overlay_image = Image.fromarray(overlay.astype(np.uint8))
    
    # 儲存
    overlay_image.save(output_path)
    

def main():
    # 設定路徑
    train_images_dir = 'data/train/images'
    train_masks_dir = 'data/train/masks'
    output_dir = 'data/train/overlay'
    
    # 建立輸出目錄
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("建立圖片與 Mask 疊加圖")
    print("=" * 60)
    print(f"天空區域會以紅色半透明顯示")
    print()
    
    # 取得所有圖片
    image_files = [f for f in os.listdir(train_images_dir) if f.endswith('.png')]
    
    for image_file in sorted(image_files):
        image_path = os.path.join(train_images_dir, image_file)
        mask_path = os.path.join(train_masks_dir, image_file)
        output_path = os.path.join(output_dir, image_file)
        
        if os.path.exists(mask_path):
            create_overlay(image_path, mask_path, output_path, alpha=0.4)
            print(f"[OK] {image_file}")
        else:
            print(f"[SKIP] {image_file} - 找不到對應的 mask")
    
    print()
    print("=" * 60)
    print(f"[OK] 完成！疊加圖儲存在: {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
