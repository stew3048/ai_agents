"""
將 skyfinder_3888 根目錄的 3888_mask.png 複製成對應所有 images 的 masks

邏輯：
1. 讀取 images/ 資料夾中的所有圖片檔名
2. 將 3888_mask.png 複製成對應的 mask 檔案（例如 images/001.jpg → masks/001.png）
"""

import os
import shutil
from pathlib import Path
from PIL import Image

def prepare_3888_masks():
    """準備 3888 的 masks"""
    camera_folder = Path("data/skyfinder_3888")
    images_dir = camera_folder / "images"
    masks_dir = camera_folder / "masks"
    template_mask = camera_folder / "3888_mask.png"
    
    if not template_mask.exists():
        print(f"[錯誤] 找不到 template mask: {template_mask}")
        return
    
    if not images_dir.exists():
        print(f"[錯誤] 找不到 images 資料夾: {images_dir}")
        return
    
    # 確保 masks 資料夾存在
    masks_dir.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("  準備 3888 的 masks")
    print("=" * 60)
    print(f"  Template mask: {template_mask}")
    print(f"  Images 目錄: {images_dir}")
    print(f"  Masks 目錄: {masks_dir}")
    print()
    
    # 讀取所有圖片檔名
    image_extensions = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    image_files = sorted([
        f for f in images_dir.iterdir() 
        if f.is_file() and f.suffix in image_extensions
    ], key=lambda x: x.name)
    
    if not image_files:
        print("[錯誤] 找不到任何 image 檔案")
        return
    
    print(f"  找到 {len(image_files)} 張圖片")
    
    # 載入 template mask
    print(f"  載入 template mask: {template_mask.name}")
    template_img = Image.open(template_mask).convert("L")
    
    # 為每張圖片產生對應的 mask
    created_count = 0
    for img_file in image_files:
        # 取得圖片檔名（不含副檔名）
        img_stem = img_file.stem  # 例如 "001"
        mask_name = f"{img_stem}.png"
        mask_path = masks_dir / mask_name
        
        # 如果 mask 已存在，跳過
        if mask_path.exists():
            continue
        
        # 複製 template mask
        template_img.save(mask_path)
        created_count += 1
        
        if created_count % 100 == 0:
            print(f"  已處理 {created_count}/{len(image_files)} 張...")
    
    print(f"  完成！建立了 {created_count} 個 mask 檔案")
    print(f"  Masks 目錄現在有 {len(list(masks_dir.glob('*.png')))} 個檔案")
    print()


if __name__ == "__main__":
    prepare_3888_masks()
