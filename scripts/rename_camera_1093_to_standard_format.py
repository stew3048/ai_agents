"""
將 camera 1093 的 image 和 mask 重新命名為標準格式（001.jpg, 002.jpg...）

功能：
1. 將 images 重新命名為 001.jpg, 002.jpg, 003.jpg...
2. 將對應的 masks 重新命名為 001.png, 002.png, 003.png...
3. 確保 image 和 mask 的編號對應
"""

import os
from pathlib import Path
from PIL import Image


def rename_to_standard_format(camera_folder="data/skyfinder_1093"):
    """
    將 camera 1093 的檔案重新命名為標準格式
    """
    camera_path = Path(camera_folder)
    images_dir = camera_path / "images"
    masks_dir = camera_path / "masks"
    
    if not images_dir.exists() or not masks_dir.exists():
        print(f"[錯誤] 找不到 images/ 或 masks/ 資料夾")
        return False
    
    print("=" * 60)
    print(f"  將 Camera 1093 重新命名為標準格式")
    print("=" * 60)
    print()
    
    # 1. 收集所有 image 和 mask，並配對
    print("  [1] 收集並配對 image 和 mask...")
    image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    
    images = sorted([f for f in images_dir.iterdir() 
                    if f.is_file() and f.suffix.lower() in image_extensions],
                   key=lambda x: x.name)
    masks = sorted([f for f in masks_dir.iterdir() 
                   if f.is_file() and f.suffix.lower() == '.png'],
                  key=lambda x: x.name)
    
    print(f"      找到 {len(images)} 張 image")
    print(f"      找到 {len(masks)} 個 mask")
    
    # 配對：根據檔名（不含副檔名）配對
    image_dict = {img.stem: img for img in images}
    mask_dict = {mask.stem: mask for mask in masks}
    
    paired = []
    for img_stem, img_file in image_dict.items():
        if img_stem in mask_dict:
            paired.append((img_file, mask_dict[img_stem]))
        else:
            print(f"  [警告] Image {img_file.name} 沒有對應的 mask")
    
    print(f"      成功配對: {len(paired)} 組")
    print()
    
    if len(paired) == 0:
        print("[錯誤] 沒有成功配對的檔案")
        return False
    
    # 2. 重新命名為標準格式（001.jpg, 001.png...）
    print("  [2] 重新命名為標準格式...")
    
    # 先創建臨時目錄避免名稱衝突
    temp_images_dir = images_dir.parent / "images_temp"
    temp_masks_dir = masks_dir.parent / "masks_temp"
    temp_images_dir.mkdir(exist_ok=True)
    temp_masks_dir.mkdir(exist_ok=True)
    
    renamed_count = 0
    for idx, (img_file, mask_file) in enumerate(paired, start=1):
        # 新檔名：001.jpg, 002.jpg...
        new_img_name = f"{idx:03d}.jpg"
        new_mask_name = f"{idx:03d}.png"
        
        # 先移動到臨時目錄
        temp_img_path = temp_images_dir / new_img_name
        temp_mask_path = temp_masks_dir / new_mask_name
        
        shutil.move(str(img_file), str(temp_img_path))
        shutil.move(str(mask_file), str(temp_mask_path))
        
        renamed_count += 1
    
    print(f"      重新命名了 {renamed_count} 組檔案")
    print()
    
    # 3. 刪除舊的 images 和 masks 目錄，然後將臨時目錄重新命名
    print("  [3] 更新目錄結構...")
    
    # 刪除舊目錄中剩餘的檔案
    for img_file in images_dir.iterdir():
        if img_file.is_file():
            img_file.unlink()
    for mask_file in masks_dir.iterdir():
        if mask_file.is_file():
            mask_file.unlink()
    
    # 將臨時目錄的檔案移回
    for temp_img in temp_images_dir.iterdir():
        shutil.move(str(temp_img), str(images_dir / temp_img.name))
    for temp_mask in temp_masks_dir.iterdir():
        shutil.move(str(temp_mask), str(masks_dir / temp_mask.name))
    
    # 刪除臨時目錄
    temp_images_dir.rmdir()
    temp_masks_dir.rmdir()
    
    print("      目錄結構更新完成")
    print()
    
    # 4. 驗證最終結果
    print("  [4] 驗證最終結果...")
    final_images = sorted([f for f in images_dir.iterdir() 
                          if f.is_file() and f.suffix.lower() in image_extensions],
                         key=lambda x: x.name)
    final_masks = sorted([f for f in masks_dir.iterdir() 
                        if f.is_file() and f.suffix.lower() == '.png'],
                       key=lambda x: x.name)
    
    print(f"      最終 image 數: {len(final_images)}")
    print(f"      最終 mask 數: {len(final_masks)}")
    
    # 檢查命名是否正確
    all_match = True
    for i, (img, mask) in enumerate(zip(final_images, final_masks), start=1):
        expected_img = f"{i:03d}.jpg"
        expected_mask = f"{i:03d}.png"
        if img.name != expected_img or mask.name != expected_mask:
            print(f"  [錯誤] 編號 {i} 不匹配: image={img.name}, mask={mask.name}")
            all_match = False
    
    if all_match:
        print("      [OK] 所有檔案命名正確")
    
    print()
    print("=" * 60)
    print("  重新命名完成！")
    print("=" * 60)
    print()
    print(f"  最終統計:")
    print(f"    Image: {len(final_images)} 張")
    print(f"    Mask: {len(final_masks)} 張")
    print(f"    命名格式: 001.jpg/001.png, 002.jpg/002.png, ...")
    
    return True


if __name__ == "__main__":
    import shutil
    rename_to_standard_format("data/skyfinder_1093")
