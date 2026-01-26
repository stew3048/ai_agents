"""
整理 data/1093 資料夾結構，並分析 day/night 分布

功能：
1. 將 1093 資料夾整理成 skyfinder_* 格式（images/ 和 masks/）
2. 重新命名為 skyfinder_1093
3. 分析 day/night 分布（使用與訓練/評估一致的 luma 計算）
"""

import os
import shutil
from pathlib import Path
from PIL import Image
import numpy as np


def srgb_to_linear(v):
    """sRGB 轉 linear RGB"""
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


def calculate_mean_luma(image_path):
    """
    計算圖片的平均 luma（與訓練/評估一致）
    
    使用：
    - sRGB → linear RGB 轉換
    - ITU-R BT.709 係數：0.2126*R + 0.7152*G + 0.0722*B
    - 閾值：luma < 0.20 為 night
    """
    try:
        img = Image.open(image_path).convert('RGB')
        img_array = np.array(img, dtype=np.float32) / 255.0  # [H, W, 3], [0, 1]
        
        # sRGB → linear RGB
        R_lin = srgb_to_linear(img_array[:, :, 0])
        G_lin = srgb_to_linear(img_array[:, :, 1])
        B_lin = srgb_to_linear(img_array[:, :, 2])
        
        # 計算 luma (ITU-R BT.709)
        luma = 0.2126 * R_lin + 0.7152 * G_lin + 0.0722 * B_lin
        mean_luma = np.mean(luma)
        
        return mean_luma
    except Exception as e:
        print(f"  錯誤：無法讀取 {image_path}: {e}")
        return None


def organize_camera_folder(camera_folder="data/1093", target_name="skyfinder_1093"):
    """
    整理 camera 資料夾結構
    
    步驟：
    1. 創建 images/ 資料夾
    2. 移動所有圖片到 images/
    3. 將 mask/ 重新命名為 masks/
    4. 重新命名整個資料夾為 skyfinder_1093
    """
    camera_path = Path(camera_folder)
    if not camera_path.exists():
        print(f"[錯誤] 找不到資料夾: {camera_path}")
        return False
    
    print("=" * 60)
    print(f"  整理 Camera 資料夾: {camera_folder}")
    print("=" * 60)
    print()
    
    # 1. 創建 images/ 資料夾
    images_dir = camera_path / "images"
    images_dir.mkdir(exist_ok=True)
    print(f"  [1] 創建 images/ 資料夾: {images_dir}")
    
    # 2. 移動所有圖片到 images/
    image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    moved_count = 0
    for file in camera_path.iterdir():
        if file.is_file() and file.suffix in image_extensions:
            target_path = images_dir / file.name
            if not target_path.exists():
                shutil.move(str(file), str(target_path))
                moved_count += 1
            else:
                print(f"  [警告] 檔案已存在，跳過: {file.name}")
    
    print(f"  [2] 移動 {moved_count} 張圖片到 images/")
    
    # 3. 將 mask/ 重新命名為 masks/
    mask_dir = camera_path / "mask"
    masks_dir = camera_path / "masks"
    
    if mask_dir.exists() and mask_dir.is_dir():
        if masks_dir.exists():
            print(f"  [警告] masks/ 已存在，合併內容...")
            # 合併 mask/ 的內容到 masks/
            for file in mask_dir.iterdir():
                target_path = masks_dir / file.name
                if not target_path.exists():
                    shutil.move(str(file), str(target_path))
                else:
                    print(f"  [警告] 檔案已存在，跳過: {file.name}")
            mask_dir.rmdir()
        else:
            mask_dir.rename(masks_dir)
        print(f"  [3] 重新命名 mask/ → masks/")
    elif masks_dir.exists():
        print(f"  [3] masks/ 已存在，跳過")
    else:
        print(f"  [警告] 找不到 mask/ 或 masks/ 資料夾")
    
    # 4. 重新命名整個資料夾為 skyfinder_1093
    parent_dir = camera_path.parent
    target_folder = parent_dir / target_name
    
    if target_folder.exists():
        print(f"  [警告] 目標資料夾已存在: {target_folder}")
        print(f"  [4] 跳過重新命名")
        # 如果目標資料夾已存在，使用目標資料夾繼續
        camera_path = target_folder
    else:
        try:
            shutil.move(str(camera_path), str(target_folder))
            print(f"  [4] 重新命名資料夾: {camera_folder} → {target_name}")
            camera_path = target_folder
        except Exception as e:
            print(f"  [警告] 重新命名失敗: {e}")
            print(f"  [4] 請手動重新命名資料夾: {camera_folder} → {target_name}")
    
    print()
    print("=" * 60)
    print("  整理完成！")
    print("=" * 60)
    print()
    
    # 返回最終的資料夾路徑
    return str(camera_path)


def analyze_day_night_distribution(camera_folder="data/skyfinder_1093"):
    """
    分析 camera 資料夾中圖片的 day/night 分布
    """
    camera_path = Path(camera_folder)
    images_dir = camera_path / "images"
    
    if not images_dir.exists():
        print(f"[錯誤] 找不到 images/ 資料夾: {images_dir}")
        return
    
    print("=" * 60)
    print(f"  分析 Day/Night 分布: {camera_folder}")
    print("=" * 60)
    print()
    
    # 收集所有圖片
    image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    image_files = [f for f in images_dir.iterdir() 
                   if f.is_file() and f.suffix in image_extensions]
    
    print(f"  總圖片數: {len(image_files)}")
    print()
    
    # 計算每張圖片的 luma
    luma_values = []
    night_count = 0
    day_count = 0
    luma_threshold = 0.20
    
    print("  計算 luma...")
    for i, img_file in enumerate(image_files):
        if (i + 1) % 100 == 0:
            print(f"    處理中: {i + 1}/{len(image_files)}")
        
        mean_luma = calculate_mean_luma(img_file)
        if mean_luma is not None:
            luma_values.append(mean_luma)
            if mean_luma < luma_threshold:
                night_count += 1
            else:
                day_count += 1
    
    if len(luma_values) == 0:
        print("  [錯誤] 無法計算任何圖片的 luma")
        return
    
    luma_array = np.array(luma_values)
    
    # 統計資訊
    print()
    print("=" * 60)
    print("  Day/Night 分布統計")
    print("=" * 60)
    print()
    print(f"  總圖片數: {len(luma_values)}")
    print(f"  Night (luma < 0.20): {night_count} 張 ({night_count/len(luma_values)*100:.1f}%)")
    print(f"  Day (luma >= 0.20): {day_count} 張 ({day_count/len(luma_values)*100:.1f}%)")
    print()
    print(f"  Luma 統計:")
    print(f"    平均: {np.mean(luma_array):.4f}")
    print(f"    標準差: {np.std(luma_array):.4f}")
    print(f"    最小值: {np.min(luma_array):.4f}")
    print(f"    最大值: {np.max(luma_array):.4f}")
    print(f"    中位數: {np.median(luma_array):.4f}")
    print()
    print("=" * 60)
    print("  分析完成！")
    print("=" * 60)


def main():
    # 整理資料夾結構
    result = organize_camera_folder("data/1093", "skyfinder_1093")
    if result:
        # 分析 day/night 分布（使用整理後的資料夾路徑）
        final_folder = result if isinstance(result, str) else "data/skyfinder_1093"
        analyze_day_night_distribution(final_folder)
    else:
        print("[錯誤] 整理失敗，跳過分析")


if __name__ == "__main__":
    main()
