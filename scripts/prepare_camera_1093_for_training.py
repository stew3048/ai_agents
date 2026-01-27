"""
準備 camera 1093 資料用於訓練

功能：
1. 從 870 張 image 中選出約 80-100 張（保持 day/night 比例）
2. 為每張選出的 image 創建對應的 mask（檔名要對應）
3. 刪除未選中的 image
"""

import os
import shutil
from pathlib import Path
from PIL import Image
import numpy as np
import random


def srgb_to_linear(v):
    """sRGB 轉 linear RGB"""
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


def calculate_mean_luma(image_path):
    """計算圖片的平均 luma（與訓練/評估一致）"""
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


def prepare_camera_1093(
    camera_folder="data/skyfinder_1093",
    target_count=90,  # 目標總數（約 80-100 張）
    night_ratio=0.25,  # Night 比例（約 20-25%）
    seed=42
):
    """
    準備 camera 1093 資料用於訓練
    
    參數:
        camera_folder: camera 資料夾路徑
        target_count: 目標總數
        night_ratio: Night 比例
        seed: 隨機種子（確保可重現）
    """
    camera_path = Path(camera_folder)
    images_dir = camera_path / "images"
    masks_dir = camera_path / "masks"
    
    if not images_dir.exists():
        print(f"[錯誤] 找不到 images/ 資料夾: {images_dir}")
        return False
    
    if not masks_dir.exists():
        print(f"[錯誤] 找不到 masks/ 資料夾: {masks_dir}")
        return False
    
    print("=" * 60)
    print(f"  準備 Camera 1093 資料用於訓練")
    print("=" * 60)
    print()
    print(f"  目標總數: {target_count} 張")
    print(f"  Night 比例: {night_ratio*100:.1f}%")
    print(f"  預期 Night: {int(target_count * night_ratio)} 張")
    print(f"  預期 Day: {int(target_count * (1 - night_ratio))} 張")
    print()
    
    # 1. 收集所有圖片並計算 luma
    print("  [1] 讀取所有圖片並計算 luma...")
    image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    all_images = [f for f in images_dir.iterdir() 
                  if f.is_file() and f.suffix in image_extensions]
    
    print(f"      總圖片數: {len(all_images)}")
    
    image_luma_list = []
    for i, img_file in enumerate(all_images):
        if (i + 1) % 100 == 0:
            print(f"      處理中: {i + 1}/{len(all_images)}")
        
        mean_luma = calculate_mean_luma(img_file)
        if mean_luma is not None:
            image_luma_list.append({
                'file': img_file,
                'luma': mean_luma,
                'is_night': mean_luma < 0.20
            })
    
    print(f"      成功讀取: {len(image_luma_list)} 張")
    print()
    
    # 2. 分類為 night/day
    night_images = [x for x in image_luma_list if x['is_night']]
    day_images = [x for x in image_luma_list if not x['is_night']]
    
    print(f"  [2] 分類結果:")
    print(f"      Night (luma < 0.20): {len(night_images)} 張")
    print(f"      Day (luma >= 0.20): {len(day_images)} 張")
    print()
    
    # 3. 計算目標數量
    target_night = int(target_count * night_ratio)
    target_day = target_count - target_night
    
    print(f"  [3] 目標選取:")
    print(f"      Night: {target_night} 張 (從 {len(night_images)} 張中選)")
    print(f"      Day: {target_day} 張 (從 {len(day_images)} 張中選)")
    print()
    
    # 4. 隨機選取（使用 seed 確保可重現）
    random.seed(seed)
    np.random.seed(seed)
    
    if len(night_images) < target_night:
        print(f"  [警告] Night 圖片不足，只能選 {len(night_images)} 張")
        selected_night = night_images.copy()
    else:
        # 隨機選取，但盡量選擇 luma 分布均勻的
        night_lumas = [x['luma'] for x in night_images]
        # 使用分層採樣：將 luma 範圍分成幾個區間，每個區間選一些
        night_sorted = sorted(night_images, key=lambda x: x['luma'])
        step = len(night_sorted) / target_night
        selected_night = [night_sorted[int(i * step)] for i in range(target_night)]
        # 如果還不夠，隨機補充
        if len(selected_night) < target_night:
            remaining = [x for x in night_images if x not in selected_night]
            selected_night.extend(random.sample(remaining, target_night - len(selected_night)))
    
    if len(day_images) < target_day:
        print(f"  [警告] Day 圖片不足，只能選 {len(day_images)} 張")
        selected_day = day_images.copy()
    else:
        # 隨機選取
        day_sorted = sorted(day_images, key=lambda x: x['luma'])
        step = len(day_sorted) / target_day
        selected_day = [day_sorted[int(i * step)] for i in range(target_day)]
        # 如果還不夠，隨機補充
        if len(selected_day) < target_day:
            remaining = [x for x in day_images if x not in selected_day]
            selected_day.extend(random.sample(remaining, target_day - len(selected_day)))
    
    selected_images = selected_night + selected_day
    print(f"  [4] 選取完成:")
    print(f"      選取 Night: {len(selected_night)} 張")
    print(f"      選取 Day: {len(selected_day)} 張")
    print(f"      總計: {len(selected_images)} 張")
    print()
    
    # 5. 檢查現有的 mask
    print("  [5] 檢查現有的 mask...")
    existing_masks = list(masks_dir.iterdir())
    print(f"      現有 mask 檔案數: {len(existing_masks)}")
    
    # 找到可用的 mask 檔案（優先使用與 camera_id 相同的，否則用第一個）
    source_mask = None
    for mask_file in existing_masks:
        if mask_file.is_file() and mask_file.suffix.lower() in {'.png', '.jpg', '.jpeg'}:
            if '1093' in mask_file.stem.lower():
                source_mask = mask_file
                break
    
    if source_mask is None and existing_masks:
        source_mask = existing_masks[0]
    
    if source_mask is None:
        print(f"  [錯誤] 找不到可用的 mask 檔案")
        return False
    
    print(f"      使用 mask 來源: {source_mask.name}")
    print()
    
    # 6. 為選中的 image 創建對應的 mask（檔名要對應）
    print("  [6] 為選中的 image 創建對應的 mask...")
    created_count = 0
    for img_info in selected_images:
        img_file = img_info['file']
        # 將 image 檔名轉換為 mask 檔名（副檔名改為 .png）
        mask_name = img_file.stem + '.png'
        mask_path = masks_dir / mask_name
        
        # 如果 mask 不存在，複製 source_mask
        if not mask_path.exists():
            shutil.copy2(str(source_mask), str(mask_path))
            created_count += 1
    
    print(f"      創建了 {created_count} 個 mask 檔案")
    print()
    
    # 7. 刪除未選中的 image
    print("  [7] 刪除未選中的 image...")
    selected_files = {img_info['file'] for img_info in selected_images}
    deleted_count = 0
    
    for img_file in all_images:
        if img_file not in selected_files:
            img_file.unlink()
            deleted_count += 1
    
    print(f"      刪除了 {deleted_count} 張 image")
    print()
    
    # 8. 清理未使用的 mask（只保留選中 image 對應的 mask）
    print("  [8] 清理未使用的 mask...")
    selected_mask_names = {img_info['file'].stem + '.png' for img_info in selected_images}
    all_masks = [f for f in masks_dir.iterdir() if f.is_file() and f.suffix.lower() == '.png']
    deleted_mask_count = 0
    
    for mask_file in all_masks:
        if mask_file.name not in selected_mask_names:
            mask_file.unlink()
            deleted_mask_count += 1
    
    print(f"      刪除了 {deleted_mask_count} 個未使用的 mask")
    print()
    
    # 9. 驗證最終結果
    print("  [9] 驗證最終結果...")
    final_images = list(images_dir.iterdir())
    final_masks = list(masks_dir.iterdir())
    final_images = [f for f in final_images if f.is_file() and f.suffix.lower() in image_extensions]
    final_masks = [f for f in final_masks if f.is_file() and f.suffix.lower() == '.png']
    
    print(f"      最終 image 數: {len(final_images)}")
    print(f"      最終 mask 數: {len(final_masks)}")
    
    # 檢查配對
    image_names = {f.stem for f in final_images}
    mask_names = {f.stem for f in final_masks}
    
    missing_masks = image_names - mask_names
    extra_masks = mask_names - image_names
    
    if missing_masks:
        print(f"  [警告] 有 {len(missing_masks)} 張 image 沒有對應的 mask")
    if extra_masks:
        print(f"  [警告] 有 {len(extra_masks)} 個 mask 沒有對應的 image")
    
    if not missing_masks and not extra_masks:
        print(f"      [OK] 所有 image 都有對應的 mask")
    
    print()
    print("=" * 60)
    print("  準備完成！")
    print("=" * 60)
    print()
    print(f"  最終統計:")
    print(f"    Image: {len(final_images)} 張")
    print(f"    Mask: {len(final_masks)} 張")
    print(f"    配對: {'完整' if not missing_masks and not extra_masks else '不完整'}")
    
    return True


def main():
    prepare_camera_1093(
        camera_folder="data/skyfinder_1093",
        target_count=90,  # 約 80-100 張
        night_ratio=0.25,  # 25% night
        seed=42
    )


if __name__ == "__main__":
    main()
