"""
建立所有影像的 metadata.csv，用於 in-domain split

功能：
- 掃描所有 data/skyfinder_*/images/ 與 masks/
- 排除 skyfinder_sample（不納入）
- 對每張影像產出 metadata 欄位：
  - image_path（相對路徑，如 skyfinder_10066/images/001.jpg，不含 data/ 前綴）
  - camera_id
  - filename_or_timestamp（例如 001.jpg，用於時間序排序）
  - brightness_estimation（用 mean_luma_linear）
  - sky_area_ratio（從 GT mask 計算，若 mask 存在）
  - is_no_sky（sky_area_ratio == 0 或 mask 全黑）

輸出：outputs/metadata_all_images.csv
"""

import os
import csv
import sys
from pathlib import Path
from PIL import Image
import numpy as np
import torch

# 專案根目錄
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.image_utils import srgb_to_linear


def find_camera_folders(base_dir: str = "data"):
    """
    找出 data/ 底下所有以 skyfinder_ 開頭的 camera 資料夾
    排除 skyfinder_sample
    """
    if not os.path.exists(base_dir):
        return []
    
    folders = []
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        # 只保留以 skyfinder_ 開頭且不是 skyfinder_masks 或 skyfinder_sample 的資料夾
        if (os.path.isdir(item_path) and 
            item.startswith("skyfinder_") and 
            item not in ["skyfinder_masks", "skyfinder_sample"]):
            folders.append(item_path)
    
    return sorted(folders)


def extract_camera_id(folder_path: str):
    """從資料夾路徑提取 camera ID"""
    folder_name = os.path.basename(folder_path)
    if folder_name.startswith("skyfinder_"):
        return folder_name.replace("skyfinder_", "")
    return folder_name


def calculate_brightness_estimation(image_path: str) -> float:
    """
    計算圖片的平均 luma（linear RGB）
    使用與訓練/評估一致的計算方式
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
        
        return float(mean_luma)
    except Exception as e:
        print(f"  警告：無法計算亮度 {image_path}: {e}")
        return None


def calculate_sky_area_ratio(mask_path: str) -> float:
    """
    從 GT mask 計算 sky_area_ratio
    返回 0.0 到 1.0 之間的值
    """
    try:
        if not os.path.exists(mask_path):
            return None
        
        mask = Image.open(mask_path)
        mask_array = np.array(mask)
        
        # 轉換為二值化 mask（0 或 255）
        if mask_array.dtype != np.uint8:
            mask_array = mask_array.astype(np.uint8)
        
        # 如果 mask 是彩色，取第一個通道或轉為灰階
        if len(mask_array.shape) == 3:
            mask_array = mask_array[:, :, 0]
        
        # 二值化：> 128 視為 sky
        sky_pixels = np.sum(mask_array > 128)
        total_pixels = mask_array.size
        
        if total_pixels == 0:
            return 0.0
        
        ratio = sky_pixels / total_pixels
        return float(ratio)
    except Exception as e:
        print(f"  警告：無法計算 sky_area_ratio {mask_path}: {e}")
        return None


def find_mask_path(images_dir: str, masks_dir: str, image_filename: str) -> str:
    """
    找到對應的 mask 路徑
    嘗試多種可能的 mask 檔名
    """
    if not os.path.exists(masks_dir):
        return None
    
    img_base = os.path.splitext(image_filename)[0]
    
    # 嘗試多種可能的 mask 檔名
    possible_mask_names = [
        f"{img_base}.png",
        f"{img_base}.pgm",
        f"{img_base}.jpg",
        f"{image_filename}",  # 相同檔名
    ]
    
    for mask_name in possible_mask_names:
        mask_path = os.path.join(masks_dir, mask_name)
        if os.path.exists(mask_path):
            return mask_path
    
    return None


def is_image_valid(img_path: str) -> bool:
    """檢查圖片是否有效（未受損）"""
    try:
        img = Image.open(img_path)
        img.verify()  # 驗證圖片結構完整性
        img.close()
        
        # 重新打開並載入數據（verify 後需要重新打開）
        img = Image.open(img_path)
        img.load()  # 載入圖片數據
        img.close()
        
        return True
    except Exception:
        return False


def main():
    print("=" * 60)
    print("  建立所有影像的 metadata.csv（用於 in-domain split）")
    print("=" * 60)
    print()
    
    base_dir = "data"
    output_file = "outputs/metadata_all_images.csv"
    
    # 確保輸出目錄存在
    os.makedirs("outputs", exist_ok=True)
    
    # 找出所有 camera 資料夾
    print("掃描 camera 資料夾...")
    camera_folders = find_camera_folders(base_dir)
    
    if len(camera_folders) == 0:
        print("  錯誤：找不到任何 camera 資料夾")
        return
    
    print(f"  找到 {len(camera_folders)} 個 camera 資料夾")
    for folder in camera_folders:
        print(f"    - {os.path.basename(folder)}")
    print()
    
    # 收集所有影像的 metadata
    metadata_list = []
    total_images = 0
    skipped_images = 0
    
    for camera_folder in camera_folders:
        camera_id = extract_camera_id(camera_folder)
        images_dir = os.path.join(camera_folder, "images")
        masks_dir = os.path.join(camera_folder, "masks")
        
        if not os.path.exists(images_dir):
            print(f"  跳過 {camera_id}：找不到 images 目錄")
            continue
        
        print(f"處理 camera {camera_id}...")
        
        # 取得所有影像檔案
        image_files = []
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
        
        for filename in os.listdir(images_dir):
            if any(filename.lower().endswith(ext) for ext in valid_extensions):
                image_files.append(filename)
        
        image_files = sorted(image_files)  # 排序以確保 deterministic
        
        print(f"  找到 {len(image_files)} 張影像")
        
        processed = 0
        for image_filename in image_files:
            image_path = os.path.join(images_dir, image_filename)
            
            # 檢查圖片是否有效
            if not is_image_valid(image_path):
                skipped_images += 1
                print(f"    跳過受損圖片：{image_filename}")
                continue
            
            # 相對路徑（不含 data/ 前綴）
            relative_image_path = os.path.join(
                os.path.basename(camera_folder), 
                "images", 
                image_filename
            ).replace("\\", "/")  # 統一使用 / 作為路徑分隔符
            
            # 計算 brightness_estimation
            brightness = calculate_brightness_estimation(image_path)
            
            # 找到對應的 mask
            mask_path = find_mask_path(images_dir, masks_dir, image_filename)
            sky_area_ratio = None
            is_no_sky = None
            
            if mask_path:
                sky_area_ratio = calculate_sky_area_ratio(mask_path)
                if sky_area_ratio is not None:
                    is_no_sky = (sky_area_ratio == 0.0)
            
            # 建立 metadata 記錄
            metadata = {
                'image_path': relative_image_path,
                'camera_id': camera_id,
                'filename_or_timestamp': image_filename,
                'brightness_estimation': brightness if brightness is not None else '',
                'sky_area_ratio': sky_area_ratio if sky_area_ratio is not None else '',
                'is_no_sky': is_no_sky if is_no_sky is not None else ''
            }
            
            metadata_list.append(metadata)
            processed += 1
            total_images += 1
            
            if processed % 100 == 0:
                print(f"    已處理 {processed}/{len(image_files)} 張影像...")
        
        print(f"  Camera {camera_id}：處理完成，共 {processed} 張有效影像")
        print()
    
    # 寫入 CSV
    print(f"寫入 metadata 到 {output_file}...")
    
    fieldnames = [
        'image_path',
        'camera_id',
        'filename_or_timestamp',
        'brightness_estimation',
        'sky_area_ratio',
        'is_no_sky'
    ]
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metadata_list)
    
    print(f"  完成！共 {total_images} 張影像")
    if skipped_images > 0:
        print(f"  跳過 {skipped_images} 張受損圖片")
    print(f"  輸出檔案：{output_file}")
    print()


if __name__ == "__main__":
    main()
