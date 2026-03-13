"""
數據準備腳本（Colab 專屬版本，參數化）

功能：
- 步驟 1（metadata）：建立所有影像的 metadata.csv
- 步驟 2（splits）：建立 train/val/test splits

所有路徑都可通過 CLI 指定
"""

import os
import csv
import json
import sys
import argparse
import re
from pathlib import Path
from collections import defaultdict
from PIL import Image
import numpy as np

# 確保輸出立即刷新
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# 添加專案根目錄到 Python 路徑
_script_dir = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_script_dir)
sys.path.insert(0, _root)
sys.path.insert(0, _script_dir)

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


def create_metadata(data_dir: str, outputs_dir: str, metadata_file: str):
    """步驟 1：建立 metadata CSV"""
    print("=" * 60)
    print("  步驟 1：建立所有影像的 metadata.csv")
    print("=" * 60)
    print()
    
    # 確保輸出目錄存在
    os.makedirs(outputs_dir, exist_ok=True)
    
    # 找出所有 camera 資料夾
    print("掃描 camera 資料夾...")
    camera_folders = find_camera_folders(data_dir)
    
    if len(camera_folders) == 0:
        print("  錯誤：找不到任何 camera 資料夾")
        return False
    
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
    print(f"寫入 metadata 到 {metadata_file}...")
    
    fieldnames = [
        'image_path',
        'camera_id',
        'filename_or_timestamp',
        'brightness_estimation',
        'sky_area_ratio',
        'is_no_sky'
    ]
    
    with open(metadata_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metadata_list)
    
    print(f"  完成！共 {total_images} 張影像")
    if skipped_images > 0:
        print(f"  跳過 {skipped_images} 張受損圖片")
    print(f"  輸出檔案：{metadata_file}")
    print()
    
    return True


def natural_sort_key(filename: str):
    """
    自然排序的 key function
    例如：001.jpg < 002.jpg < 010.jpg < 100.jpg
    """
    # 提取數字部分
    parts = re.split(r'(\d+)', filename)
    return [int(part) if part.isdigit() else part.lower() for part in parts]


def create_splits(metadata_file: str, outputs_dir: str):
    """步驟 2：建立 train/val/test splits"""
    print("=" * 60)
    print("  步驟 2：建立 in-domain splits（每個 camera 80/10/10）")
    print("=" * 60)
    print()
    
    # 讀取 metadata
    print(f"讀取 metadata：{metadata_file}...")
    if not os.path.exists(metadata_file):
        raise FileNotFoundError(f"找不到 metadata 檔案：{metadata_file}")
    
    metadata_list = []
    with open(metadata_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            metadata_list.append(row)
    
    print(f"  共 {len(metadata_list)} 張影像")
    print()
    
    # 依 camera_id 分組
    camera_groups = defaultdict(list)
    for metadata in metadata_list:
        camera_id = metadata['camera_id']
        camera_groups[camera_id].append(metadata)
    
    print(f"找到 {len(camera_groups)} 個 camera")
    print()
    
    # 對每個 camera 做 split
    train_list = []
    val_list = []
    test_list = []
    
    splits_summary = {}
    
    for camera_id in sorted(camera_groups.keys()):
        images = camera_groups[camera_id]
        
        # 依 filename_or_timestamp 排序（自然排序）
        images.sort(key=lambda x: natural_sort_key(x['filename_or_timestamp']))
        
        n = len(images)
        train_end = int(n * 0.8)
        val_end = int(n * 0.9)
        
        # 前 80% → train
        train_images = images[:train_end]
        train_count = len(train_images)
        
        # 接下來 10% → val
        val_images = images[train_end:val_end]
        val_count = len(val_images)
        
        # 最後 10% → test
        test_images = images[val_end:]
        test_count = len(test_images)
        
        # 加入對應的 list
        for img in train_images:
            train_list.append(img['image_path'])
        for img in val_images:
            val_list.append(img['image_path'])
        for img in test_images:
            test_list.append(img['image_path'])
        
        # 記錄統計資訊
        splits_summary[camera_id] = {
            'total': n,
            'train': train_count,
            'val': val_count,
            'test': test_count,
            'train_ratio': train_count / n if n > 0 else 0.0,
            'val_ratio': val_count / n if n > 0 else 0.0,
            'test_ratio': test_count / n if n > 0 else 0.0
        }
        
        print(f"Camera {camera_id}:")
        print(f"  總數: {n}")
        print(f"  Train: {train_count} ({train_count/n*100:.1f}%)")
        print(f"  Val:   {val_count} ({val_count/n*100:.1f}%)")
        print(f"  Test:  {test_count} ({test_count/n*100:.1f}%)")
        print()
    
    # 確保輸出目錄存在
    os.makedirs(outputs_dir, exist_ok=True)
    
    # 寫入 train_list.txt
    train_file = os.path.join(outputs_dir, 'train_list.txt')
    print(f"寫入 {train_file}...")
    with open(train_file, 'w', encoding='utf-8') as f:
        for image_path in train_list:
            f.write(f"{image_path}\n")
    print(f"  完成，共 {len(train_list)} 張影像")
    
    # 寫入 val_list.txt
    val_file = os.path.join(outputs_dir, 'val_list.txt')
    print(f"寫入 {val_file}...")
    with open(val_file, 'w', encoding='utf-8') as f:
        for image_path in val_list:
            f.write(f"{image_path}\n")
    print(f"  完成，共 {len(val_list)} 張影像")
    
    # 寫入 test_list.txt（固定，鎖死）
    test_file = os.path.join(outputs_dir, 'test_list.txt')
    print(f"寫入 {test_file}...")
    print(f"  [注意] 此檔案鎖死後不得再修改！")
    with open(test_file, 'w', encoding='utf-8') as f:
        for image_path in test_list:
            f.write(f"{image_path}\n")
    print(f"  完成，共 {len(test_list)} 張影像")
    
    # 寫入 summary JSON
    summary_file = os.path.join(outputs_dir, 'in_domain_splits_summary.json')
    print(f"寫入 {summary_file}...")
    
    summary_data = {
        'split_method': 'in_domain_per_camera_80_10_10',
        'description': '每個 camera 依 filename 時間序排序，前 80% → train，接下來 10% → val，最後 10% → test',
        'total_images': {
            'train': len(train_list),
            'val': len(val_list),
            'test': len(test_list),
            'total': len(train_list) + len(val_list) + len(test_list)
        },
        'per_camera': splits_summary
    }
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    
    print(f"  完成")
    print()
    
    # 總計
    print("=" * 60)
    print("  Split 完成")
    print("=" * 60)
    print(f"總計：")
    print(f"  Train: {len(train_list)} 張")
    print(f"  Val:   {len(val_list)} 張")
    print(f"  Test:  {len(test_list)} 張（鎖死）")
    print(f"  總計: {len(train_list) + len(val_list) + len(test_list)} 張")
    print()
    print(f"輸出檔案：")
    print(f"  - {train_file}")
    print(f"  - {val_file}")
    print(f"  - {test_file}（鎖死）")
    print(f"  - {summary_file}")
    print()


def main():
    parser = argparse.ArgumentParser(description='數據準備腳本（Colab 專屬版本，參數化）')
    
    parser.add_argument('--step', type=str, required=True, choices=['metadata', 'splits', 'all'],
                        help='執行步驟：metadata（建立 metadata CSV）、splits（建立 splits）、all（執行全部）')
    
    # 路徑參數
    parser.add_argument('--data_dir', type=str, default='data',
                        help='數據根目錄（預設：data）')
    parser.add_argument('--outputs_dir', type=str, default='outputs',
                        help='輸出目錄（預設：outputs）')
    parser.add_argument('--metadata_file', type=str, default=None,
                        help='metadata CSV 路徑（預設：{outputs_dir}/metadata_all_images.csv）')
    
    args = parser.parse_args()
    
    # 設定預設 metadata_file
    if args.metadata_file is None:
        args.metadata_file = os.path.join(args.outputs_dir, 'metadata_all_images.csv')
    
    # 執行步驟
    if args.step == 'metadata' or args.step == 'all':
        success = create_metadata(args.data_dir, args.outputs_dir, args.metadata_file)
        if not success:
            sys.exit(1)
    
    if args.step == 'splits' or args.step == 'all':
        create_splits(args.metadata_file, args.outputs_dir)


if __name__ == "__main__":
    main()
