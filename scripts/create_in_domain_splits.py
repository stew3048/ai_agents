"""
方案 S - 每個 camera 做 80/10/10 split（時間序）

規則：
- 讀取 outputs/metadata_all_images.csv
- 對每個 camera_id：
  - 依 filename_or_timestamp 排序（例如 001.jpg < 002.jpg）
  - 前 80% → train
  - 接下來 10% → val
  - 最後 10% → test
- 嚴格規定：
  - 同一張影像不可出現在多個 split
  - Split 必須 deterministic（固定排序，不隨機）
  - 產出三份固定清單：train_list.txt、val_list.txt、test_list.txt（每行一個相對路徑 image_path）

輸出：
- outputs/train_list.txt（固定）
- outputs/val_list.txt（固定）
- outputs/test_list.txt（固定，之後不得修改）
- outputs/in_domain_splits_summary.json（統計：每個 camera 的 train/val/test 張數）
"""

import os
import csv
import json
from collections import defaultdict
from pathlib import Path


def natural_sort_key(filename: str):
    """
    自然排序的 key function
    例如：001.jpg < 002.jpg < 010.jpg < 100.jpg
    """
    import re
    # 提取數字部分
    parts = re.split(r'(\d+)', filename)
    return [int(part) if part.isdigit() else part.lower() for part in parts]


def load_metadata(metadata_file: str):
    """讀取 metadata_all_images.csv"""
    metadata_list = []
    
    if not os.path.exists(metadata_file):
        raise FileNotFoundError(f"找不到 metadata 檔案：{metadata_file}")
    
    with open(metadata_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            metadata_list.append(row)
    
    return metadata_list


def create_in_domain_splits(metadata_file: str = "outputs/metadata_all_images.csv"):
    """
    建立 in-domain splits（每個 camera 80/10/10）
    
    參數:
        metadata_file: metadata CSV 檔案路徑
    """
    print("=" * 60)
    print("  方案 S - 每個 camera 做 80/10/10 split（時間序）")
    print("=" * 60)
    print()
    
    # 讀取 metadata
    print(f"讀取 metadata：{metadata_file}...")
    metadata_list = load_metadata(metadata_file)
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
    os.makedirs("outputs", exist_ok=True)
    
    # 寫入 train_list.txt
    train_file = "outputs/train_list.txt"
    print(f"寫入 {train_file}...")
    with open(train_file, 'w', encoding='utf-8') as f:
        for image_path in train_list:
            f.write(f"{image_path}\n")
    print(f"  完成，共 {len(train_list)} 張影像")
    
    # 寫入 val_list.txt
    val_file = "outputs/val_list.txt"
    print(f"寫入 {val_file}...")
    with open(val_file, 'w', encoding='utf-8') as f:
        for image_path in val_list:
            f.write(f"{image_path}\n")
    print(f"  完成，共 {len(val_list)} 張影像")
    
    # 寫入 test_list.txt（固定，鎖死）
    test_file = "outputs/test_list.txt"
    print(f"寫入 {test_file}...")
    print(f"  [注意] 此檔案鎖死後不得再修改！")
    with open(test_file, 'w', encoding='utf-8') as f:
        for image_path in test_list:
            f.write(f"{image_path}\n")
    print(f"  完成，共 {len(test_list)} 張影像")
    
    # 寫入 summary JSON
    summary_file = "outputs/in_domain_splits_summary.json"
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


if __name__ == "__main__":
    import sys
    
    metadata_file = sys.argv[1] if len(sys.argv) > 1 else "outputs/metadata_all_images.csv"
    
    create_in_domain_splits(metadata_file)
