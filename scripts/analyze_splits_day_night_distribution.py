"""
分析 train/val/test splits 的 day/night 分布

使用 camera_inventory.csv 中的 night_ratio 來估算每個 split 的分布
"""

import json
import csv
from collections import defaultdict


def load_camera_inventory():
    """載入 camera_inventory.csv"""
    inventory = {}
    with open('outputs/camera_inventory.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            camera_id = row['camera_id']
            night_ratio = float(row['night_ratio']) if row['night_ratio'] else 0.0
            image_count = int(row['image_count']) if row['image_count'] else 0
            inventory[camera_id] = {
                'night_ratio': night_ratio,
                'image_count': image_count
            }
    return inventory


def analyze_splits_distribution():
    """分析 splits 的 day/night 分布"""
    # 載入 splits
    with open('outputs/multi_camera_splits.json', 'r', encoding='utf-8') as f:
        splits = json.load(f)
    
    # 載入 camera inventory
    inventory = load_camera_inventory()
    
    print("=" * 60)
    print("  Train/Val/Test Day/Night 分布分析")
    print("=" * 60)
    print()
    
    # 統計每個 split 的 camera 分布
    split_stats = {}
    
    for split_name in ['train', 'sanity', 'val', 'test']:
        split_data = splits[split_name]
        
        # 統計每個 camera 的數量
        camera_counts = defaultdict(int)
        for item in split_data:
            camera_id = item['camera_id']
            camera_counts[camera_id] += 1
        
        # 計算總數和 night/day 分布
        total_count = len(split_data)
        night_count = 0
        day_count = 0
        
        for camera_id, count in camera_counts.items():
            if camera_id in inventory:
                night_ratio = inventory[camera_id]['night_ratio']
                camera_night = int(count * night_ratio)
                camera_day = count - camera_night
                night_count += camera_night
                day_count += camera_day
        
        split_stats[split_name] = {
            'total': total_count,
            'night': night_count,
            'day': day_count,
            'night_ratio': night_count / total_count if total_count > 0 else 0.0,
            'day_ratio': day_count / total_count if total_count > 0 else 0.0,
            'camera_counts': dict(camera_counts)
        }
    
    # 顯示結果
    for split_name in ['train', 'sanity', 'val', 'test']:
        stats = split_stats[split_name]
        print(f"  [{split_name.upper()}]")
        print(f"    總數: {stats['total']} 張")
        print(f"    Night: {stats['night']} 張 ({stats['night_ratio']*100:.1f}%)")
        print(f"    Day: {stats['day']} 張 ({stats['day_ratio']*100:.1f}%)")
        print(f"    Cameras: {stats['camera_counts']}")
        print()
    
    # 合併 train + sanity 作為整體 train
    train_total = split_stats['train']['total'] + split_stats['sanity']['total']
    train_night = split_stats['train']['night'] + split_stats['sanity']['night']
    train_day = split_stats['train']['day'] + split_stats['sanity']['day']
    
    print("=" * 60)
    print("  整體統計（Train+Sanity 合併）")
    print("=" * 60)
    print()
    print(f"  [TRAIN] (train + sanity)")
    print(f"    總數: {train_total} 張")
    print(f"    Night: {train_night} 張 ({train_night/train_total*100:.1f}%)")
    print(f"    Day: {train_day} 張 ({train_day/train_total*100:.1f}%)")
    print()
    print(f"  [VAL]")
    print(f"    總數: {split_stats['val']['total']} 張")
    print(f"    Night: {split_stats['val']['night']} 張 ({split_stats['val']['night_ratio']*100:.1f}%)")
    print(f"    Day: {split_stats['val']['day']} 張 ({split_stats['val']['day_ratio']*100:.1f}%)")
    print()
    print(f"  [TEST]")
    print(f"    總數: {split_stats['test']['total']} 張")
    print(f"    Night: {split_stats['test']['night']} 張 ({split_stats['test']['night_ratio']*100:.1f}%)")
    print(f"    Day: {split_stats['test']['day']} 張 ({split_stats['test']['day_ratio']*100:.1f}%)")
    print()
    print("=" * 60)


if __name__ == '__main__':
    analyze_splits_distribution()
