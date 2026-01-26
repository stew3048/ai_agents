"""
分析 Train 資料的 luma 分布與 aug 套用情況

目的：
1. 檢查 train 中每個樣本的 luma 值
2. 確認哪些樣本會被判定為 night（luma < 0.20）並套用 aug
3. 檢查是否有 day 樣本被誤判為 night（做了 aug）
4. 統計實際的 night/day 比例與 aug 套用比例
"""

import os
import json
import numpy as np
from PIL import Image
from pathlib import Path
from collections import defaultdict

def srgb_to_linear(v):
    """sRGB 轉 linear RGB"""
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)

def calculate_mean_luma(image_path):
    """計算圖片的平均 luma（linear RGB）"""
    img = Image.open(image_path).convert('RGB')
    img_np = np.array(img, dtype=np.float32) / 255.0  # [H, W, 3], [0, 1]
    
    # 轉為 linear RGB
    R_lin = srgb_to_linear(img_np[:, :, 0])
    G_lin = srgb_to_linear(img_np[:, :, 1])
    B_lin = srgb_to_linear(img_np[:, :, 2])
    
    # 計算 luma (ITU-R BT.709)
    luma = 0.2126 * R_lin + 0.7152 * G_lin + 0.0722 * B_lin
    mean_luma = luma.mean()
    
    return mean_luma

def analyze_train_luma_aug():
    """分析 train 資料的 luma 分布與 aug 套用情況"""
    
    # 讀取 split
    split_path = Path("outputs/multi_camera_splits.json")
    with open(split_path, 'r', encoding='utf-8') as f:
        splits = json.load(f)
    
    train_list = splits['train']
    base_data_dir = Path("data")
    
    # 統計資訊
    luma_threshold = 0.20
    stats = {
        'total': len(train_list),
        'night_count': 0,  # luma < 0.20
        'day_count': 0,    # luma >= 0.20
        'luma_values': [],
        'camera_stats': defaultdict(lambda: {'total': 0, 'night': 0, 'day': 0}),
        'samples': []
    }
    
    print(f"分析 {len(train_list)} 個 train 樣本...")
    print(f"Luma 閾值: {luma_threshold}")
    print()
    
    # 處理每個樣本
    for idx, item in enumerate(train_list):
        camera_id = item['camera_id']
        image_filename = item['image']
        
        image_path = base_data_dir / f"skyfinder_{camera_id}" / "images" / image_filename
        
        if not image_path.exists():
            print(f"  [警告] 找不到檔案: {image_path}")
            continue
        
        # 計算 luma
        try:
            mean_luma = calculate_mean_luma(image_path)
            is_night = mean_luma < luma_threshold
            
            stats['luma_values'].append(mean_luma)
            stats['camera_stats'][camera_id]['total'] += 1
            
            if is_night:
                stats['night_count'] += 1
                stats['camera_stats'][camera_id]['night'] += 1
            else:
                stats['day_count'] += 1
                stats['camera_stats'][camera_id]['day'] += 1
            
            stats['samples'].append({
                'camera_id': camera_id,
                'image': image_filename,
                'luma': float(mean_luma),
                'is_night': is_night,
                'will_aug': is_night
            })
            
            if (idx + 1) % 50 == 0:
                print(f"  處理進度: {idx + 1}/{len(train_list)}")
        
        except Exception as e:
            print(f"  [錯誤] 處理 {image_path} 時發生錯誤: {e}")
            continue
    
    # 計算統計
    luma_array = np.array(stats['luma_values'])
    
    print()
    print("=" * 60)
    print("Train 資料 Luma 分布與 Aug 套用統計")
    print("=" * 60)
    print()
    
    print(f"總樣本數: {stats['total']}")
    print(f"Night 樣本 (luma < {luma_threshold}): {stats['night_count']} ({stats['night_count']/stats['total']*100:.1f}%)")
    print(f"Day 樣本 (luma >= {luma_threshold}): {stats['day_count']} ({stats['day_count']/stats['total']*100:.1f}%)")
    print()
    
    print("Luma 統計:")
    print(f"  最小值: {luma_array.min():.4f}")
    print(f"  最大值: {luma_array.max():.4f}")
    print(f"  平均值: {luma_array.mean():.4f}")
    print(f"  中位數: {np.median(luma_array):.4f}")
    print(f"  標準差: {luma_array.std():.4f}")
    print()
    
    # 檢查邊界附近的樣本（可能誤判）
    boundary_samples = [s for s in stats['samples'] 
                       if 0.15 <= s['luma'] <= 0.25]
    print(f"邊界樣本 (0.15 <= luma <= 0.25): {len(boundary_samples)} 個")
    if len(boundary_samples) > 0:
        print("  這些樣本可能因 luma 閾值而誤判")
        print("  前 10 個邊界樣本:")
        for s in sorted(boundary_samples, key=lambda x: abs(x['luma'] - luma_threshold))[:10]:
            print(f"    {s['camera_id']}/{s['image']}: luma={s['luma']:.4f}, "
                  f"is_night={s['is_night']}, will_aug={s['will_aug']}")
    print()
    
    # 各 Camera 統計
    print("各 Camera 統計:")
    for camera_id in sorted(stats['camera_stats'].keys()):
        cam_stats = stats['camera_stats'][camera_id]
        night_pct = cam_stats['night'] / cam_stats['total'] * 100 if cam_stats['total'] > 0 else 0
        print(f"  Camera {camera_id}:")
        print(f"    總數: {cam_stats['total']}")
        print(f"    Night: {cam_stats['night']} ({night_pct:.1f}%)")
        print(f"    Day: {cam_stats['day']} ({100-night_pct:.1f}%)")
    print()
    
    # 保存詳細結果
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    
    # 保存 CSV
    csv_path = output_dir / "train_luma_aug_analysis.csv"
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write("camera_id,image,luma,is_night,will_aug\n")
        for s in stats['samples']:
            f.write(f"{s['camera_id']},{s['image']},{s['luma']:.6f},"
                   f"{s['is_night']},{s['will_aug']}\n")
    print(f"詳細結果已保存至: {csv_path}")
    
    # 保存 JSON 摘要
    json_path = output_dir / "train_luma_aug_summary.json"
    summary = {
        'luma_threshold': luma_threshold,
        'total_samples': stats['total'],
        'night_count': stats['night_count'],
        'day_count': stats['day_count'],
        'night_ratio': stats['night_count'] / stats['total'] if stats['total'] > 0 else 0,
        'luma_stats': {
            'min': float(luma_array.min()),
            'max': float(luma_array.max()),
            'mean': float(luma_array.mean()),
            'median': float(np.median(luma_array)),
            'std': float(luma_array.std())
        },
        'boundary_samples_count': len(boundary_samples),
        'camera_stats': {k: dict(v) for k, v in stats['camera_stats'].items()}
    }
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"摘要已保存至: {json_path}")
    
    return stats

if __name__ == "__main__":
    analyze_train_luma_aug()
