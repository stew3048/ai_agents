"""
分析 multi_camera_splits.json 並印出統計資訊
"""

import json
import os
from collections import Counter
from datetime import datetime


def extract_time_from_filename(filename):
    """
    嘗試從檔名提取時間資訊
    如果檔名不包含時間，返回 None
    """
    # 常見的時間格式：YYYYMMDD_HHMMSS, YYYY-MM-DD_HH-MM-SS, timestamp 等
    # 目前檔名格式是 "001.jpg"，沒有時間資訊
    return None


def get_file_mtime(filepath):
    """取得檔案修改時間"""
    try:
        return os.path.getmtime(filepath)
    except:
        return None


def analyze_splits(json_file='outputs/multi_camera_splits.json'):
    """分析 splits 並印出統計資訊"""
    
    # 讀取 JSON
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    metadata = data['metadata']
    train = data['train']
    sanity = data['sanity']
    val = data['val']
    test = data['test']
    
    print("=" * 60)
    print("  Split 統計")
    print("=" * 60)
    print()
    
    # 1. 各 split 張數
    print("1. 各 Split 張數：")
    print(f"   Train:  {len(train)} 張")
    print(f"   Sanity: {len(sanity)} 張")
    print(f"   Val:    {len(val)} 張")
    print(f"   Test:   {len(test)} 張")
    print(f"   總計:   {len(train) + len(sanity) + len(val) + len(test)} 張")
    print()
    
    # 2. 各 split 包含哪些 camera
    print("2. 各 Split 包含的 Camera：")
    
    train_cameras = Counter([item['camera_id'] for item in train])
    sanity_cameras = Counter([item['camera_id'] for item in sanity])
    val_cameras = Counter([item['camera_id'] for item in val])
    test_cameras = Counter([item['camera_id'] for item in test])
    
    print(f"   Train cameras: {sorted(train_cameras.keys())}")
    for cam_id, count in sorted(train_cameras.items()):
        print(f"     - Camera {cam_id}: {count} 張")
    
    print(f"   Sanity cameras: {sorted(sanity_cameras.keys())}")
    for cam_id, count in sorted(sanity_cameras.items()):
        print(f"     - Camera {cam_id}: {count} 張")
    
    print(f"   Val camera: {sorted(val_cameras.keys())}")
    for cam_id, count in sorted(val_cameras.items()):
        print(f"     - Camera {cam_id}: {count} 張")
    
    print(f"   Test camera: {sorted(test_cameras.keys())}")
    for cam_id, count in sorted(test_cameras.items()):
        print(f"     - Camera {cam_id}: {count} 張")
    print()
    
    # 3. Sanity 的時間範圍
    print("3. Sanity 的時間範圍：")
    
    if len(sanity) == 0:
        print("   Sanity set 為空")
    else:
        # 嘗試從檔名提取時間，或使用檔案修改時間
        sanity_files_by_camera = {}
        for item in sanity:
            cam_id = item['camera_id']
            if cam_id not in sanity_files_by_camera:
                sanity_files_by_camera[cam_id] = []
            sanity_files_by_camera[cam_id].append(item['image'])
        
        # 檢查檔案修改時間
        base_dir = "data"
        time_ranges = {}
        
        for cam_id, filenames in sanity_files_by_camera.items():
            camera_folder = os.path.join(base_dir, f"skyfinder_{cam_id}")
            images_dir = os.path.join(camera_folder, "images")
            
            if os.path.exists(images_dir):
                file_times = []
                for filename in sorted(filenames):
                    filepath = os.path.join(images_dir, filename)
                    mtime = get_file_mtime(filepath)
                    if mtime:
                        file_times.append(mtime)
                
                if file_times:
                    min_time = min(file_times)
                    max_time = max(file_times)
                    time_ranges[cam_id] = {
                        'min': datetime.fromtimestamp(min_time),
                        'max': datetime.fromtimestamp(max_time),
                        'count': len(file_times)
                    }
        
        if time_ranges:
            for cam_id, time_range in sorted(time_ranges.items()):
                print(f"   Camera {cam_id}:")
                print(f"     檔案數量: {time_range['count']} 張")
                print(f"     時間範圍: {time_range['min'].strftime('%Y-%m-%d %H:%M:%S')} ~ {time_range['max'].strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            print("   無法取得檔案時間資訊（檔案可能不存在）")
        
        # 也顯示檔名範圍（如果檔名有序號）
        print()
        print("   Sanity 檔名範圍（按檔名排序）：")
        for cam_id, filenames in sorted(sanity_files_by_camera.items()):
            sorted_files = sorted(filenames)
            if len(sorted_files) > 0:
                print(f"     Camera {cam_id}: {sorted_files[0]} ~ {sorted_files[-1]} ({len(sorted_files)} 張)")
    
    print()
    print("=" * 60)
    print()
    
    # 額外資訊：metadata
    print("Metadata 資訊：")
    print(f"   策略: {metadata['strategy']}")
    print(f"   建立時間: {metadata['created_at']}")
    print(f"   Train cameras: {metadata['train_cameras']}")
    print(f"   Val camera: {metadata['val_camera']}")
    print(f"   Test camera: {metadata['test_camera']}")
    print(f"   Train ratio: {metadata['train_ratio']}")
    print(f"   Sanity ratio: {metadata['sanity_ratio']:.2f}")
    print()


if __name__ == '__main__':
    analyze_splits()
