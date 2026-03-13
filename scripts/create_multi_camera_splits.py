"""
建立 Multi-Camera 資料集的 Train/Val/Test/Sanity 分類

跨場景分類規則（按相機分類）：
- train_cameras: 從已可用的 cameras 中選 4 個（不含 10870，務必含 10066）
- val_camera: 從剩下的 cameras 中選 1 個（unseen）
- test_camera: 固定 10870（整個 camera 全部資料）

split 規則：
- train: train_cameras 的所有影像（或每 camera 前 90%）
- sanity: train_cameras 每個 camera 的最後 10%
- val: val_camera 的全部影像
- test: 10870 的全部影像

輸出：outputs/multi_camera_splits.json
"""

import os
import json
import random
import csv
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from PIL import Image


def find_camera_folders(base_dir: str = "data") -> List[str]:
    """找出所有相機資料夾"""
    if not os.path.exists(base_dir):
        return []
    
    folders = []
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if (os.path.isdir(item_path) and 
            item.startswith("skyfinder_") and 
            item not in ["skyfinder_masks", "skyfinder_sample"]):
            folders.append(item_path)
    
    return sorted(folders)


def extract_camera_id(folder_path: str) -> str:
    """從資料夾路徑提取 camera ID"""
    folder_name = os.path.basename(folder_path)
    if folder_name.startswith("skyfinder_"):
        return folder_name.replace("skyfinder_", "")
    return folder_name


def get_ready_cameras() -> List[str]:
    """
    從 camera_inventory.csv 取得所有 ready 狀態的相機 ID
    
    Returns:
        List of ready camera IDs
    """
    inventory_file = 'outputs/camera_inventory.csv'
    
    if not os.path.exists(inventory_file):
        # 如果沒有 inventory 檔案，直接檢查資料夾
        print("  警告：找不到 camera_inventory.csv，直接檢查資料夾...")
        camera_folders = find_camera_folders()
        ready_cameras = []
        for folder in camera_folders:
            camera_id = extract_camera_id(folder)
            images_dir = os.path.join(folder, "images")
            if os.path.exists(images_dir):
                image_files = [f for f in os.listdir(images_dir) 
                             if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if len(image_files) > 0:
                    ready_cameras.append(camera_id)
        return ready_cameras
    
    ready_cameras = []
    with open(inventory_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('folder_status') == 'ready':
                ready_cameras.append(row['camera_id'])
    
    return ready_cameras


def is_image_valid(img_path: str) -> bool:
    """
    檢查圖片是否有效（未受損）
    
    Returns:
        True if image is valid, False otherwise
    """
    try:
        img = Image.open(img_path)
        img.verify()  # 驗證圖片結構完整性
        img.close()
        
        # 重新打開並載入數據（verify 後需要重新打開）
        img = Image.open(img_path)
        img.load()  # 載入圖片數據
        
        # 檢查尺寸是否合理
        if img.size[0] <= 0 or img.size[1] <= 0:
            img.close()
            return False
        
        # 嘗試轉換為 RGB（測試是否能正常處理）
        if img.mode not in ['RGB', 'RGBA', 'L', 'P']:
            try:
                img.convert('RGB')
            except:
                img.close()
                return False
        
        img.close()
        return True
    except Exception:
        return False


def get_camera_images(camera_folder: str) -> List[Dict[str, str]]:
    """
    取得單個相機的所有有效圖片-mask 配對（按檔名排序）
    會自動跳過受損的圖片
    
    Returns:
        List of dicts with 'image', 'mask', and 'camera_id' keys
    """
    images_dir = os.path.join(camera_folder, "images")
    masks_dir = os.path.join(camera_folder, "masks")
    
    if not os.path.exists(images_dir):
        return []
    
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    pairs = []
    
    for filename in sorted(os.listdir(images_dir)):
        if any(filename.lower().endswith(ext) for ext in valid_extensions):
            img_path = os.path.join(images_dir, filename)
            
            # 檢查圖片是否受損
            if not is_image_valid(img_path):
                continue  # 跳過受損的圖片
            
            base_name = os.path.splitext(filename)[0]
            
            # 嘗試找到對應的 mask
            mask_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']
            mask_found = False
            
            for ext in mask_extensions:
                mask_name = base_name + ext
                mask_path = os.path.join(masks_dir, mask_name)
                if os.path.exists(mask_path):
                    # 也檢查 mask 是否有效
                    if is_image_valid(mask_path):
                        pairs.append({
                            'image': filename,
                            'mask': mask_name,
                            'camera_id': extract_camera_id(camera_folder)
                        })
                        mask_found = True
                        break
            
            # 如果找不到有效的 mask，跳過這張圖片
            if not mask_found:
                continue
    
    return pairs


def create_cross_camera_splits(
    train_cameras: List[str],
    val_camera: str,
    test_camera: str = '10870',
    train_ratio: float = 0.9
) -> Dict[str, List[Dict]]:
    """
    建立跨場景分類（按相機分類，包含 sanity set）
    
    參數:
        train_cameras: 用於訓練的相機 ID 列表（必須包含 10066，不含 10870）
        val_camera: 用於驗證的相機 ID（單個）
        test_camera: 用於測試的相機 ID（預設 10870）
        train_ratio: train_cameras 中每台相機用於訓練的比例（剩餘的用於 sanity）
    
    返回:
        Dict with 'train', 'sanity', 'val', 'test' keys
    """
    print(f"  分類策略: 跨場景分類（按相機）")
    print(f"    Train cameras: {train_cameras} (每台前 {train_ratio*100:.0f}% 用於 train，後 {100-train_ratio*100:.0f}% 用於 sanity)")
    print(f"    Val camera: {val_camera} (全部)")
    print(f"    Test camera: {test_camera} (全部)")
    print()
    
    train_data = []
    sanity_data = []
    val_data = []
    test_data = []
    
    camera_folders = find_camera_folders()
    camera_folder_map = {extract_camera_id(f): f for f in camera_folders}
    
    # 處理 train_cameras（分成 train 和 sanity）
    for camera_id in train_cameras:
        if camera_id not in camera_folder_map:
            print(f"  警告：找不到 Camera {camera_id}，跳過")
            continue
        
        folder = camera_folder_map[camera_id]
        pairs = get_camera_images(folder)
        
        if len(pairs) == 0:
            print(f"  警告：Camera {camera_id} 沒有圖片，跳過")
            continue
        
        # 正常處理：前 90% train，後 10% sanity
        n = len(pairs)
        train_end = int(n * train_ratio)
        
        # 前 90% 用於 train
        train_data.extend(pairs[:train_end])
        
        # 後 10% 用於 sanity
        if train_end < n:
            sanity_data.extend(pairs[train_end:])
        
        print(f"    Camera {camera_id}: {len(pairs)} 張 -> train: {train_end}, sanity: {n - train_end}")
    
    # 處理 val_camera（全部用於 val）
    if val_camera in camera_folder_map:
        folder = camera_folder_map[val_camera]
        pairs = get_camera_images(folder)
        val_data.extend(pairs)
        print(f"    Camera {val_camera} (val): {len(pairs)} 張")
    else:
        print(f"  警告：找不到 Val Camera {val_camera}")
    
    # 處理 test_camera（全部，固定 10870）
    if test_camera in camera_folder_map:
        folder = camera_folder_map[test_camera]
        pairs = get_camera_images(folder)
        test_data.extend(pairs)
        print(f"    Camera {test_camera} (test): {len(pairs)} 張")
    else:
        print(f"  警告：找不到 Test Camera {test_camera}")
    
    print()
    
    return {
        'train': train_data,
        'sanity': sanity_data,
        'val': val_data,
        'test': test_data
    }


def create_splits(
    train_cameras: List[str] = None,
    val_camera: str = None,
    test_camera: str = '10870',
    train_ratio: float = 0.9,
    seed: int = 42,
    output_file: str = 'outputs/multi_camera_splits.json'
) -> Dict:
    """
    建立跨場景多相機資料集分類
    
    參數:
        train_cameras: 用於訓練的相機 ID 列表（如果 None，自動選擇 4 個，必須包含 10066，不含 10870）
        val_camera: 用於驗證的相機 ID（如果 None，自動從剩餘的選擇 1 個）
        test_camera: 用於測試的相機 ID（預設 10870）
        train_ratio: train_cameras 中每台相機用於訓練的比例（剩餘的用於 sanity，預設 0.9）
        seed: 隨機種子（用於自動選擇相機）
        output_file: 輸出 JSON 檔案路徑
    """
    print("=" * 60)
    print("  Multi-Camera Cross-Scene Dataset Split Creation")
    print("=" * 60)
    print()
    
    # 取得所有 ready 的相機
    ready_cameras = get_ready_cameras()
    
    if len(ready_cameras) == 0:
        raise ValueError("找不到任何 ready 狀態的相機！")
    
    print(f"找到 {len(ready_cameras)} 個 ready 相機: {ready_cameras}")
    print()
    
    # 排除 test_camera (10870)
    available_cameras = [c for c in ready_cameras if c != test_camera]
    
    print(f"  排除 test_camera ({test_camera}) 後，可用相機: {available_cameras} ({len(available_cameras)} 個)")
    
    # 自動選擇 train_cameras（如果未指定）
    if train_cameras is None:
        # 必須包含 10066
        if '10066' not in available_cameras:
            raise ValueError("找不到 Camera 10066！")
        
        random.seed(seed)
        candidates = [c for c in available_cameras if c != '10066']
        random.shuffle(candidates)
        
        # 如果可用相機 >= 5 個，選 4 個作為 train（包含 10066）
        # 如果可用相機 = 4 個，選 3 個作為 train（包含 10066），剩下 1 個作為 val
        if len(available_cameras) >= 5:
            train_cameras = ['10066'] + candidates[:3]  # 10066 + 其他 3 個 = 4 個
        elif len(available_cameras) == 4:
            train_cameras = ['10066'] + candidates[:2]  # 10066 + 其他 2 個 = 3 個
            print(f"  注意：可用相機只有 4 個，選 3 個作為 train_cameras，剩下 1 個將作為 val")
        else:
            raise ValueError(f"可用相機不足！需要至少 4 個（目前 {len(available_cameras)} 個），但 test_camera {test_camera} 已被排除")
        
        print(f"  自動選擇 train_cameras: {train_cameras}")
    else:
        # 驗證 train_cameras
        if '10066' not in train_cameras:
            raise ValueError("train_cameras 必須包含 10066！")
        if test_camera in train_cameras:
            raise ValueError(f"train_cameras 不能包含 test_camera ({test_camera})！")
        if len(train_cameras) < 3:
            raise ValueError(f"train_cameras 至少需要 3 個，目前有 {len(train_cameras)} 個")
    
    # 自動選擇 val_camera（如果未指定）
    if val_camera is None:
        # 從剩餘的相機中選 1 個
        remaining = [c for c in available_cameras if c not in train_cameras]
        if len(remaining) == 0:
            raise ValueError("沒有剩餘的相機可用於 val！請減少 train_cameras 的數量或手動指定 val_camera")
        
        random.seed(seed + 1)  # 使用不同的種子
        val_camera = random.choice(remaining)
        print(f"  自動選擇 val_camera: {val_camera}")
    
    # 驗證 val_camera
    if val_camera in train_cameras:
        raise ValueError(f"val_camera ({val_camera}) 不能在 train_cameras 中！")
    if val_camera == test_camera:
        raise ValueError(f"val_camera ({val_camera}) 不能是 test_camera ({test_camera})！")
    
    print()
    
    # 統計每個相機的圖片數量
    camera_folders = find_camera_folders()
    camera_folder_map = {extract_camera_id(f): f for f in camera_folders}
    
    camera_stats = {}
    for camera_id in train_cameras + [val_camera, test_camera]:
        if camera_id in camera_folder_map:
            folder = camera_folder_map[camera_id]
            pairs = get_camera_images(folder)
            camera_stats[camera_id] = len(pairs)
            print(f"  Camera {camera_id}: {len(pairs)} 張圖片")
    
    print()
    
    # 建立分類
    splits = create_cross_camera_splits(
        train_cameras=train_cameras,
        val_camera=val_camera,
        test_camera=test_camera,
        train_ratio=train_ratio
    )
    
    # 建立 metadata
    metadata = {
        'strategy': 'cross_camera_split',
        'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'train_cameras': train_cameras,
        'val_camera': val_camera,
        'test_camera': test_camera,
        'train_ratio': train_ratio,  # train_cameras 中每台相機用於 train 的比例
        'sanity_ratio': 1.0 - train_ratio,  # train_cameras 中每台相機用於 sanity 的比例
        'seed': seed,
        'camera_stats': camera_stats,
        'note': 'Cross-scene evaluation: train on some cameras, test on unseen camera (10870)'
    }
    
    # 建立完整的 splits 資料結構
    result = {
        'metadata': metadata,
        'train': splits['train'],
        'sanity': splits['sanity'],
        'val': splits['val'],
        'test': splits['test']
    }
    
    # 儲存 JSON
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print("=" * 60)
    print("  Summary")
    print("=" * 60)
    print()
    print(f"  Train:  {len(result['train'])} samples")
    print(f"  Sanity: {len(result['sanity'])} samples")
    print(f"  Val:    {len(result['val'])} samples")
    print(f"  Test:   {len(result['test'])} samples")
    print()
    print(f"  Splits saved to: {output_file}")
    print()
    
    return result


def main():
    """主函數 - 使用預設設定"""
    # 跨場景分類規則：
    # - train_cameras: 4 個（不含 10870，務必含 10066）
    # - val_camera: 1 個（從剩餘的選擇）
    # - test_camera: 固定 10870
    
    create_splits(
        train_cameras=None,  # 自動選擇 4 個（包含 10066）
        val_camera=None,     # 自動從剩餘的選擇 1 個
        test_camera='10870', # 固定為 10870
        train_ratio=0.9,     # train_cameras 中每台相機前 90% 用於 train，後 10% 用於 sanity
        seed=42,
        output_file='outputs/multi_camera_splits.json'
    )


if __name__ == '__main__':
    main()
