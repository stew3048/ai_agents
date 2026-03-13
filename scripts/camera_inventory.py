"""
SkyFinder Camera Inventory Pipeline
偵測並分析 data/ 底下所有以 skyfinder_ 開頭的 camera 資料夾

功能：
- 統計影像張數
- 刪除壞掉的圖片
- 計算平均亮度分佈（判斷是否有夜晚）
- 檢查 mask 是否可正確配對
- 輸出結果到 outputs/camera_inventory.csv

即使部分 camera 尚未下載完成，也能安全跳過並記錄狀態。
"""

import os
import csv
import glob
from pathlib import Path
from PIL import Image
import numpy as np
from typing import Dict, List, Tuple, Optional


def find_camera_folders(base_dir: str = "data") -> List[str]:
    """
    找出 data/ 底下所有以 skyfinder_ 開頭的 camera 資料夾
    
    Args:
        base_dir: 基礎目錄，預設為 "data"
    
    Returns:
        List of camera folder paths
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


def extract_camera_id(folder_path: str) -> Optional[str]:
    """從資料夾路徑提取 camera ID（從 skyfinder_<camera_id> 格式中提取）"""
    folder_name = os.path.basename(folder_path)
    if folder_name.startswith("skyfinder_"):
        return folder_name.replace("skyfinder_", "")
    return folder_name


def count_images(images_dir: str) -> Tuple[int, List[str]]:
    """
    統計影像張數，並刪除壞掉的圖片
    
    Returns:
        (count, list of image filenames)
    """
    if not os.path.exists(images_dir):
        return 0, []
    
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    image_files = []
    corrupted_count = 0
    
    for filename in os.listdir(images_dir):
        if any(filename.lower().endswith(ext) for ext in valid_extensions):
            img_path = os.path.join(images_dir, filename)
            # 檢查圖片是否損壞
            try:
                img = Image.open(img_path)
                img.verify()  # 驗證圖片完整性
                img.close()
                # 再次打開以確保可以正常讀取
                img = Image.open(img_path)
                img.load()  # 載入圖片數據
                img.close()
                image_files.append(filename)
            except Exception as e:
                # 圖片損壞，刪除它
                try:
                    os.remove(img_path)
                    corrupted_count += 1
                    print(f"    [刪除壞圖] {filename}")
                except Exception as del_err:
                    pass
    
    if corrupted_count > 0:
        print(f"    已刪除 {corrupted_count} 張壞掉的圖片")
    
    return len(image_files), sorted(image_files)


def count_masks(masks_dir: str) -> Tuple[int, List[str]]:
    """
    統計 mask 張數
    
    Returns:
        (count, list of mask filenames)
    """
    if not os.path.exists(masks_dir):
        return 0, []
    
    valid_extensions = {'.png', '.pgm', '.jpg', '.jpeg'}
    mask_files = []
    
    for filename in os.listdir(masks_dir):
        if any(filename.lower().endswith(ext) for ext in valid_extensions):
            mask_files.append(filename)
    
    return len(mask_files), sorted(mask_files)


def check_mask_pairing(images_dir: str, masks_dir: str, image_files: List[str]) -> Dict[str, any]:
    """
    檢查 mask 是否可正確配對
    
    Returns:
        Dict with pairing statistics
    """
    if not os.path.exists(images_dir) or not os.path.exists(masks_dir):
        return {
            'paired_count': 0,
            'missing_masks': len(image_files),
            'status': 'missing_dirs'
        }
    
    paired_count = 0
    missing_masks = []
    
    for img_file in image_files:
        # 嘗試找到對應的 mask
        # SkyFinder 通常使用相同的檔名，但可能副檔名不同
        img_base = os.path.splitext(img_file)[0]
        
        # 嘗試多種可能的 mask 檔名
        possible_mask_names = [
            f"{img_base}.png",
            f"{img_base}.pgm",
            f"{img_base}.jpg",
            f"{img_file}",  # 相同檔名
        ]
        
        mask_found = False
        for mask_name in possible_mask_names:
            mask_path = os.path.join(masks_dir, mask_name)
            if os.path.exists(mask_path):
                paired_count += 1
                mask_found = True
                break
        
        if not mask_found:
            missing_masks.append(img_file)
    
    status = 'complete' if len(missing_masks) == 0 else 'incomplete'
    if len(image_files) == 0:
        status = 'no_images'
    
    return {
        'paired_count': paired_count,
        'missing_masks': len(missing_masks),
        'missing_mask_files': missing_masks[:10],  # 只記錄前 10 個
        'status': status
    }


def srgb_to_linear(v):
    """
    sRGB [0,1]（gamma 編碼）→ linear [0,1]。
    係數 0.2126/0.7152/0.0722 的 luma 公式必須用在 linear RGB，直接用在 sRGB 會高估暗部。
    """
    v = np.clip(np.asarray(v, dtype=np.float64), 0, 1)
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4).astype(np.float32)


def calculate_brightness_stats(images_dir: str, image_files: List[str], sample_size: int = 50) -> Dict[str, float]:
    """
    計算平均亮度分佈（用來判斷是否有夜晚）
    
    使用與訓練/評估一致的計算方式：
    - sRGB → linear RGB 轉換
    - luma = 0.2126*R_lin + 0.7152*G_lin + 0.0722*B_lin (ITU-R BT.709)
    - night 判斷：luma < 0.20（linear RGB 空間，值域 0-1）
    
    Args:
        images_dir: 圖片目錄
        image_files: 圖片檔案列表
        sample_size: 取樣數量（如果圖片太多，只取樣部分）
    
    Returns:
        Dict with brightness statistics
    """
    if not os.path.exists(images_dir) or len(image_files) == 0:
        return {
            'mean_brightness': 0.0,
            'std_brightness': 0.0,
            'min_brightness': 0.0,
            'max_brightness': 0.0,
            'night_ratio': 0.0,  # luma < 0.20 的比例（linear RGB 空間）
            'samples_processed': 0,
            'status': 'no_images'
        }
    
    # 取樣（如果圖片太多）
    if len(image_files) > sample_size:
        step = len(image_files) // sample_size
        sampled_files = image_files[::step][:sample_size]
    else:
        sampled_files = image_files
    
    luma_values = []
    night_count = 0
    luma_threshold = 0.20  # 與訓練/評估一致
    
    for img_file in sampled_files:
        try:
            img_path = os.path.join(images_dir, img_file)
            img = Image.open(img_path)
            
            # 轉換為 RGB（如果是 RGBA 或其他格式）
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # 轉換為 numpy array，並正規化到 [0, 1]
            img_array = np.array(img, dtype=np.float32) / 255.0  # [H, W, 3], [0, 1]
            
            # sRGB → linear RGB
            R_lin = srgb_to_linear(img_array[:, :, 0])
            G_lin = srgb_to_linear(img_array[:, :, 1])
            B_lin = srgb_to_linear(img_array[:, :, 2])
            
            # 計算 luma (ITU-R BT.709)
            luma = 0.2126 * R_lin + 0.7152 * G_lin + 0.0722 * B_lin
            mean_luma = np.mean(luma)
            
            luma_values.append(mean_luma)
            
            # 判斷是否為夜晚（luma < 0.20，與訓練/評估一致）
            if mean_luma < luma_threshold:
                night_count += 1
                
        except Exception as e:
            # 如果讀取失敗，跳過
            continue
    
    if len(luma_values) == 0:
        return {
            'mean_brightness': 0.0,
            'std_brightness': 0.0,
            'min_brightness': 0.0,
            'max_brightness': 0.0,
            'night_ratio': 0.0,
            'samples_processed': 0,
            'status': 'read_error'
        }
    
    luma_array = np.array(luma_values)
    
    # 為了向後相容，mean_brightness 仍使用 luma 值（但現在是 linear RGB 空間的 luma，值域 0-1）
    # 如果需要顯示為 0-255 範圍，可以乘以 255，但這裡保持 0-1 範圍以與訓練/評估一致
    return {
        'mean_brightness': float(np.mean(luma_array) * 255.0),  # 轉換為 0-255 範圍以保持向後相容
        'std_brightness': float(np.std(luma_array) * 255.0),
        'min_brightness': float(np.min(luma_array) * 255.0),
        'max_brightness': float(np.max(luma_array) * 255.0),
        'night_ratio': night_count / len(luma_values),  # luma < 0.20 的比例
        'samples_processed': len(luma_values),
        'status': 'success'
    }


def analyze_camera(camera_folder: str) -> Dict[str, any]:
    """
    分析單個 camera 資料夾
    
    Returns:
        Dict with analysis results
    """
    camera_id = extract_camera_id(camera_folder)
    if not camera_id:
        camera_id = os.path.basename(camera_folder) or 'unknown'
    
    images_dir = os.path.join(camera_folder, "images")
    masks_dir = os.path.join(camera_folder, "masks")
    
    # 統計影像
    image_count, image_files = count_images(images_dir)
    
    # 統計 masks
    mask_count, mask_files = count_masks(masks_dir)
    
    # 檢查配對
    pairing_info = check_mask_pairing(images_dir, masks_dir, image_files)
    
    # 計算亮度統計（只在有圖片時計算）
    brightness_info = {}
    if image_count > 0:
        brightness_info = calculate_brightness_stats(images_dir, image_files)
    else:
        brightness_info = {
            'mean_brightness': 0.0,
            'std_brightness': 0.0,
            'min_brightness': 0.0,
            'max_brightness': 0.0,
            'night_ratio': 0.0,
            'samples_processed': 0,
            'status': 'no_images'
        }
    
    # 檢查資料夾狀態
    has_images_dir = os.path.exists(images_dir)
    has_masks_dir = os.path.exists(masks_dir)
    
    if not has_images_dir and not has_masks_dir:
        folder_status = 'not_downloaded'
    elif has_images_dir and image_count == 0:
        folder_status = 'downloading'  # 可能正在下載中
    elif has_images_dir and image_count > 0 and pairing_info['status'] == 'complete':
        folder_status = 'ready'
    elif has_images_dir and image_count > 0 and pairing_info['status'] == 'incomplete':
        folder_status = 'incomplete'
    else:
        folder_status = 'unknown'
    
    return {
        'camera_id': camera_id,
        'folder_path': camera_folder,
        'image_count': image_count,
        'mask_count': mask_count,
        'paired_count': pairing_info['paired_count'],
        'missing_masks': pairing_info['missing_masks'],
        'pairing_status': pairing_info['status'],
        'mean_brightness': brightness_info['mean_brightness'],
        'std_brightness': brightness_info['std_brightness'],
        'min_brightness': brightness_info['min_brightness'],
        'max_brightness': brightness_info['max_brightness'],
        'night_ratio': brightness_info['night_ratio'],
        'brightness_samples': brightness_info['samples_processed'],
        'brightness_status': brightness_info['status'],
        'folder_status': folder_status,
        'has_images_dir': has_images_dir,
        'has_masks_dir': has_masks_dir,
    }


def save_inventory_csv(results: List[Dict], output_path: str = "outputs/camera_inventory.csv"):
    """將結果儲存為 CSV"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if len(results) == 0:
        print("  No cameras found to save.")
        return
    
    fieldnames = [
        'camera_id',
        'folder_status',
        'image_count',
        'mask_count',
        'paired_count',
        'missing_masks',
        'pairing_status',
        'mean_brightness',
        'std_brightness',
        'min_brightness',
        'max_brightness',
        'night_ratio',
        'brightness_samples',
        'brightness_status',
        'folder_path',
    ]
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for result in results:
            # 只寫入需要的欄位
            row = {field: result.get(field, '') for field in fieldnames}
            writer.writerow(row)
    
    print(f"  Inventory saved to: {output_path}")


def main():
    print("=" * 60)
    print("  SkyFinder Camera Inventory Pipeline")
    print("=" * 60)
    print()
    
    # 找出所有 camera 資料夾
    print("[1] Scanning for camera folders...")
    camera_folders = find_camera_folders()
    print(f"  Found {len(camera_folders)} camera folder(s)")
    
    if len(camera_folders) == 0:
        print("  No camera folders found in data/")
        print("  Expected format: data/skyfinder_<camera_id>/")
        return
    
    print()
    print("[2] Analyzing cameras...")
    print()
    
    results = []
    for i, folder in enumerate(camera_folders, 1):
        camera_id = extract_camera_id(folder)
        print(f"  [{i}/{len(camera_folders)}] Camera {camera_id}...", end=" ", flush=True)
        
        try:
            result = analyze_camera(folder)
            if result:
                results.append(result)
                status = result.get('folder_status', 'unknown')
                img_count = result.get('image_count', 0)
                print(f"[{status}] {img_count} images")
            else:
                # 如果分析失敗，至少記錄基本資訊
                print("[ERROR] Analysis failed")
                results.append({
                    'camera_id': camera_id or 'unknown',
                    'folder_path': folder,
                    'folder_status': 'error',
                    'image_count': 0,
                })
        except Exception as e:
            print(f"[ERROR] {str(e)}")
            # 即使出錯也記錄
            results.append({
                'camera_id': camera_id or 'unknown',
                'folder_path': folder,
                'folder_status': 'error',
                'error_message': str(e),
                'image_count': 0,
            })
    
    print()
    print("[3] Saving inventory...")
    save_inventory_csv(results)
    
    print()
    print("=" * 60)
    print("  Summary")
    print("=" * 60)
    print()
    
    # 統計
    total_cameras = len(results)
    ready_cameras = sum(1 for r in results if r.get('folder_status') == 'ready')
    downloading_cameras = sum(1 for r in results if r.get('folder_status') == 'downloading')
    incomplete_cameras = sum(1 for r in results if r.get('folder_status') == 'incomplete')
    not_downloaded_cameras = sum(1 for r in results if r.get('folder_status') == 'not_downloaded')
    
    total_images = sum(r.get('image_count', 0) for r in results)
    total_paired = sum(r.get('paired_count', 0) for r in results)
    
    print(f"  Total cameras: {total_cameras}")
    print(f"    Ready: {ready_cameras}")
    print(f"    Downloading: {downloading_cameras}")
    print(f"    Incomplete: {incomplete_cameras}")
    print(f"    Not downloaded: {not_downloaded_cameras}")
    print()
    print(f"  Total images: {total_images}")
    print(f"  Total paired: {total_paired}")
    print()
    
    # 顯示有夜晚場景的 cameras
    cameras_with_night = [r for r in results if r.get('night_ratio', 0) > 0.1]
    if cameras_with_night:
        print(f"  Cameras with night scenes (night_ratio > 0.1): {len(cameras_with_night)}")
        for r in cameras_with_night[:5]:  # 只顯示前 5 個
            print(f"    - Camera {r['camera_id']}: night_ratio={r['night_ratio']:.2%}")
    print()


if __name__ == "__main__":
    main()
