"""
批量下載多個 SkyFinder camera 的樣本圖片
用於快速驗證每個 camera 的場景特性，決定是否用於訓練

使用方式：
    python scripts/download_multiple_cameras_sample.py --cameras 9417 10204 10523 10789 --samples_per_camera 20
    python scripts/download_multiple_cameras_sample.py --cameras 9417 10204 10523 10789 --samples_per_camera 10 --output_dir data/camera_samples
"""

import os
import argparse
import urllib.request
import zipfile
import shutil
from PIL import Image

# Zenodo base URL
ZENODO_BASE = "https://zenodo.org/records/5884485/files"


def download_file(url, dest_path):
    """下載檔案並顯示進度"""
    print(f"  Downloading: {url}")
    print(f"  To: {dest_path}")
    
    def progress_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100, downloaded * 100 / total_size)
            mb_downloaded = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            print(f"\r  Progress: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end="", flush=True)
    
    try:
        urllib.request.urlretrieve(url, dest_path, progress_hook)
        print()  # 換行
        return True
    except Exception as e:
        print(f"\n  Error: {e}")
        return False


def download_camera_sample(camera_id, samples_per_camera, output_base_dir, shared_masks_dir=None):
    """
    下載單個 camera 的樣本圖片
    
    Args:
        camera_id: Camera ID (str)
        samples_per_camera: 每個 camera 下載的圖片數量
        output_base_dir: 輸出基礎目錄
        shared_masks_dir: 共用的 masks 目錄（如果已下載過）
    
    Returns:
        (success: bool, image_count: int, mask_found: bool)
    """
    print("=" * 60)
    print(f"  Processing Camera {camera_id}")
    print("=" * 60)
    print()
    
    # 設定目錄
    download_dir = os.path.join(output_base_dir, "downloads", f"camera_{camera_id}")
    output_dir = os.path.join(output_base_dir, f"camera_{camera_id}")
    
    # 建立目錄
    os.makedirs(download_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "masks"), exist_ok=True)
    
    # 1. 處理 masks（使用共用或下載）
    masks_extract_dir = shared_masks_dir
    if not masks_extract_dir or not os.path.exists(masks_extract_dir):
        print("[1] Downloading masks (if needed)...")
        masks_zip = os.path.join(download_dir, "skyfinder_masks.zip")
        masks_extract_dir = os.path.join(download_dir, "masks_extracted")
        
        if not os.path.exists(masks_extract_dir):
            if not os.path.exists(masks_zip):
                masks_url = f"{ZENODO_BASE}/skyfinder_masks.zip?download=1"
                if not download_file(masks_url, masks_zip):
                    print("  Failed to download masks. Exiting.")
                    return False, 0, False
            else:
                print("  Masks already downloaded.")
            
            # 解壓縮 masks
            print("  Extracting masks...")
            with zipfile.ZipFile(masks_zip, 'r') as zip_ref:
                zip_ref.extractall(masks_extract_dir)
            print("  Masks extracted.")
        else:
            print("  Masks already extracted.")
    else:
        print("[1] Using shared masks...")
    
    # 2. 下載 camera 圖片
    print()
    print(f"[2] Downloading camera {camera_id} images...")
    camera_url = f"{ZENODO_BASE}/{camera_id}.zip?download=1"
    camera_zip = os.path.join(download_dir, f"{camera_id}.zip")
    
    if not os.path.exists(camera_zip):
        if not download_file(camera_url, camera_zip):
            print(f"  Failed to download camera {camera_id} images.")
            print(f"  Note: This camera might not exist in the dataset.")
            return False, 0, False
    else:
        print("  Camera images already downloaded.")
    
    # 3. 解壓縮 camera 圖片
    print()
    print(f"[3] Extracting camera {camera_id} images...")
    camera_extract_dir = os.path.join(download_dir, f"{camera_id}_extracted")
    if not os.path.exists(camera_extract_dir):
        with zipfile.ZipFile(camera_zip, 'r') as zip_ref:
            zip_ref.extractall(camera_extract_dir)
        print("  Camera images extracted.")
    else:
        print("  Camera images already extracted.")
    
    # 4. 複製樣本圖片
    print()
    print(f"[4] Copying {samples_per_camera} sample images...")
    image_count = 0
    
    for root, dirs, files in os.walk(camera_extract_dir):
        for f in sorted(files):
            if f.endswith(('.jpg', '.png', '.jpeg')):
                src = os.path.join(root, f)
                dst = os.path.join(output_dir, "images", f"{image_count+1:03d}.jpg")
                shutil.copy2(src, dst)
                image_count += 1
                
                if image_count >= samples_per_camera:
                    break
        if image_count >= samples_per_camera:
            break
    
    print(f"  Copied {image_count} images to {output_dir}/images/")
    
    # 5. 處理 mask
    print()
    print("[5] Processing masks...")
    
    # 找到 mask 檔案
    mask_src = None
    for root, dirs, files in os.walk(masks_extract_dir):
        for f in files:
            if f.startswith(camera_id) or camera_id in f:
                if f.endswith(('.png', '.pgm', '.jpg')):
                    mask_src = os.path.join(root, f)
                    print(f"  Found mask file: {mask_src}")
                    break
        if mask_src:
            break
    
    mask_found = False
    if mask_src:
        # 為每張圖片複製相同的 mask
        mask_img = Image.open(mask_src)
        print(f"  Mask size: {mask_img.size}")
        print(f"  Mask mode: {mask_img.mode}")
        
        for i in range(image_count):
            dst = os.path.join(output_dir, "masks", f"{i+1:03d}.png")
            mask_img.save(dst)
        
        print(f"  Copied mask to {image_count} files in {output_dir}/masks/")
        mask_found = True
    else:
        print(f"  Warning: Could not find mask file for camera {camera_id}!")
        print("  Available mask files (first 10):")
        mask_files = []
        for root, dirs, files in os.walk(masks_extract_dir):
            for f in files[:10]:
                mask_files.append(f)
        for f in mask_files[:10]:
            print(f"    - {f}")
    
    print()
    print(f"  [OK] Camera {camera_id} completed: {image_count} images, mask={'found' if mask_found else 'NOT found'}")
    print()
    
    return True, image_count, mask_found


def main():
    parser = argparse.ArgumentParser(
        description='批量下載多個 SkyFinder camera 的樣本圖片',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例：
  # 下載 4 個 camera，每個 20 張圖片
  python scripts/download_multiple_cameras_sample.py --cameras 9417 10204 10523 10789 --samples_per_camera 20
  
  # 下載到指定目錄，每個 10 張圖片
  python scripts/download_multiple_cameras_sample.py --cameras 9417 10204 10523 10789 --samples_per_camera 10 --output_dir data/camera_samples
        """
    )
    parser.add_argument('--cameras', type=str, nargs='+', required=True,
                       help='要下載的 Camera ID 列表（例如：9417 10204 10523 10789）')
    parser.add_argument('--samples_per_camera', type=int, default=20,
                       help='每個 camera 下載的圖片數量（預設：20）')
    parser.add_argument('--output_dir', type=str, default='data/camera_samples',
                       help='輸出目錄（預設：data/camera_samples）')
    parser.add_argument('--use_shared_masks', type=str, default=None,
                       help='使用共用的 masks 目錄（例如：data/skyfinder_sample/downloads/masks_extracted）')
    
    args = parser.parse_args()
    
    camera_ids = args.cameras
    samples_per_camera = args.samples_per_camera
    output_dir = args.output_dir
    
    # 檢查是否有共用的 masks 目錄
    shared_masks_dir = args.use_shared_masks
    if not shared_masks_dir:
        # 自動檢查是否有已下載的 masks
        possible_shared_dirs = [
            "data/skyfinder_sample/downloads/masks_extracted",
            "data/skyfinder_10066/downloads/masks_extracted",
        ]
        for dir_path in possible_shared_dirs:
            if os.path.exists(dir_path):
                shared_masks_dir = dir_path
                print(f"Found shared masks directory: {shared_masks_dir}")
                break
    
    print("=" * 60)
    print("  SkyFinder Multi-Camera Sample Downloader")
    print("=" * 60)
    print()
    print(f"  Cameras to download: {', '.join(camera_ids)}")
    print(f"  Samples per camera: {samples_per_camera}")
    print(f"  Output directory: {output_dir}")
    if shared_masks_dir:
        print(f"  Shared masks: {shared_masks_dir}")
    print()
    
    # 建立輸出目錄
    os.makedirs(output_dir, exist_ok=True)
    
    # 下載每個 camera
    results = []
    for camera_id in camera_ids:
        success, image_count, mask_found = download_camera_sample(
            camera_id, 
            samples_per_camera, 
            output_dir,
            shared_masks_dir
        )
        results.append({
            'camera_id': camera_id,
            'success': success,
            'image_count': image_count,
            'mask_found': mask_found
        })
    
    # 總結
    print("=" * 60)
    print("  Download Summary")
    print("=" * 60)
    print()
    
    for result in results:
        status = "OK" if result['success'] else "FAIL"
        mask_status = "OK" if result['mask_found'] else "FAIL"
        print(f"  [{status:4}] Camera {result['camera_id']:>6}: "
              f"{result['image_count']:>3} images, mask: {mask_status}")
    
    print()
    print("=" * 60)
    print("  Next Steps")
    print("=" * 60)
    print()
    print("  1. 查看每個 camera 的樣本圖片，驗證場景特性：")
    for camera_id in camera_ids:
        print(f"     - {output_dir}/camera_{camera_id}/images/")
    print()
    print("  2. 根據場景特性決定是否使用這些 camera 進行訓練")
    print("  3. 如果特性符合需求，可以下載完整資料集")
    print()


if __name__ == "__main__":
    main()
