"""
下載 SkyFinder 資料集的指定 camera
從 Zenodo 下載圖片和對應的 mask

Zenodo DOI: 10.5281/zenodo.5884485

使用方式：
    python scripts/download_skyfinder_camera.py --camera 10870
    python scripts/download_skyfinder_camera.py --camera 9417
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


def main():
    parser = argparse.ArgumentParser(description='Download SkyFinder camera data')
    parser.add_argument('--camera', type=str, required=True, help='Camera ID to download (e.g., 10870, 9417)')
    parser.add_argument('--max_images', type=int, default=100, help='Maximum number of images to download')
    args = parser.parse_args()
    
    camera_id = args.camera
    max_images = args.max_images
    
    # 設定目錄
    download_dir = f"data/skyfinder_{camera_id}/downloads"
    output_dir = f"data/skyfinder_{camera_id}"
    
    print("=" * 60)
    print(f"  SkyFinder Camera {camera_id} Downloader")
    print("=" * 60)
    print()
    
    # 建立目錄
    os.makedirs(download_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "masks"), exist_ok=True)
    
    # 1. 下載 masks（如果還沒下載過）
    # masks 是共用的，可能已經在 skyfinder_sample 下載過
    masks_zip = os.path.join(download_dir, "skyfinder_masks.zip")
    masks_extract_dir = os.path.join(download_dir, "masks_extracted")
    
    # 檢查是否有共用的 masks
    shared_masks_dir = "data/skyfinder_sample/downloads/masks_extracted"
    if os.path.exists(shared_masks_dir):
        print("[1] Using shared masks from skyfinder_sample...")
        masks_extract_dir = shared_masks_dir
    else:
        print("[1] Downloading masks...")
        masks_url = f"{ZENODO_BASE}/skyfinder_masks.zip?download=1"
        
        if not os.path.exists(masks_zip):
            if not download_file(masks_url, masks_zip):
                print("  Failed to download masks. Exiting.")
                return
        else:
            print("  Masks already downloaded.")
        
        # 解壓縮 masks
        print()
        print("[2] Extracting masks...")
        if not os.path.exists(masks_extract_dir):
            with zipfile.ZipFile(masks_zip, 'r') as zip_ref:
                zip_ref.extractall(masks_extract_dir)
            print("  Masks extracted.")
        else:
            print("  Masks already extracted.")
    
    # 2. 下載 camera 圖片
    print()
    print(f"[3] Downloading camera {camera_id} images...")
    camera_url = f"{ZENODO_BASE}/{camera_id}.zip?download=1"
    camera_zip = os.path.join(download_dir, f"{camera_id}.zip")
    
    if not os.path.exists(camera_zip):
        if not download_file(camera_url, camera_zip):
            print("  Failed to download camera images. Exiting.")
            return
    else:
        print("  Camera images already downloaded.")
    
    # 3. 解壓縮 camera 圖片
    print()
    print(f"[4] Extracting camera {camera_id} images...")
    camera_extract_dir = os.path.join(download_dir, f"{camera_id}_extracted")
    if not os.path.exists(camera_extract_dir):
        with zipfile.ZipFile(camera_zip, 'r') as zip_ref:
            zip_ref.extractall(camera_extract_dir)
        print("  Camera images extracted.")
    else:
        print("  Camera images already extracted.")
    
    # 4. 整理檔案結構
    print()
    print("[5] Organizing files...")
    
    # 複製圖片到 images 目錄
    image_count = 0
    
    for root, dirs, files in os.walk(camera_extract_dir):
        for f in sorted(files):
            if f.endswith(('.jpg', '.png', '.jpeg')):
                src = os.path.join(root, f)
                dst = os.path.join(output_dir, "images", f"{image_count+1:03d}.jpg")
                shutil.copy2(src, dst)
                image_count += 1
                
                if image_count >= max_images:
                    break
        if image_count >= max_images:
            break
    
    print(f"  Copied {image_count} images to {output_dir}/images/")
    
    # 5. 處理 mask
    print()
    print("[6] Processing masks...")
    
    # 找到 mask 檔案
    mask_src = None
    for root, dirs, files in os.walk(masks_extract_dir):
        for f in files:
            # SkyFinder masks 通常是 [camera_id].png 或類似格式
            if f.startswith(camera_id) or camera_id in f:
                if f.endswith(('.png', '.pgm', '.jpg')):
                    mask_src = os.path.join(root, f)
                    print(f"  Found mask file: {mask_src}")
                    break
        if mask_src:
            break
    
    if mask_src:
        # 為每張圖片複製相同的 mask
        mask_img = Image.open(mask_src)
        
        print(f"  Mask size: {mask_img.size}")
        print(f"  Mask mode: {mask_img.mode}")
        
        for i in range(image_count):
            dst = os.path.join(output_dir, "masks", f"{i+1:03d}.png")
            mask_img.save(dst)
        
        print(f"  Copied mask to {image_count} files in {output_dir}/masks/")
    else:
        print(f"  Warning: Could not find mask file for camera {camera_id}!")
        print("  Available mask files:")
        for root, dirs, files in os.walk(masks_extract_dir):
            for f in files[:10]:
                print(f"    - {f}")
    
    # 6. 完成
    print()
    print("=" * 60)
    print("  Download Complete!")
    print("=" * 60)
    print()
    print(f"  Camera ID: {camera_id}")
    print(f"  Images: {output_dir}/images/ ({image_count} files)")
    print(f"  Masks:  {output_dir}/masks/ ({image_count} files)")
    print()
    print("  Note: SkyFinder uses ONE mask per camera (static scene).")
    print("        All images from the same camera share the same mask.")
    print()


if __name__ == "__main__":
    main()
