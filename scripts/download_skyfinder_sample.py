"""
下載 SkyFinder 資料集的一小部分用於測試
從 Zenodo 下載一個 camera 的圖片和對應的 masks

Zenodo DOI: 10.5281/zenodo.5884485
"""

import os
import urllib.request
import zipfile
import shutil

# Zenodo base URL
ZENODO_BASE = "https://zenodo.org/records/5884485/files"

# 下載目標
DOWNLOAD_DIR = "data/skyfinder_sample/downloads"
OUTPUT_DIR = "data/skyfinder_sample"

# 選擇一個較小的 camera 來測試（10066 是其中一個）
CAMERA_ID = "10066"


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
    print("=" * 60)
    print("  SkyFinder Sample Dataset Downloader")
    print("=" * 60)
    print()
    
    # 建立目錄
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "images"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "masks"), exist_ok=True)
    
    # 1. 下載 masks
    print("[1] Downloading masks...")
    masks_url = f"{ZENODO_BASE}/skyfinder_masks.zip?download=1"
    masks_zip = os.path.join(DOWNLOAD_DIR, "skyfinder_masks.zip")
    
    if not os.path.exists(masks_zip):
        if not download_file(masks_url, masks_zip):
            print("  Failed to download masks. Exiting.")
            return
    else:
        print("  Masks already downloaded.")
    
    # 2. 下載一個 camera 的圖片
    print()
    print(f"[2] Downloading camera {CAMERA_ID} images...")
    camera_url = f"{ZENODO_BASE}/{CAMERA_ID}.zip?download=1"
    camera_zip = os.path.join(DOWNLOAD_DIR, f"{CAMERA_ID}.zip")
    
    if not os.path.exists(camera_zip):
        if not download_file(camera_url, camera_zip):
            print("  Failed to download camera images. Exiting.")
            return
    else:
        print("  Camera images already downloaded.")
    
    # 3. 解壓縮 masks
    print()
    print("[3] Extracting masks...")
    masks_extract_dir = os.path.join(DOWNLOAD_DIR, "masks_extracted")
    if not os.path.exists(masks_extract_dir):
        with zipfile.ZipFile(masks_zip, 'r') as zip_ref:
            zip_ref.extractall(masks_extract_dir)
        print("  Masks extracted.")
    else:
        print("  Masks already extracted.")
    
    # 4. 解壓縮 camera 圖片
    print()
    print(f"[4] Extracting camera {CAMERA_ID} images...")
    camera_extract_dir = os.path.join(DOWNLOAD_DIR, f"{CAMERA_ID}_extracted")
    if not os.path.exists(camera_extract_dir):
        with zipfile.ZipFile(camera_zip, 'r') as zip_ref:
            zip_ref.extractall(camera_extract_dir)
        print("  Camera images extracted.")
    else:
        print("  Camera images already extracted.")
    
    # 5. 整理檔案結構
    print()
    print("[5] Organizing files...")
    
    # 找到這個 camera 的 mask
    # SkyFinder 的 mask 是以 camera ID 命名的
    mask_found = False
    for root, dirs, files in os.walk(masks_extract_dir):
        for f in files:
            if CAMERA_ID in f and f.endswith(('.png', '.jpg', '.pgm')):
                src = os.path.join(root, f)
                # 複製到 masks 目錄
                print(f"  Found mask: {f}")
                mask_found = True
                # 我們需要為每張圖片複製相同的 mask
                break
    
    # 複製圖片到 images 目錄（只取前 50 張）
    image_count = 0
    max_images = 50
    
    for root, dirs, files in os.walk(camera_extract_dir):
        for f in sorted(files):
            if f.endswith(('.jpg', '.png', '.jpeg')):
                src = os.path.join(root, f)
                dst = os.path.join(OUTPUT_DIR, "images", f"{image_count+1:03d}.jpg")
                shutil.copy2(src, dst)
                image_count += 1
                
                if image_count >= max_images:
                    break
        if image_count >= max_images:
            break
    
    print(f"  Copied {image_count} images to {OUTPUT_DIR}/images/")
    
    # 6. 處理 mask（SkyFinder 每個 camera 只有一個 mask）
    # 需要把這個 mask 複製給每張圖片
    print()
    print("[6] Processing masks...")
    
    # 找到 mask 檔案
    mask_src = None
    for root, dirs, files in os.walk(masks_extract_dir):
        for f in files:
            # SkyFinder masks 通常是 [camera_id].png 或類似格式
            if f.startswith(CAMERA_ID) or CAMERA_ID in f:
                if f.endswith(('.png', '.pgm', '.jpg')):
                    mask_src = os.path.join(root, f)
                    print(f"  Found mask file: {mask_src}")
                    break
        if mask_src:
            break
    
    if mask_src:
        # 為每張圖片複製相同的 mask
        from PIL import Image
        mask_img = Image.open(mask_src)
        
        # 確保 mask 是二值的
        mask_array = list(mask_img.getdata())
        print(f"  Mask size: {mask_img.size}")
        print(f"  Mask mode: {mask_img.mode}")
        
        for i in range(image_count):
            dst = os.path.join(OUTPUT_DIR, "masks", f"{i+1:03d}.png")
            mask_img.save(dst)
        
        print(f"  Copied mask to {image_count} files in {OUTPUT_DIR}/masks/")
    else:
        print("  Warning: Could not find mask file!")
    
    # 7. 完成
    print()
    print("=" * 60)
    print("  Download Complete!")
    print("=" * 60)
    print()
    print(f"  Images: {OUTPUT_DIR}/images/ ({image_count} files)")
    print(f"  Masks:  {OUTPUT_DIR}/masks/ ({image_count} files)")
    print()
    print("  Note: SkyFinder uses ONE mask per camera (static scene).")
    print("        All images from the same camera share the same mask.")
    print()


if __name__ == "__main__":
    main()
