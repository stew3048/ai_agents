"""
清理所有受損的圖片
檢查並刪除無法正常讀取的圖片檔案

一般處理受損圖片的方法：
1. 使用 PIL Image.verify() 檢查檔案結構完整性
2. 使用 PIL Image.load() 測試是否能正常載入數據
3. 檢查圖片尺寸是否合理（寬高 > 0）
4. 檢查圖片模式是否有效
5. 刪除無法修復的圖片
"""

import os
from PIL import Image
from typing import List, Tuple


def is_image_corrupted(img_path: str) -> Tuple[bool, str]:
    """
    檢查圖片是否受損
    
    Returns:
        (is_corrupted, error_message)
    """
    try:
        # 步驟 1: 嘗試打開圖片
        img = Image.open(img_path)
        
        # 步驟 2: 驗證圖片結構完整性
        img.verify()
        img.close()
        
        # 步驟 3: 重新打開並載入數據（verify 後需要重新打開）
        img = Image.open(img_path)
        
        # 步驟 4: 載入圖片數據
        img.load()
        
        # 步驟 5: 檢查尺寸是否合理
        if img.size[0] <= 0 or img.size[1] <= 0:
            img.close()
            return True, f"Invalid size: {img.size}"
        
        # 步驟 6: 嘗試轉換為 RGB（測試是否能正常處理）
        if img.mode not in ['RGB', 'RGBA', 'L', 'P']:
            # 嘗試轉換，如果失敗則視為受損
            try:
                img.convert('RGB')
            except:
                img.close()
                return True, f"Invalid mode: {img.mode}"
        
        img.close()
        return False, ""
        
    except Exception as e:
        return True, str(e)


def clean_camera_images(camera_folder: str) -> Tuple[int, List[str]]:
    """
    清理單個相機資料夾中的受損圖片
    
    Returns:
        (deleted_count, deleted_files)
    """
    images_dir = os.path.join(camera_folder, "images")
    
    if not os.path.exists(images_dir):
        return 0, []
    
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    deleted_count = 0
    deleted_files = []
    
    print(f"\n  檢查 {images_dir}...")
    
    for filename in os.listdir(images_dir):
        if any(filename.lower().endswith(ext) for ext in valid_extensions):
            img_path = os.path.join(images_dir, filename)
            
            is_corrupted, error_msg = is_image_corrupted(img_path)
            
            if is_corrupted:
                try:
                    os.remove(img_path)
                    deleted_count += 1
                    deleted_files.append(filename)
                    print(f"    [刪除] {filename} - {error_msg}")
                except Exception as del_err:
                    print(f"    [刪除失敗] {filename} - {del_err}")
    
    return deleted_count, deleted_files


def main():
    print("=" * 60)
    print("  清理受損圖片")
    print("=" * 60)
    print()
    
    base_dir = "data"
    if not os.path.exists(base_dir):
        print(f"  錯誤：找不到 {base_dir} 目錄")
        return
    
    # 找出所有相機資料夾
    camera_folders = []
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if (os.path.isdir(item_path) and 
            item.startswith("skyfinder_") and 
            item not in ["skyfinder_masks", "skyfinder_sample"]):
            camera_folders.append(item_path)
    
    camera_folders = sorted(camera_folders)
    
    print(f"找到 {len(camera_folders)} 個相機資料夾")
    print()
    
    total_deleted = 0
    total_files = []
    
    for i, folder in enumerate(camera_folders, 1):
        camera_id = os.path.basename(folder).replace("skyfinder_", "")
        print(f"[{i}/{len(camera_folders)}] Camera {camera_id}...", end=" ", flush=True)
        
        deleted_count, deleted_files = clean_camera_images(folder)
        total_deleted += deleted_count
        total_files.extend(deleted_files)
        
        if deleted_count > 0:
            print(f"已刪除 {deleted_count} 張受損圖片")
        else:
            print("無受損圖片")
    
    print()
    print("=" * 60)
    print("  清理完成")
    print("=" * 60)
    print(f"  總共刪除 {total_deleted} 張受損圖片")
    
    if total_deleted > 0:
        print(f"\n  刪除的檔案列表（前 20 個）：")
        for filename in total_files[:20]:
            print(f"    - {filename}")
        if len(total_files) > 20:
            print(f"    ... 還有 {len(total_files) - 20} 個檔案")
    print()


if __name__ == "__main__":
    main()
