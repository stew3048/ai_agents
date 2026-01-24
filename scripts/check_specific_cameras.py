"""檢查指定 camera 的下載狀態"""
import os
from datetime import datetime

cameras = ["9291", "9383", "9708", "100870", "10917"]

print("=" * 60)
print("  個別 Camera 下載狀態檢查")
print("=" * 60)
print(f"檢查時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

for cam in cameras:
    print(f"{'='*60}")
    print(f"Camera {cam}")
    print(f"{'='*60}")
    
    dir_path = f"data/skyfinder_{cam}"
    zip_path = f"{dir_path}/downloads/{cam}.zip"
    images_dir = f"{dir_path}/images"
    masks_dir = f"{dir_path}/masks"
    
    # 檢查資料夾是否存在
    if not os.path.exists(dir_path):
        print(f"  [錯誤] 資料夾不存在: {dir_path}")
        print()
        continue
    
    # 檢查 Images
    if os.path.exists(images_dir):
        image_files = [f for f in os.listdir(images_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"  Images: {len(image_files)} files")
        if len(image_files) > 0:
            print(f"    [OK] 資料夾存在且有檔案")
        else:
            print(f"    [WARNING] 資料夾存在但無檔案")
    else:
        print(f"  Images: 資料夾不存在")
    
    # 檢查 Masks
    if os.path.exists(masks_dir):
        mask_files = [f for f in os.listdir(masks_dir) 
                     if f.lower().endswith(('.png', '.pgm', '.jpg', '.jpeg'))]
        print(f"  Masks: {len(mask_files)} files")
        if len(mask_files) > 0:
            print(f"    [OK] 資料夾存在且有檔案")
        else:
            print(f"    [WARNING] 資料夾存在但無檔案")
    else:
        print(f"  Masks: 資料夾不存在")
    
    # 檢查 ZIP
    if os.path.exists(zip_path):
        size_mb = os.path.getsize(zip_path) / (1024 * 1024)
        file_time = datetime.fromtimestamp(os.path.getmtime(zip_path))
        time_since = (datetime.now() - file_time).total_seconds() / 60
        print(f"  ZIP: {size_mb:.1f} MB (最後更新: {time_since:.1f} 分鐘前)")
    else:
        print(f"  ZIP: 不存在")
    
    # 判斷狀態
    if os.path.exists(images_dir) and os.path.exists(masks_dir):
        img_count = len([f for f in os.listdir(images_dir) 
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        mask_count = len([f for f in os.listdir(masks_dir) 
                         if f.lower().endswith(('.png', '.pgm', '.jpg', '.jpeg'))])
        if img_count > 0 and mask_count > 0:
            if img_count == mask_count:
                print(f"  Status: [完成] OK ({img_count} images, {mask_count} masks)")
            else:
                print(f"  Status: [不完整] Images={img_count}, Masks={mask_count} (數量不一致)")
        elif img_count > 0:
            print(f"  Status: [部分完成] 有 {img_count} images 但缺少 masks")
        else:
            print(f"  Status: [未完成] 資料夾存在但無檔案")
    elif os.path.exists(zip_path):
        size_mb = os.path.getsize(zip_path) / (1024 * 1024)
        print(f"  Status: [下載中] ZIP 檔案存在 ({size_mb:.1f} MB)")
    else:
        print(f"  Status: [未開始] 無任何檔案")
    
    print()
