"""詳細檢查所有 camera 下載是否完成"""
import os
from datetime import datetime

cameras = ["9112", "9291", "9483", "9708", "10917"]

print("=" * 60)
print("  詳細下載狀態檢查")
print("=" * 60)
print(f"檢查時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

completed = []
incomplete = []
not_started = []

for cam in cameras:
    dir_path = f"data/skyfinder_{cam}"
    zip_path = f"{dir_path}/downloads/{cam}.zip"
    images_dir = f"{dir_path}/images"
    masks_dir = f"{dir_path}/masks"
    
    print(f"Camera {cam}:")
    
    # 檢查圖片
    if os.path.exists(images_dir):
        image_files = [f for f in os.listdir(images_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        image_count = len(image_files)
        print(f"  Images: {image_count} files")
    else:
        image_count = 0
        print(f"  Images: 資料夾不存在")
    
    # 檢查 masks
    if os.path.exists(masks_dir):
        mask_files = [f for f in os.listdir(masks_dir) 
                     if f.lower().endswith(('.png', '.pgm', '.jpg', '.jpeg'))]
        mask_count = len(mask_files)
        print(f"  Masks: {mask_count} files")
    else:
        mask_count = 0
        print(f"  Masks: 資料夾不存在")
    
    # 檢查 ZIP 檔案
    if os.path.exists(zip_path):
        size_mb = os.path.getsize(zip_path) / (1024 * 1024)
        file_time = datetime.fromtimestamp(os.path.getmtime(zip_path))
        time_since = (datetime.now() - file_time).total_seconds() / 60
        print(f"  ZIP: {size_mb:.1f} MB (最後更新: {time_since:.1f} 分鐘前)")
    else:
        print(f"  ZIP: 不存在")
    
    # 判斷狀態
    if image_count > 0 and mask_count > 0:
        if image_count == mask_count:
            print(f"  Status: [完成] ✓")
            completed.append(cam)
        else:
            print(f"  Status: [不完整] Images 和 Masks 數量不一致")
            incomplete.append(cam)
    elif image_count > 0:
        print(f"  Status: [部分完成] 有 Images 但缺少 Masks")
        incomplete.append(cam)
    elif os.path.exists(zip_path):
        size_mb = os.path.getsize(zip_path) / (1024 * 1024)
        percent = (size_mb / 51.7 * 100) if 51.7 > 0 else 0
        if percent >= 95:
            print(f"  Status: [下載中] ZIP 檔案幾乎完成 ({percent:.1f}%)，可能正在解壓縮")
        else:
            print(f"  Status: [下載中] {size_mb:.1f} MB ({percent:.1f}%)")
        incomplete.append(cam)
    else:
        print(f"  Status: [未開始]")
        not_started.append(cam)
    
    print()

# 總結
print("=" * 60)
print("  總結")
print("=" * 60)
print(f"完成: {len(completed)} 個")
for cam in completed:
    images_dir = f"data/skyfinder_{cam}/images"
    image_count = len([f for f in os.listdir(images_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    print(f"  ✓ Camera {cam} ({image_count} images)")

if incomplete:
    print(f"\n進行中/不完整: {len(incomplete)} 個")
    for cam in incomplete:
        print(f"  - Camera {cam}")

if not_started:
    print(f"\n未開始: {len(not_started)} 個")
    for cam in not_started:
        print(f"  - Camera {cam}")

print()
print(f"總進度: {len(completed)}/{len(cameras)} 完成 ({len(completed)/len(cameras)*100:.1f}%)")
print()
