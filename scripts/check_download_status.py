"""快速檢查下載狀態"""
import os
from datetime import datetime

cameras = ["9112", "9291", "9483", "9708", "10917"]

print("=" * 60)
print("  下載狀態檢查")
print("=" * 60)
print(f"檢查時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

for cam in cameras:
    dir_path = f"data/skyfinder_{cam}"
    zip_path = f"{dir_path}/downloads/{cam}.zip"
    images_dir = f"{dir_path}/images"
    
    if os.path.exists(images_dir):
        image_files = [f for f in os.listdir(images_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if len(image_files) > 0:
            print(f"Camera {cam}: [完成] {len(image_files)} images")
            continue
    
    if os.path.exists(zip_path):
        size_mb = os.path.getsize(zip_path) / (1024 * 1024)
        percent = (size_mb / 51.7 * 100) if 51.7 > 0 else 0
        file_time = datetime.fromtimestamp(os.path.getmtime(zip_path))
        time_since = (datetime.now() - file_time).total_seconds() / 60
        print(f"Camera {cam}: [下載中] {size_mb:.1f} MB ({percent:.1f}%) - 最後更新: {time_since:.1f} 分鐘前")
    else:
        print(f"Camera {cam}: [未開始]")

print()
