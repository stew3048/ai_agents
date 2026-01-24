"""快速檢查下載狀態"""
import os

cameras = ["9112", "9291", "9483", "9708", "10870", "10917"]

print("=" * 60)
print("  快速狀態檢查")
print("=" * 60)

completed = []
downloading = []

for cam in cameras:
    dir_path = f"data/skyfinder_{cam}"
    images_dir = f"{dir_path}/images"
    
    if os.path.exists(images_dir):
        image_files = [f for f in os.listdir(images_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if len(image_files) > 0:
            print(f"Camera {cam}: [完成] {len(image_files)} images")
            completed.append(cam)
        else:
            print(f"Camera {cam}: [下載中]")
            downloading.append(cam)
    else:
        print(f"Camera {cam}: [下載中]")
        downloading.append(cam)

print()
print(f"完成: {len(completed)}/{len(cameras)}")
if downloading:
    print(f"進行中: {len(downloading)} 個")
print()
