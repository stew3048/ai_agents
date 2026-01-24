"""確保所有 camera 下載完成"""
import os
import sys
import subprocess
import time

cameras = ["9291", "9708", "10917"]

def check_complete(camera_id):
    """檢查是否完成"""
    images_dir = f"data/skyfinder_{camera_id}/images"
    if os.path.exists(images_dir):
        image_files = [f for f in os.listdir(images_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        return len(image_files) > 0
    return False

def download_camera(camera_id):
    """下載 camera"""
    print(f"\n開始下載 Camera {camera_id}...")
    cmd = [sys.executable, "scripts/download_skyfinder_camera.py", "--camera", camera_id]
    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode == 0
    except:
        return False

print("=" * 60)
print("  確保下載完成")
print("=" * 60)

for cam in cameras:
    if check_complete(cam):
        print(f"Camera {cam}: 已完成")
    else:
        print(f"Camera {cam}: 未完成，開始下載...")
        success = download_camera(cam)
        if success and check_complete(cam):
            print(f"Camera {cam}: 下載完成！")
        else:
            print(f"Camera {cam}: 下載失敗或未完成")

print("\n最終狀態:")
for cam in cameras:
    if check_complete(cam):
        images_dir = f"data/skyfinder_{cam}/images"
        count = len([f for f in os.listdir(images_dir) 
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        print(f"  Camera {cam}: {count} images")
    else:
        print(f"  Camera {cam}: 未完成")
