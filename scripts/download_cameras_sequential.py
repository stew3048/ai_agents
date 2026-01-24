"""
依序下載多個 SkyFinder camera 資料
一次只下載一個，完成後再開始下一個
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def check_download_complete(camera_id):
    """檢查下載是否完成"""
    output_dir = f"data/skyfinder_{camera_id}"
    images_dir = os.path.join(output_dir, "images")
    
    # 檢查是否有圖片檔案
    if os.path.exists(images_dir):
        image_files = [f for f in os.listdir(images_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if len(image_files) > 0:
            return True
    return False

def download_camera(camera_id):
    """下載單個 camera"""
    print("=" * 60)
    print(f"開始下載 Camera {camera_id}")
    print("=" * 60)
    
    cmd = [sys.executable, "scripts/download_skyfinder_camera.py", "--camera", camera_id]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        if result.returncode == 0:
            print(f"\n[OK] Camera {camera_id} 下載完成！\n")
            return True
        else:
            print(f"\n[ERROR] Camera {camera_id} 下載失敗（返回碼: {result.returncode}）\n")
            return False
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Camera {camera_id} 下載失敗: {e}\n")
        return False
    except KeyboardInterrupt:
        print(f"\n\n下載被使用者中斷（Camera {camera_id}）")
        return False

def main():
    cameras = ["9112", "9291", "9483", "9708", "10917"]
    
    print("=" * 60)
    print("  SkyFinder Camera 批次下載")
    print("=" * 60)
    print(f"\n將依序下載以下 {len(cameras)} 個 camera:")
    for i, cam in enumerate(cameras, 1):
        status = "[OK] 已完成" if check_download_complete(cam) else "待下載"
        print(f"  {i}. Camera {cam} - {status}")
    print()
    
    completed = []
    failed = []
    
    for i, camera_id in enumerate(cameras, 1):
        # 檢查是否已經下載完成
        if check_download_complete(camera_id):
            print(f"[{i}/{len(cameras)}] Camera {camera_id} 已經存在，跳過")
            completed.append(camera_id)
            continue
        
        print(f"\n[{i}/{len(cameras)}] 開始下載 Camera {camera_id}...")
        success = download_camera(camera_id)
        
        if success:
            completed.append(camera_id)
        else:
            failed.append(camera_id)
            # 詢問是否繼續
            response = input(f"\nCamera {camera_id} 下載失敗，是否繼續下載下一個？ (y/n): ")
            if response.lower() != 'y':
                print("\n下載已停止")
                break
    
    # 總結
    print("\n" + "=" * 60)
    print("  下載總結")
    print("=" * 60)
    print(f"  成功: {len(completed)} 個")
    for cam in completed:
        print(f"    [OK] Camera {cam}")
    
    if failed:
        print(f"\n  失敗: {len(failed)} 個")
        for cam in failed:
            print(f"    [ERROR] Camera {cam}")
    print()

if __name__ == "__main__":
    main()
