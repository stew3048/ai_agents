"""
監控 SkyFinder camera 下載進度
自動檢查下載狀態，完成後開始下一個
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
            return True, len(image_files)
    return False, 0

def check_zip_progress(camera_id):
    """檢查 ZIP 下載進度"""
    zip_path = f"data/skyfinder_{camera_id}/downloads/{camera_id}.zip"
    if os.path.exists(zip_path):
        file_size = os.path.getsize(zip_path)
        size_mb = file_size / (1024 * 1024)
        # 假設每個 camera 約 51.7 MB（實際可能不同）
        estimated_total = 51.7
        percent = (size_mb / estimated_total * 100) if estimated_total > 0 else 0
        return True, size_mb, percent
    return False, 0, 0

def download_camera(camera_id):
    """下載單個 camera"""
    print("=" * 60)
    print(f"開始下載 Camera {camera_id}")
    print("=" * 60)
    
    cmd = [sys.executable, "scripts/download_skyfinder_camera.py", "--camera", camera_id]
    
    try:
        result = subprocess.run(cmd, check=True)
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
    print("  SkyFinder Camera 下載監控")
    print("=" * 60)
    print(f"  開始時間: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  將下載 {len(cameras)} 個 camera: {', '.join(cameras)}")
    print("  此腳本會自動監控並依序下載所有 camera")
    print("  可以安全關閉終端，下載會繼續進行")
    print("=" * 60)
    print()
    
    completed = []
    failed = []
    current_downloading = None
    
    for i, camera_id in enumerate(cameras, 1):
        # 檢查是否已經完成
        is_complete, image_count = check_download_complete(camera_id)
        if is_complete:
            print(f"[{i}/{len(cameras)}] Camera {camera_id}: [OK] 已完成 ({image_count} images)")
            completed.append(camera_id)
            continue
        
        # 檢查是否正在下載
        has_zip, zip_size, zip_percent = check_zip_progress(camera_id)
        if has_zip and zip_percent < 100:
            print(f"[{i}/{len(cameras)}] Camera {camera_id}: 下載中 ({zip_size:.1f} MB, {zip_percent:.1f}%)")
            current_downloading = camera_id
            # 等待下載完成
            print(f"  等待 Camera {camera_id} 下載完成...")
            max_wait_time = 7200  # 最多等待 2 小時（考慮網路較慢的情況）
            wait_interval = 60  # 每 60 秒檢查一次（減少輸出頻率）
            elapsed = 0
            last_size = zip_size
            
            while elapsed < max_wait_time:
                time.sleep(wait_interval)
                elapsed += wait_interval
                
                is_complete, image_count = check_download_complete(camera_id)
                if is_complete:
                    print(f"  [OK] Camera {camera_id} 下載完成！({image_count} images)")
                    completed.append(camera_id)
                    break
                
                has_zip, zip_size, zip_percent = check_zip_progress(camera_id)
                if has_zip:
                    # 只有當進度有明顯變化時才輸出
                    if abs(zip_size - last_size) > 1.0 or zip_percent >= 100:
                        print(f"  進度: {zip_size:.1f} MB ({zip_percent:.1f}%) - 已等待 {elapsed//60} 分鐘")
                        last_size = zip_size
                else:
                    # ZIP 檔案不存在，可能下載失敗，重新開始
                    print(f"  ZIP 檔案遺失，重新下載 Camera {camera_id}...")
                    break
            else:
                print(f"  [WARNING] Camera {camera_id} 下載超時，嘗試重新下載...")
                # 刪除不完整的檔案並重新下載
                zip_path = f"data/skyfinder_{camera_id}/downloads/{camera_id}.zip"
                if os.path.exists(zip_path):
                    try:
                        os.remove(zip_path)
                        print(f"  已刪除不完整的 ZIP 檔案")
                    except:
                        pass
                # 不加入 failed，讓它重新嘗試下載
                continue
        
        # 開始下載
        if not is_complete:
            print(f"\n[{i}/{len(cameras)}] 開始下載 Camera {camera_id}...")
            success = download_camera(camera_id)
            
            if success:
                is_complete, image_count = check_download_complete(camera_id)
                if is_complete:
                    print(f"[OK] Camera {camera_id} 下載完成！({image_count} images)")
                    completed.append(camera_id)
                else:
                    print(f"[WARNING] Camera {camera_id} 下載完成但未找到圖片")
                    failed.append(camera_id)
            else:
                failed.append(camera_id)
                print(f"[ERROR] Camera {camera_id} 下載失敗，跳過")
    
    # 總結
    print("\n" + "=" * 60)
    print("  下載總結")
    print("=" * 60)
    print(f"  成功: {len(completed)} 個")
    for cam in completed:
        is_complete, image_count = check_download_complete(cam)
        print(f"    [OK] Camera {cam} ({image_count} images)")
    
    if failed:
        print(f"\n  失敗: {len(failed)} 個")
        for cam in failed:
            print(f"    [ERROR] Camera {cam}")
    print()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n下載監控被使用者中斷")
        sys.exit(1)
