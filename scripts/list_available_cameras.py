"""
列出所有可用的 SkyFinder camera ID
"""

import os

def list_available_cameras():
    """從 masks 目錄列出所有可用的 camera ID"""
    masks_dir = "data/skyfinder_sample/downloads/masks_extracted"
    
    if not os.path.exists(masks_dir):
        print("Masks directory not found. Please download masks first.")
        return []
    
    camera_ids = []
    for root, dirs, files in os.walk(masks_dir):
        for f in files:
            if f.endswith(('.png', '.pgm', '.jpg')):
                # 提取 camera ID（檔案名稱，去掉副檔名）
                camera_id = os.path.splitext(f)[0]
                try:
                    # 確認是數字
                    int(camera_id)
                    camera_ids.append(camera_id)
                except ValueError:
                    pass
    
    # 排序（數字排序）
    camera_ids.sort(key=lambda x: int(x))
    return camera_ids


if __name__ == "__main__":
    print("=" * 60)
    print("  Available SkyFinder Camera IDs")
    print("=" * 60)
    print()
    
    camera_ids = list_available_cameras()
    
    if camera_ids:
        print(f"  Total: {len(camera_ids)} cameras")
        print()
        print("  Camera IDs:")
        for i, cam_id in enumerate(camera_ids, 1):
            print(f"    {cam_id:>6}", end="  ")
            if i % 10 == 0:
                print()
        if len(camera_ids) % 10 != 0:
            print()
    else:
        print("  No cameras found.")
    
    print()
