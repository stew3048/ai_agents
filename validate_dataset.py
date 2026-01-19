"""
資料集驗證腳本
確認 image 與 mask 的數量、檔名、尺寸、像素值是否正確
"""

import os
import random
import numpy as np
from PIL import Image


def get_image_files(directory):
    """取得資料夾內所有圖片檔案"""
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    files = []
    
    if not os.path.exists(directory):
        raise FileNotFoundError(f"資料夾不存在: {directory}")
    
    for f in os.listdir(directory):
        if os.path.splitext(f.lower())[1] in valid_extensions:
            files.append(f)
    
    return sorted(files)


def get_base_name(filename):
    """取得不含副檔名的檔名"""
    return os.path.splitext(filename)[0]


def validate_dataset(images_dir, masks_dir, num_samples=5):
    """
    驗證資料集
    
    參數:
        images_dir: 圖片資料夾路徑
        masks_dir: mask 資料夾路徑
        num_samples: 隨機抽樣驗證的數量
    """
    print("=" * 60)
    print("資料集驗證")
    print("=" * 60)
    print(f"圖片資料夾: {images_dir}")
    print(f"Mask 資料夾: {masks_dir}")
    print()
    
    # ========== 步驟 1: 檢查檔案數量與配對 ==========
    print("[Step 1] 檢查檔案數量與配對...")
    
    image_files = get_image_files(images_dir)
    mask_files = get_image_files(masks_dir)
    
    print(f"  圖片數量: {len(image_files)}")
    print(f"  Mask 數量: {len(mask_files)}")
    
    # 檢查數量是否一致
    if len(image_files) != len(mask_files):
        raise ValueError(
            f"[ERROR] 圖片與 Mask 數量不一致！\n"
            f"  圖片: {len(image_files)} 個\n"
            f"  Mask: {len(mask_files)} 個"
        )
    
    if len(image_files) == 0:
        raise ValueError(f"[ERROR] 找不到任何圖片檔案: {images_dir}")
    
    # 建立檔名配對（使用 base name）
    image_base_names = {get_base_name(f): f for f in image_files}
    mask_base_names = {get_base_name(f): f for f in mask_files}
    
    # 檢查配對
    paired_files = []
    missing_masks = []
    missing_images = []
    
    for base_name, image_file in image_base_names.items():
        if base_name in mask_base_names:
            paired_files.append({
                'base_name': base_name,
                'image': image_file,
                'mask': mask_base_names[base_name]
            })
        else:
            missing_masks.append(image_file)
    
    for base_name in mask_base_names:
        if base_name not in image_base_names:
            missing_images.append(mask_base_names[base_name])
    
    if missing_masks:
        raise ValueError(
            f"[ERROR] 以下圖片找不到對應的 Mask:\n"
            f"  {missing_masks}"
        )
    
    if missing_images:
        raise ValueError(
            f"[ERROR] 以下 Mask 找不到對應的圖片:\n"
            f"  {missing_images}"
        )
    
    print(f"  成功配對: {len(paired_files)} 組")
    print("  [OK] 檔案數量與配對檢查通過")
    print()
    
    # ========== 步驟 2: 隨機抽樣驗證 ==========
    print(f"[Step 2] 隨機抽樣 {num_samples} 組進行詳細驗證...")
    print()
    
    # 隨機選擇樣本
    if len(paired_files) < num_samples:
        samples = paired_files
    else:
        samples = random.sample(paired_files, num_samples)
    
    for i, pair in enumerate(samples):
        print(f"  [{i+1}/{len(samples)}] 檔案: {pair['base_name']}")
        
        image_path = os.path.join(images_dir, pair['image'])
        mask_path = os.path.join(masks_dir, pair['mask'])
        
        # 讀取圖片
        try:
            image = Image.open(image_path)
            image_array = np.array(image)
        except Exception as e:
            raise ValueError(f"[ERROR] 無法讀取圖片: {image_path}\n  錯誤: {e}")
        
        # 讀取 mask
        try:
            mask = Image.open(mask_path)
            mask_array = np.array(mask)
        except Exception as e:
            raise ValueError(f"[ERROR] 無法讀取 Mask: {mask_path}\n  錯誤: {e}")
        
        # 取得圖片資訊
        if len(image_array.shape) == 3:
            img_h, img_w, img_c = image_array.shape
        elif len(image_array.shape) == 2:
            img_h, img_w = image_array.shape
            img_c = 1
        else:
            raise ValueError(
                f"[ERROR] 圖片格式不正確: {image_path}\n"
                f"  Shape: {image_array.shape}"
            )
        
        # 取得 mask 資訊
        if len(mask_array.shape) == 2:
            mask_h, mask_w = mask_array.shape
        elif len(mask_array.shape) == 3:
            # 如果 mask 是 RGB，取第一個通道或轉灰階
            mask_h, mask_w = mask_array.shape[:2]
            print(f"       [WARN] Mask 是多通道圖片 (shape: {mask_array.shape})，建議使用灰階")
            mask_array = mask_array[:, :, 0]  # 取第一個通道來檢查
        else:
            raise ValueError(
                f"[ERROR] Mask 格式不正確: {mask_path}\n"
                f"  Shape: {mask_array.shape}"
            )
        
        mask_unique = np.unique(mask_array)
        
        # 印出資訊
        print(f"       Image: {img_h} x {img_w} x {img_c} (H x W x C)")
        print(f"       Mask:  {mask_h} x {mask_w}")
        print(f"       Mask unique values: {set(mask_unique)}")
        
        # 驗證 1: 圖片通道數
        if img_c not in [1, 3, 4]:
            raise ValueError(
                f"[ERROR] 圖片通道數異常: {image_path}\n"
                f"  預期: 1, 3, 或 4\n"
                f"  實際: {img_c}"
            )
        
        # 驗證 2: 尺寸是否一致
        if img_h != mask_h or img_w != mask_w:
            raise ValueError(
                f"[ERROR] 圖片與 Mask 尺寸不一致: {pair['base_name']}\n"
                f"  圖片: {img_h} x {img_w}\n"
                f"  Mask: {mask_h} x {mask_w}"
            )
        
        # 驗證 3: Mask 值是否為二進制
        valid_mask_values = {0, 255}
        if not set(mask_unique).issubset(valid_mask_values):
            raise ValueError(
                f"[ERROR] Mask 值不是二進制 (0/255): {mask_path}\n"
                f"  預期: {{0, 255}}\n"
                f"  實際: {set(mask_unique)}"
            )
        
        print(f"       [OK] 驗證通過")
        print()
    
    # ========== 完成 ==========
    print("=" * 60)
    print("[OK] 所有驗證通過！")
    print("=" * 60)
    print()
    print("摘要:")
    print(f"  總配對數: {len(paired_files)}")
    print(f"  抽樣驗證: {len(samples)}")
    print(f"  圖片格式: RGB ({img_c} channels)")
    print(f"  Mask 格式: 灰階 (0=非天空, 255=天空)")
    
    return paired_files


def main():
    """主函數"""
    print()
    
    # 驗證訓練集
    print(">>> 驗證訓練集")
    print()
    train_pairs = validate_dataset(
        images_dir='data/train/images',
        masks_dir='data/train/masks',
        num_samples=5
    )
    
    print()
    print()
    
    # 驗證驗證集
    print(">>> 驗證驗證集")
    print()
    val_pairs = validate_dataset(
        images_dir='data/val/images',
        masks_dir='data/val/masks',
        num_samples=2  # 驗證集只有 2 張，全部驗證
    )
    
    print()
    print("=" * 60)
    print("[OK] 訓練集和驗證集全部驗證通過！")
    print("=" * 60)


if __name__ == '__main__':
    main()
