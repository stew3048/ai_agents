"""
測試資料集載入器
用於驗證 dataset loader 是否正常工作
"""

import torch
from utils import get_dataloader
import matplotlib.pyplot as plt
import os


def test_dataset():
    """測試資料集載入功能"""
    
    # 設定資料集路徑
    # 請根據您的資料集結構調整這些路徑
    # 範例 1: 簡單格式
    # images_dir = "data/images"
    # masks_dir = "data/masks"
    
    # 範例 2: 分組格式（train）
    images_dir = "data/train/images"
    masks_dir = "data/train/masks"
    
    # 如果 train 資料夾不存在，嘗試簡單格式
    if not os.path.exists(images_dir):
        images_dir = "data/images"
        masks_dir = "data/masks"
    
    print("=" * 50)
    print("測試資料集載入器")
    print("=" * 50)
    print(f"圖片路徑: {images_dir}")
    print(f"Mask 路徑: {masks_dir}")
    print()
    
    try:
        # 創建資料載入器（測試模式，不進行資料增強）
        dataloader = get_dataloader(
            images_dir=images_dir,
            masks_dir=masks_dir,
            batch_size=4,
            shuffle=False,
            transform=False,  # 測試時關閉資料增強
            image_size=(256, 256),
            num_workers=0
        )
        
        print(f"✓ 成功創建 DataLoader")
        print(f"✓ 資料集大小: {len(dataloader.dataset)}")
        print()
        
        # 測試讀取一個批次
        print("測試讀取資料...")
        for batch_idx, (images, masks) in enumerate(dataloader):
            print(f"✓ 批次 {batch_idx + 1}:")
            print(f"  - 圖片形狀: {images.shape}")  # [B, C, H, W]
            print(f"  - Mask 形狀: {masks.shape}")   # [B, 1, H, W]
            print(f"  - 圖片值範圍: [{images.min():.3f}, {images.max():.3f}]")
            print(f"  - Mask 值範圍: [{masks.min():.3f}, {masks.max():.3f}]")
            
            # 只測試第一個批次
            if batch_idx == 0:
                break
        
        print()
        print("=" * 50)
        print("✓ 資料集載入器測試通過！")
        print("=" * 50)
        
        # 測試資料增強模式
        print()
        print("測試資料增強模式...")
        aug_dataloader = get_dataloader(
            images_dir=images_dir,
            masks_dir=masks_dir,
            batch_size=1,
            shuffle=False,
            transform=True,  # 開啟資料增強
            image_size=(256, 256),
            num_workers=0
        )
        
        images, masks = next(iter(aug_dataloader))
        print(f"✓ 資料增強測試通過")
        print(f"  - 增強後圖片形狀: {images.shape}")
        print(f"  - 增強後 Mask 形狀: {masks.shape}")
        
    except ValueError as e:
        print(f"✗ 錯誤: {e}")
        print()
        print("請確認:")
        print("1. 資料集已下載並放置在正確路徑")
        print("2. 圖片和 mask 檔案名稱對應（例如: img1.jpg 對應 img1.png）")
        print("3. 資料夾結構正確")
        print()
        print("支援的資料集結構:")
        print("  方式 1: data/images/ 和 data/masks/")
        print("  方式 2: data/train/images/ 和 data/train/masks/")


if __name__ == '__main__':
    test_dataset()
