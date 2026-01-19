"""
簡單測試腳本
使用虛擬資料快速測試整個流程是否正常
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
import tempfile
import shutil

# 修復 Windows 終端機 Unicode 編碼問題
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# 導入專案模組
from models import create_unet_model
from utils import get_dataloader, calculate_metrics


def create_dummy_dataset(num_images=10, image_size=(256, 256)):
    """
    創建虛擬測試資料集
    
    參數:
        num_images: 創建的圖片數量
        image_size: 圖片尺寸 (height, width)
    
    返回:
        temp_dir: 臨時資料夾路徑
        images_dir: 圖片資料夾路徑
        masks_dir: mask 資料夾路徑
    """
    # 創建臨時資料夾
    temp_dir = tempfile.mkdtemp(prefix='sky_seg_test_')
    images_dir = os.path.join(temp_dir, 'images')
    masks_dir = os.path.join(temp_dir, 'masks')
    
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)
    
    print(f"創建虛擬資料集在: {temp_dir}")
    
    # 生成虛擬圖片和 mask
    for i in range(num_images):
        # 創建隨機 RGB 圖片
        img_array = np.random.randint(0, 255, (image_size[0], image_size[1], 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        img_path = os.path.join(images_dir, f'img_{i:03d}.png')
        img.save(img_path)
        
        # 創建簡單的 mask（上半部分是天空，下半部分不是）
        mask_array = np.zeros((image_size[0], image_size[1]), dtype=np.uint8)
        # 上半部分設為 255（天空）
        mask_array[:image_size[0]//2, :] = 255
        # 添加一些隨機噪聲讓它更真實
        noise = np.random.randint(0, 50, (image_size[0], image_size[1]), dtype=np.uint8)
        mask_array = np.clip(mask_array.astype(np.int16) + noise - 25, 0, 255).astype(np.uint8)
        
        mask = Image.fromarray(mask_array)
        mask_path = os.path.join(masks_dir, f'img_{i:03d}.png')
        mask.save(mask_path)
    
    print(f"[OK] 創建了 {num_images} 組圖片和 mask")
    return temp_dir, images_dir, masks_dir


def test_dataset_loader(images_dir, masks_dir):
    """測試資料集載入器"""
    print("\n" + "="*60)
    print("測試 1: 資料集載入器")
    print("="*60)
    
    try:
        # 測試不進行資料增強
        dataloader = get_dataloader(
            images_dir=images_dir,
            masks_dir=masks_dir,
            batch_size=2,
            shuffle=False,
            transform=False,
            image_size=(256, 256),
            num_workers=0
        )
        
        print(f"[OK] 資料集大小: {len(dataloader.dataset)}")
        
        # 測試讀取一個批次
        images, masks = next(iter(dataloader))
        print(f"[OK] 圖片形狀: {images.shape}")
        print(f"[OK] Mask 形狀: {masks.shape}")
        print(f"[OK] 圖片值範圍: [{images.min():.3f}, {images.max():.3f}]")
        print(f"[OK] Mask 值範圍: [{masks.min():.3f}, {masks.max():.3f}]")
        
        # 測試資料增強
        aug_dataloader = get_dataloader(
            images_dir=images_dir,
            masks_dir=masks_dir,
            batch_size=2,
            shuffle=False,
            transform=True,
            image_size=(256, 256),
            num_workers=0
        )
        aug_images, aug_masks = next(iter(aug_dataloader))
        print(f"[OK] 資料增強測試通過")
        
        return True, dataloader
        
    except Exception as e:
        print(f"[FAIL] 測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_model():
    """測試模型"""
    print("\n" + "="*60)
    print("測試 2: U-Net 模型")
    print("="*60)
    
    try:
        # 創建模型
        model = create_unet_model(n_channels=3, n_classes=1)
        print(f"[OK] 模型創建成功")
        
        # 計算參數數量
        total_params = sum(p.numel() for p in model.parameters())
        print(f"[OK] 模型參數總數: {total_params:,}")
        
        # 測試前向傳播
        dummy_input = torch.randn(2, 3, 256, 256)
        model.eval()
        with torch.no_grad():
            output = model(dummy_input)
        
        print(f"[OK] 輸入形狀: {dummy_input.shape}")
        print(f"[OK] 輸出形狀: {output.shape}")
        print(f"[OK] 輸出值範圍: [{output.min():.3f}, {output.max():.3f}]")
        
        # 測試 predict_mask 方法
        mask = model.predict_mask(dummy_input)
        print(f"[OK] Mask 預測形狀: {mask.shape}")
        print(f"[OK] Mask 值範圍: [{mask.min():.3f}, {mask.max():.3f}]")
        
        return True, model
        
    except Exception as e:
        print(f"[FAIL] 測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_training_loop(model, dataloader):
    """測試訓練迴圈（只跑幾個 batch）"""
    print("\n" + "="*60)
    print("測試 3: 訓練迴圈（快速測試）")
    print("="*60)
    
    # 檢查 GPU 記憶體，如果小於 4GB 則使用 CPU
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if gpu_memory < 4.0:
            print(f"[WARN] GPU 記憶體只有 {gpu_memory:.1f}GB，改用 CPU 進行測試")
            use_cuda = False
    
    device = torch.device('cuda' if use_cuda else 'cpu')
    print(f"使用設備: {device}")
    
    try:
        # 清理 GPU 記憶體
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        model = model.to(device)
        model.train()
        
        # 定義損失函數和優化器
        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        
        print("\n進行 3 個 batch 的訓練測試...")
        
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (images, masks) in enumerate(dataloader):
            if batch_idx >= 3:  # 只測試 3 個 batch
                break
            
            # 移到設備
            images = images.to(device)
            masks = masks.to(device)
            
            # 前向傳播
            optimizer.zero_grad()
            outputs = model(images)
            
            # 計算損失
            loss = criterion(outputs, masks)
            
            # 反向傳播
            loss.backward()
            optimizer.step()
            
            # 計算指標
            with torch.no_grad():
                metrics = calculate_metrics(outputs, masks)
            
            total_loss += loss.item()
            num_batches += 1
            
            print(f"  Batch {batch_idx + 1}: Loss={loss.item():.4f}, "
                  f"IoU={metrics['iou']:.4f}, Dice={metrics['dice']:.4f}")
        
        avg_loss = total_loss / num_batches
        print(f"\n[OK] 平均損失: {avg_loss:.4f}")
        print(f"[OK] 訓練迴圈測試通過！")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] 測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主測試函數"""
    print("="*60)
    print("天空分割專案 - 簡單測試")
    print("="*60)
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 可用: {torch.cuda.is_available()}")
    
    # 創建虛擬資料集
    temp_dir = None
    try:
        temp_dir, images_dir, masks_dir = create_dummy_dataset(num_images=10)
        
        # 測試 1: 資料集載入器
        success1, dataloader = test_dataset_loader(images_dir, masks_dir)
        if not success1:
            print("\n[FAIL] 資料集載入器測試失敗，停止測試")
            return
        
        # 測試 2: 模型
        success2, model = test_model()
        if not success2:
            print("\n[FAIL] 模型測試失敗，停止測試")
            return
        
        # 測試 3: 訓練迴圈
        success3 = test_training_loop(model, dataloader)
        if not success3:
            print("\n[FAIL] 訓練迴圈測試失敗")
            return
        
        # 所有測試通過
        print("\n" + "="*60)
        print("[OK] 所有測試通過！")
        print("="*60)
        print("\n專案組件運作正常，可以開始使用真實資料集進行訓練。")
        print(f"\n使用範例:")
        print(f"  python train.py \\")
        print(f"    --train_images_dir data/train/images \\")
        print(f"    --train_masks_dir data/train/masks \\")
        print(f"    --val_images_dir data/val/images \\")
        print(f"    --val_masks_dir data/val/masks \\")
        print(f"    --epochs 50 --batch_size 8")
        
    except Exception as e:
        print(f"\n[FAIL] 測試過程中發生錯誤: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 清理臨時檔案
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                print(f"\n[OK] 已清理臨時檔案: {temp_dir}")
            except:
                print(f"\n[WARN] 無法清理臨時檔案: {temp_dir} (可手動刪除)")


if __name__ == '__main__':
    main()
