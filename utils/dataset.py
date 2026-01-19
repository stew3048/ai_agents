"""
天空分割資料集載入器 (Sky Segmentation Dataset Loader)
支援讀取圖片與 mask，並提供資料增強功能
"""

import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random


class SkySegmentationDataset(Dataset):
    """
    天空分割資料集類別
    
    支援的資料集結構：
    1. 簡單格式：
       data/
         images/
           img1.jpg, img2.jpg, ...
         masks/
           img1.png, img2.png, ...
    
    2. 分組格式（train/val/test）：
       data/
         train/
           images/
           masks/
         val/
           images/
           masks/
    
    參數:
        images_dir: 圖片資料夾路徑
        masks_dir: mask 資料夾路徑
        transform: 是否在訓練時進行資料增強（預設 True）
        image_size: 目標圖片尺寸 (height, width)，預設 (256, 256)
    """
    
    def __init__(self, images_dir, masks_dir, transform=True, image_size=(256, 256)):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.image_size = image_size
        
        # 獲取所有圖片檔案名稱（不包含路徑）
        # 假設圖片和 mask 使用相同的檔名，但可能副檔名不同
        self.image_files = []
        
        # 支援常見圖片格式
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
        
        if os.path.exists(images_dir):
            # 讀取所有圖片檔案
            for filename in os.listdir(images_dir):
                if any(filename.lower().endswith(ext) for ext in valid_extensions):
                    # 檢查對應的 mask 檔案是否存在
                    mask_name = self._find_mask_file(filename)
                    if mask_name and os.path.exists(os.path.join(masks_dir, mask_name)):
                        self.image_files.append(filename)
        
        if len(self.image_files) == 0:
            raise ValueError(
                f"在 {images_dir} 中找不到有效的圖片檔案，"
                f"或對應的 mask 檔案在 {masks_dir} 中不存在。"
                f"請確認資料集路徑正確。"
            )
        
        print(f"成功載入 {len(self.image_files)} 組圖片-mask 配對")
    
    def _find_mask_file(self, image_filename):
        """
        根據圖片檔名找到對應的 mask 檔名
        
        參數:
            image_filename: 圖片檔名
        
        返回:
            mask 檔名，如果找不到則返回 None
        """
        # 移除圖片副檔名
        base_name = os.path.splitext(image_filename)[0]
        
        # 常見的 mask 副檔名
        mask_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']
        
        # 先檢查相同檔名但不同副檔名
        for ext in mask_extensions:
            mask_name = base_name + ext
            if os.path.exists(os.path.join(self.masks_dir, mask_name)):
                return mask_name
        
        # 如果找不到，返回預設的 .png 檔名（讓上層檢查是否存在）
        return base_name + '.png'
    
    def __len__(self):
        """返回資料集大小"""
        return len(self.image_files)
    
    def __getitem__(self, idx):
        """
        獲取單一資料樣本
        
        參數:
            idx: 樣本索引
        
        返回:
            image: 處理後的圖片張量 [C, H, W]，值範圍 [0, 1]
            mask: 處理後的 mask 張量 [1, H, W]，值範圍 [0, 1]（二進制）
        """
        # 讀取圖片
        image_path = os.path.join(self.images_dir, self.image_files[idx])
        image = Image.open(image_path).convert('RGB')  # 確保是 RGB 格式
        
        # 讀取對應的 mask
        mask_name = self._find_mask_file(self.image_files[idx])
        mask_path = os.path.join(self.masks_dir, mask_name)
        mask = Image.open(mask_path).convert('L')  # 轉為灰階（單通道）
        
        # 調整尺寸（確保圖片和 mask 尺寸一致）
        image = image.resize(self.image_size, Image.BILINEAR)
        mask = mask.resize(self.image_size, Image.NEAREST)  # mask 使用最近鄰插值避免模糊
        
        # 資料增強（僅在訓練時）
        if self.transform:
            image, mask = self._apply_augmentation(image, mask)
        
        # 轉換為張量
        # 圖片：轉為 [0, 1] 範圍的張量
        image_tensor = TF.to_tensor(image)
        
        # Mask：轉為 [0, 1] 範圍的張量
        # 如果 mask 是二進制的（0 和 255），先正規化到 [0, 1]
        mask_array = np.array(mask, dtype=np.float32)
        if mask_array.max() > 1.0:
            mask_array = mask_array / 255.0  # 正規化到 [0, 1]
        
        # 轉為張量並增加通道維度 [1, H, W]
        mask_tensor = torch.from_numpy(mask_array).unsqueeze(0)
        
        # 確保 mask 是二進制的（0 或 1）
        mask_tensor = (mask_tensor > 0.5).float()
        
        return image_tensor, mask_tensor
    
    def _apply_augmentation(self, image, mask):
        """
        應用資料增強
        
        包含的增強方式：
        1. 隨機水平翻轉（機率 0.5）
        2. 隨機垂直翻轉（機率 0.5）
        3. 隨機裁切並調整大小
        
        參數:
            image: PIL Image 物件
            mask: PIL Image 物件
        
        返回:
            augmented_image: 增強後的圖片
            augmented_mask: 增強後的 mask
        """
        # 隨機水平翻轉
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        
        # 隨機垂直翻轉
        if random.random() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)
        
        # 隨機裁切並調整大小
        # 裁切比例範圍：0.7 到 1.0
        if random.random() > 0.5:
            crop_ratio = random.uniform(0.7, 1.0)
            crop_size = (
                int(self.image_size[0] * crop_ratio),
                int(self.image_size[1] * crop_ratio)
            )
            
            # 隨機裁切位置
            i = random.randint(0, image.height - crop_size[0])
            j = random.randint(0, image.width - crop_size[1])
            
            # 對圖片和 mask 進行相同的裁切
            image = TF.crop(image, i, j, crop_size[0], crop_size[1])
            mask = TF.crop(mask, i, j, crop_size[0], crop_size[1])
            
            # 調整回目標尺寸
            image = image.resize(self.image_size, Image.BILINEAR)
            mask = mask.resize(self.image_size, Image.NEAREST)
        
        return image, mask


def get_dataloader(images_dir, masks_dir, batch_size=8, shuffle=True, 
                   transform=True, image_size=(256, 256), num_workers=0):
    """
    創建資料載入器 (DataLoader)
    
    參數:
        images_dir: 圖片資料夾路徑
        masks_dir: mask 資料夾路徑
        batch_size: 批次大小，預設 8
        shuffle: 是否隨機打亂資料，預設 True（訓練時使用）
        transform: 是否進行資料增強，預設 True（訓練時使用）
        image_size: 目標圖片尺寸 (height, width)，預設 (256, 256)
        num_workers: 資料載入的進程數，Windows 建議設為 0
    
    返回:
        DataLoader 物件
    """
    dataset = SkySegmentationDataset(
        images_dir=images_dir,
        masks_dir=masks_dir,
        transform=transform,
        image_size=image_size
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return dataloader
