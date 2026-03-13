"""
SkyFinder Camera 10066 PyTorch Dataset

這個 Dataset 重用 utils/dataset.py 中已驗證的核心邏輯，
只加入讀取 splits_10066.json 的功能。

核心邏輯（與 utils/dataset.py 相同）：
- image: resize 用 BILINEAR
- mask: resize 用 NEAREST
- image 轉為 [0, 1] 範圍
- mask 二值化為 {0, 1}
"""

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import torchvision.transforms.functional as TF
import random


class SkyFinder10066Dataset(Dataset):
    """
    SkyFinder Camera 10066 Dataset
    
    重用 utils/dataset.py 的核心邏輯，加入 JSON split 支援。
    
    Args:
        splits_json: splits JSON 檔案路徑
        split: 'train', 'val', 或 'test'
        images_dir: 圖片目錄（如果 None，從 JSON metadata 讀取）
        masks_dir: mask 目錄（如果 None，從 JSON metadata 讀取）
        image_size: 目標尺寸 (H, W)，預設 (256, 256)
        transform: 是否進行資料增強（預設 train=True, val/test=False）
    """
    
    def __init__(
        self,
        splits_json='outputs/splits_10066.json',
        split='train',
        images_dir=None,
        masks_dir=None,
        image_size=(256, 256),
        transform=None
    ):
        # 載入 splits JSON
        with open(splits_json, 'r', encoding='utf-8') as f:
            self.splits_data = json.load(f)
        
        # 驗證 split 名稱
        if split not in ['train', 'val', 'test']:
            raise ValueError(f"split must be 'train', 'val', or 'test', got '{split}'")
        
        self.split = split
        self.samples = self.splits_data[split]
        
        # 設定目錄路徑
        metadata = self.splits_data['metadata']
        self.images_dir = images_dir or metadata['images_dir']
        self.masks_dir = masks_dir or metadata['masks_dir']
        
        # 尺寸設定（與 utils/dataset.py 相同）
        self.image_size = image_size
        
        # Transform 設定：預設 train=True, val/test=False
        if transform is None:
            self.transform = (split == 'train')
        else:
            self.transform = transform
        
        # 驗證資料存在
        self._verify_data()
        
        print(f"成功載入 {len(self.samples)} 組圖片-mask 配對 (split={split})")
    
    def _verify_data(self):
        """驗證所有檔案都存在"""
        missing = []
        for sample in self.samples:
            img_path = os.path.join(self.images_dir, sample['image'])
            mask_path = os.path.join(self.masks_dir, sample['mask'])
            
            if not os.path.exists(img_path):
                missing.append(f"Image: {img_path}")
            if not os.path.exists(mask_path):
                missing.append(f"Mask: {mask_path}")
        
        if missing:
            raise FileNotFoundError(f"Missing files:\n" + "\n".join(missing[:5]))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        獲取單一資料樣本
        
        核心邏輯與 utils/dataset.py 的 SkySegmentationDataset.__getitem__ 相同：
        - image: resize 用 BILINEAR，轉為 [0, 1]
        - mask: resize 用 NEAREST，二值化為 {0, 1}
        
        返回:
            image: 處理後的圖片張量 [C, H, W]，值範圍 [0, 1]
            mask: 處理後的 mask 張量 [1, H, W]，值為 0 或 1
        """
        sample = self.samples[idx]
        
        # === 以下邏輯與 utils/dataset.py 相同 ===
        
        # 讀取圖片
        image_path = os.path.join(self.images_dir, sample['image'])
        image = Image.open(image_path).convert('RGB')  # 確保是 RGB 格式
        
        # 讀取對應的 mask
        mask_path = os.path.join(self.masks_dir, sample['mask'])
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
        
        與 utils/dataset.py 的 _apply_augmentation 相同：
        1. 隨機水平翻轉（機率 0.5）
        2. 隨機垂直翻轉（機率 0.5）
        3. 隨機裁切並調整大小
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
    
    def get_filename(self, idx):
        """取得指定 index 的檔名（用於除錯）"""
        return self.samples[idx]['image']


def get_skyfinder_dataloader(
    splits_json='outputs/splits_10066.json',
    split='train',
    batch_size=8,
    shuffle=None,
    num_workers=0,
    image_size=(256, 256),
    transform=None,
    **kwargs
):
    """
    建立 SkyFinder 10066 DataLoader
    
    與 utils/dataset.py 的 get_dataloader 邏輯相同，
    只是資料來源改為讀取 JSON split 檔。
    
    Args:
        splits_json: splits JSON 檔案路徑
        split: 'train', 'val', 或 'test'
        batch_size: batch 大小
        shuffle: 是否打亂（預設 train=True, val/test=False）
        num_workers: DataLoader workers
        image_size: 目標尺寸
        transform: 是否資料增強（預設 train=True, val/test=False）
        **kwargs: 其他 DataLoader 參數
    
    Returns:
        DataLoader
    """
    dataset = SkyFinder10066Dataset(
        splits_json=splits_json,
        split=split,
        image_size=image_size,
        transform=transform
    )
    
    # 預設 shuffle 設定（與 utils/dataset.py 一致）
    if shuffle is None:
        shuffle = (split == 'train')
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        **kwargs
    )
    
    return dataloader
