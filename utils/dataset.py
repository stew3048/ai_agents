"""
天空分割資料集載入器 (Sky Segmentation Dataset Loader)
支援讀取圖片與 mask，並提供資料增強功能

功能：
- 支援從 JSON 清單讀取（split script 已過濾受損檔案，無需重複檢查）
- 支援直接從資料夾掃描（通用模式，會檢查受損檔案）
- 確保圖片和 mask 的對齊
- 支援資料增強
- 適用於任何模型訓練
"""

import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random
from typing import List, Dict, Optional

# 低光/低對比 train augmentation（僅用於 transform=True；val/test 不經此）
# 不涉及 mask 的幾何或插值，mask 保持不變、不受污染
import albumentations as A

# 微調 aug 版本（方案 A：條件式使用，僅對 night 樣本套用，強度稍弱以減少對 day 的影響）
_TRAIN_AUG_LOWLIGHT = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=(-0.35, 0.10), contrast_limit=(-0.35, 0.20), p=0.70),
    A.RandomGamma(gamma_limit=(65, 135), p=0.50),   # 65~135 表示 gamma 0.65~1.35（albumentations 需 int >=1）
    A.GaussianBlur(blur_limit=(3, 5), p=0.20),
    A.GaussNoise(var_limit=(5.0, 25.0), p=0.15),
])


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
    
    def __init__(self, images_dir, masks_dir, transform=True, image_size=(256, 256), 
                 clean_corrupted=True, verbose=True, split_list: Optional[List[Dict]] = None,
                 base_data_dir: str = "data"):
        """
        初始化資料集
        
        參數:
            images_dir: 圖片資料夾路徑（如果使用 split_list，此參數可為 None）
            masks_dir: mask 資料夾路徑（如果使用 split_list，此參數可為 None）
            transform: 是否在訓練時進行資料增強（預設 True）
            image_size: 目標圖片尺寸 (height, width)，預設 (256, 256)
            clean_corrupted: 是否自動清理受損的圖片（僅在直接掃描資料夾時使用，預設 True）
            verbose: 是否顯示清理資訊（預設 True）
            split_list: 從 split script 產生的清單（格式：[{'image': 'xxx.jpg', 'mask': 'xxx.png', 'camera_id': 'xxx'}, ...]）
                       如果提供，會直接使用此清單，跳過受損檢查（因為 split script 已過濾）
            base_data_dir: 基礎資料目錄（使用 split_list 時，會從 data/skyfinder_<camera_id>/images 讀取）
        """
        self.transform = transform
        self.image_size = image_size
        self.clean_corrupted = clean_corrupted
        self.verbose = verbose
        self.split_list = split_list
        self.base_data_dir = base_data_dir
        
        # 如果提供了 split_list，使用清單模式（split script 已過濾受損檔案）
        if split_list is not None:
            self._load_from_split_list(split_list)
        else:
            # 傳統模式：直接掃描資料夾（需要檢查受損檔案）
            self.images_dir = images_dir
            self.masks_dir = masks_dir
            self._load_from_directory()
    
    def _load_from_split_list(self, split_list: List[Dict]):
        """
        從 split script 產生的清單載入資料
        split script 已經過濾了受損檔案，所以不需要再檢查
        只儲存檔名和路徑資訊，不載入圖片到記憶體（lazy loading）
        """
        self.image_files = []
        self.image_paths = []  # 完整路徑
        self.mask_paths = []   # 完整路徑
        
        for item in split_list:
            camera_id = item['camera_id']
            image_filename = item['image']
            mask_filename = item['mask']
            
            # 構建完整路徑
            image_path = os.path.join(self.base_data_dir, f"skyfinder_{camera_id}", "images", image_filename)
            mask_path = os.path.join(self.base_data_dir, f"skyfinder_{camera_id}", "masks", mask_filename)
            
            # 檢查檔案是否存在（簡單檢查，不驗證是否受損，因為 split script 已過濾）
            if os.path.exists(image_path) and os.path.exists(mask_path):
                self.image_files.append(image_filename)
                self.image_paths.append(image_path)
                self.mask_paths.append(mask_path)
            else:
                if self.verbose:
                    print(f"  [警告] 找不到檔案: {image_path} 或 {mask_path}")
        
        if len(self.image_files) == 0:
            raise ValueError("從 split_list 中找不到有效的圖片檔案")
        
        if self.verbose:
            print(f"從 split_list 載入 {len(self.image_files)} 組圖片-mask 配對（已過濾受損檔案，lazy loading）")
    
    def _load_from_directory(self):
        """
        傳統模式：直接掃描資料夾並檢查受損檔案
        """
        # 獲取所有圖片檔案名稱（不包含路徑）
        # 假設圖片和 mask 使用相同的檔名，但可能副檔名不同
        self.image_files = []
        self.image_paths = []
        self.mask_paths = []
        
        # 支援常見圖片格式
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
        
        corrupted_count = 0
        
        if os.path.exists(self.images_dir):
            # 讀取所有圖片檔案
            for filename in os.listdir(self.images_dir):
                if any(filename.lower().endswith(ext) for ext in valid_extensions):
                    # 檢查對應的 mask 檔案是否存在
                    mask_name = self._find_mask_file(filename)
                    if mask_name and os.path.exists(os.path.join(self.masks_dir, mask_name)):
                        # 檢查圖片和 mask 是否受損
                        img_path = os.path.join(self.images_dir, filename)
                        mask_path = os.path.join(self.masks_dir, mask_name)
                        
                        # 驗證圖片和 mask 是否有效
                        if self._is_valid_pair(img_path, mask_path):
                            self.image_files.append(filename)
                            self.image_paths.append(img_path)
                            self.mask_paths.append(mask_path)
                        else:
                            # 受損的檔案，刪除它們
                            if self.clean_corrupted:
                                try:
                                    if os.path.exists(img_path):
                                        os.remove(img_path)
                                    if os.path.exists(mask_path):
                                        os.remove(mask_path)
                                    corrupted_count += 1
                                    if self.verbose:
                                        print(f"  [刪除受損檔案] {filename} 及其 mask")
                                except Exception as e:
                                    if self.verbose:
                                        print(f"  [刪除失敗] {filename}: {e}")
        
        if len(self.image_files) == 0:
            raise ValueError(
                f"在 {self.images_dir} 中找不到有效的圖片檔案，"
                f"或對應的 mask 檔案在 {self.masks_dir} 中不存在。"
                f"請確認資料集路徑正確。"
            )
        
        if self.verbose:
            if corrupted_count > 0:
                print(f"已清理 {corrupted_count} 組受損的圖片-mask 配對")
            print(f"成功載入 {len(self.image_files)} 組有效的圖片-mask 配對")
    
    def _is_valid_pair(self, img_path: str, mask_path: str) -> bool:
        """
        檢查圖片和 mask 是否都是有效的（未受損）
        
        參數:
            img_path: 圖片路徑
            mask_path: mask 路徑
        
        返回:
            True 如果兩者都有效，False 如果任一受損
        """
        try:
            # 檢查圖片
            img = Image.open(img_path)
            img.verify()  # 驗證圖片結構完整性
            img.close()
            
            # 重新打開並載入數據
            img = Image.open(img_path)
            img.load()
            
            # 檢查尺寸是否合理
            if img.size[0] <= 0 or img.size[1] <= 0:
                img.close()
                return False
            
            # 嘗試轉換為 RGB（測試是否能正常處理）
            try:
                img.convert('RGB')
            except:
                img.close()
                return False
            
            img.close()
            
            # 檢查 mask
            mask = Image.open(mask_path)
            mask.verify()
            mask.close()
            
            # 重新打開並載入數據
            mask = Image.open(mask_path)
            mask.load()
            
            # 檢查尺寸是否合理
            if mask.size[0] <= 0 or mask.size[1] <= 0:
                mask.close()
                return False
            
            mask.close()
            
            return True
            
        except Exception:
            return False
    
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
        # 讀取圖片（如果使用 split_list，直接使用完整路徑；否則使用傳統方式）
        if hasattr(self, 'image_paths') and self.image_paths:
            image_path = self.image_paths[idx]
            mask_path = self.mask_paths[idx]
        else:
            image_path = os.path.join(self.images_dir, self.image_files[idx])
            mask_name = self._find_mask_file(self.image_files[idx])
            mask_path = os.path.join(self.masks_dir, mask_name)
        
        try:
            image = Image.open(image_path).convert('RGB')  # 確保是 RGB 格式
        except Exception as e:
            raise RuntimeError(f"無法讀取圖片 {image_path}: {e}")
        
        try:
            mask = Image.open(mask_path).convert('L')  # 轉為灰階（單通道）
        except Exception as e:
            raise RuntimeError(f"無法讀取 mask {mask_path}: {e}")
        
        # 調整尺寸（確保圖片和 mask 尺寸一致）
        image = image.resize(self.image_size, Image.BILINEAR)
        mask = mask.resize(self.image_size, Image.NEAREST)  # mask 使用最近鄰插值避免模糊
        
        # 資料增強（僅在訓練時，且僅對 night 樣本套用強 aug）
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
        應用 train 資料增強（方案 B：條件式）：
        - 僅對 night 樣本（luma < 0.20）套用強 aug
        - day 樣本（luma >= 0.20）不做 aug，保持原樣
        - RandomBrightnessContrast、RandomGamma、GaussianBlur、GaussNoise
        - 上述皆不修改 mask；mask 原樣回傳，無插值污染。
        """
        # 計算圖片 luma（用於判斷是否為 night 樣本）
        img_np = np.array(image, dtype=np.float32) / 255.0  # [H, W, 3], [0, 1]
        # 轉為 linear RGB 並計算 luma
        def srgb_to_linear(v):
            return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)
        R_lin = srgb_to_linear(img_np[:, :, 0])
        G_lin = srgb_to_linear(img_np[:, :, 1])
        B_lin = srgb_to_linear(img_np[:, :, 2])
        mean_luma = (0.2126 * R_lin + 0.7152 * G_lin + 0.0722 * B_lin).mean()
        
        # 僅對 night 樣本（luma < 0.20）套用強 aug
        if mean_luma < 0.20:
            img_np_uint8 = (img_np * 255).astype(np.uint8)
            out = _TRAIN_AUG_LOWLIGHT(image=img_np_uint8)
            return Image.fromarray(out["image"]), mask
        else:
            # day 樣本不做 aug，保持原樣
            return image, mask


def get_dataloader(images_dir=None, masks_dir=None, batch_size=8, shuffle=True, 
                   transform=True, image_size=(256, 256), num_workers=0,
                   split_list: Optional[List[Dict]] = None, base_data_dir: str = "data"):
    """
    創建資料載入器 (DataLoader)
    
    參數:
        images_dir: 圖片資料夾路徑（如果使用 split_list，可為 None）
        masks_dir: mask 資料夾路徑（如果使用 split_list，可為 None）
        batch_size: 批次大小，預設 8
        shuffle: 是否隨機打亂資料，預設 True（訓練時使用）
        transform: 是否進行資料增強，預設 True（訓練時使用）
        image_size: 目標圖片尺寸 (height, width)，預設 (256, 256)
        num_workers: 資料載入的進程數，Windows 建議設為 0
        split_list: 從 split script 產生的清單（如果提供，會使用此清單，跳過受損檢查）
        base_data_dir: 基礎資料目錄（使用 split_list 時使用）
    
    返回:
        DataLoader 物件
    """
    dataset = SkySegmentationDataset(
        images_dir=images_dir,
        masks_dir=masks_dir,
        transform=transform,
        image_size=image_size,
        split_list=split_list,
        base_data_dir=base_data_dir
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return dataloader


def load_splits_from_json(json_file: str = 'outputs/multi_camera_splits.json', 
                          split_name: str = 'train') -> List[Dict]:
    """
    從 split script 產生的 JSON 檔案載入特定 split 的清單
    
    參數:
        json_file: JSON 檔案路徑
        split_name: 要載入的 split 名稱（'train', 'val', 'test', 'sanity'）
    
    返回:
        清單格式：[{'image': 'xxx.jpg', 'mask': 'xxx.png', 'camera_id': 'xxx'}, ...]
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if split_name not in data:
        raise ValueError(f"JSON 檔案中找不到 '{split_name}' split")
    
    return data[split_name]
