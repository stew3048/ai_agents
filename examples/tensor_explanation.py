"""
Tensor 概念說明

Tensor 是什麼？
- Tensor 是多維陣列（multi-dimensional array）
- 可以理解為「向量的推廣」
- 在深度學習中，所有資料都以 tensor 形式表示
"""

import torch
import numpy as np
from PIL import Image

print("=" * 60)
print("  Tensor 概念說明")
print("=" * 60)
print()

# ============================================
# 1. 從簡單到複雜：Scalar → Vector → Matrix → Tensor
# ============================================
print("1. 資料維度的演進：")
print("-" * 60)

# Scalar（純量）：0 維，單一數字
scalar = 5.0
print(f"Scalar (0維): {scalar}")
print(f"  形狀: {type(scalar)}")
print()

# Vector（向量）：1 維，一列數字
vector = torch.tensor([1, 2, 3, 4, 5])
print(f"Vector (1維): {vector}")
print(f"  形狀: {vector.shape}  # [5] = 5 個元素")
print()

# Matrix（矩陣）：2 維，表格形式
matrix = torch.tensor([[1, 2, 3],
                       [4, 5, 6]])
print(f"Matrix (2維):")
print(matrix)
print(f"  形狀: {matrix.shape}  # [2, 3] = 2 列 × 3 行")
print()

# Tensor（張量）：3 維以上
tensor_3d = torch.tensor([[[1, 2], [3, 4]],
                          [[5, 6], [7, 8]]])
print(f"Tensor (3維):")
print(tensor_3d)
print(f"  形狀: {tensor_3d.shape}  # [2, 2, 2] = 2 層 × 2 列 × 2 行")
print()

# ============================================
# 2. 圖片在深度學習中的表示
# ============================================
print("=" * 60)
print("2. 圖片在深度學習中的表示：")
print("-" * 60)

# 原始圖片（PIL Image）
print("原始圖片（PIL Image）:")
print("  - 格式: PIL.Image")
print("  - 尺寸: (寬, 高) = (W, H)")
print("  - 顏色: RGB 三個通道")
print("  - 值範圍: [0, 255] (uint8)")
print()

# 轉換為 Tensor
print("轉換為 Tensor:")
print("  - 格式: torch.Tensor")
print("  - 形狀: [C, H, W] = [通道數, 高度, 寬度]")
print("  - 顏色: RGB = 3 個通道")
print("  - 值範圍: [0, 1] (float32，已正規化)")
print()

# 範例：256x256 RGB 圖片
example_image = torch.rand(3, 256, 256)  # [C, H, W]
print(f"範例：256×256 RGB 圖片")
print(f"  形狀: {example_image.shape}  # [3, 256, 256]")
print(f"  含義: 3 個通道（RGB），每個通道 256×256 像素")
print(f"  資料型態: {example_image.dtype}")
print(f"  值範圍: [{example_image.min():.3f}, {example_image.max():.3f}]")
print()

# ============================================
# 3. Batch（批次）的概念
# ============================================
print("=" * 60)
print("3. Batch（批次）的概念：")
print("-" * 60)

# 單張圖片：3 維 tensor [C, H, W]
single_image = torch.rand(3, 256, 256)
print(f"單張圖片: {single_image.shape}  # [C, H, W]")
print()

# 批次圖片：4 維 tensor [B, C, H, W]
batch_images = torch.rand(4, 3, 256, 256)  # batch_size=4
print(f"批次圖片（4張）: {batch_images.shape}  # [B, C, H, W]")
print(f"  含義: 4 張圖片，每張 3 通道，256×256 像素")
print()

# ============================================
# 4. Dataset Loader 輸出的格式
# ============================================
print("=" * 60)
print("4. Dataset Loader 輸出的格式：")
print("-" * 60)

print("當您使用 DataLoader 時：")
print()
print("  輸入: 原始圖片檔案（JPG/PNG）")
print("    - 格式: 檔案")
print("    - 尺寸: 任意大小")
print("    - 值範圍: [0, 255]")
print()
print("  ↓ DataLoader 處理")
print()
print("  輸出: Tensor")
print("    - 格式: torch.Tensor")
print("    - 形狀: [B, C, H, W] = [批次大小, 3, 256, 256]")
print("    - 值範圍: [0, 1] (已正規化)")
print("    - 資料型態: float32")
print()

# ============================================
# 5. CV 和 DL Model 都接受 Tensor
# ============================================
print("=" * 60)
print("5. CV 和 DL Model 都接受 Tensor：")
print("-" * 60)

print("CV Baseline:")
print("  - 輸入: torch.Tensor [C, H, W] 或 [B, C, H, W]")
print("  - 值範圍: [0, 1]")
print("  - 內部會轉換為 numpy 進行處理")
print()

print("DL Model (U-Net):")
print("  - 輸入: torch.Tensor [B, C, H, W]")
print("  - 值範圍: [0, 1]")
print("  - 直接在 GPU/CPU 上運算")
print()

print("兩者都使用相同的輸入格式！")
print()

# ============================================
# 6. 實際範例
# ============================================
print("=" * 60)
print("6. 實際範例：")
print("-" * 60)

# 模擬 DataLoader 輸出
batch_size = 4
images = torch.rand(batch_size, 3, 256, 256)  # [B, C, H, W]
masks = torch.rand(batch_size, 1, 256, 256)   # [B, 1, H, W]

print(f"DataLoader 輸出：")
print(f"  images: {images.shape}  # [批次大小, 通道, 高, 寬]")
print(f"  masks:  {masks.shape}   # [批次大小, 通道, 高, 寬]")
print()

print("傳給模型：")
print(f"  model(images)  # 直接傳入 tensor")
print(f"  CV baseline 也可以直接處理 images[0]  # 單張圖片 [C, H, W]")
print()

print("=" * 60)
print("總結：")
print("=" * 60)
print("1. Tensor = 多維陣列（向量的推廣）")
print("2. 圖片在深度學習中 = Tensor [C, H, W] 或 [B, C, H, W]")
print("3. DataLoader 將原始圖片轉換為 Tensor")
print("4. CV 和 DL Model 都接受 Tensor 格式")
print("5. 值範圍統一為 [0, 1]（已正規化）")
print()
