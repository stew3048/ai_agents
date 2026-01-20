# 天空區域辨識 (Sky Segmentation) 專案

## 專案目標
輸入圖片，輸出天空的 pixel-level mask。

## 技術架構
- 使用現成模型：U-Net / DeepLabV3 作為 baseline
- 框架：PyTorch

## 專案結構
```
.
├── data/                       # 資料集存放位置
│   ├── toy/                    # Toy dataset（測試用）
│   │   ├── images/             # 50 張 .jpg 圖片
│   │   └── masks/              # 50 張 .png mask
│   ├── images/                 # 圖片資料夾（或 train/images/, val/images/ 等）
│   └── masks/                  # Mask 資料夾（或 train/masks/, val/masks/ 等）
├── models/                     # 模型定義與權重
│   ├── __init__.py
│   └── unet.py                 # U-Net 模型定義
├── utils/                      # 工具函數
│   ├── __init__.py
│   ├── dataset.py              # 資料集載入器 (DataLoader)
│   └── metrics.py              # 評估指標 (IoU, Dice, Accuracy)
├── scripts/                    # 驗證腳本
│   └── verify_toy_loader_visual.py  # 視覺化驗收
├── outputs/                    # 輸出結果
│   └── toy_loader_vis/         # DataLoader 視覺化驗收結果
├── main.py                     # 主程序入口
├── train.py                    # 訓練腳本
├── generate_toy_dataset.py     # 產生 Toy Dataset
├── check_toy_dataset.py        # 檔案配對檢查
├── sanity_check_dataloader.py  # DataLoader 輸出驗證
├── test_dataset.py             # 測試資料集載入器
├── test_model.py               # 測試 U-Net 模型
├── TRAINING_GUIDE.md           # 訓練指南
└── requirements.txt            # 依賴套件
```

## 安裝依賴
```bash
pip install -r requirements.txt
```

## 資料集準備

### 支援的資料集格式

**方式 1: 簡單格式**
```
data/
├── images/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
└── masks/
    ├── img1.png
    ├── img2.png
    └── ...
```

**方式 2: 分組格式（train/val/test）**
```
data/
├── train/
│   ├── images/
│   └── masks/
├── val/
│   ├── images/
│   └── masks/
└── test/
    ├── images/
    └── masks/
```

### 測試資料集載入器

在準備好資料集後，可以執行以下命令測試：

```bash
python test_dataset.py
```

### 測試 U-Net 模型

測試模型結構和輸入輸出尺寸：

```bash
python test_model.py
```

### 簡單完整測試（推薦）

使用虛擬資料快速測試整個流程是否正常：

```bash
python simple_test.py
```

這個測試會：
- 自動生成虛擬測試資料
- 測試資料集載入器
- 測試模型前向傳播
- 測試訓練迴圈（幾個 batch）
- 自動清理臨時檔案

**適合第一次使用時快速驗證所有組件是否正常運作！**

## 模型架構

### U-Net

- **輸入**: RGB 圖片 [B, 3, 256, 256]
- **輸出**: 天空 mask logits [B, 1, 256, 256]
- **架構特點**:
  - Encoder-Decoder 結構
  - 5 個下採樣階段（編碼器）
  - 4 個上採樣階段（解碼器）
  - Skip connections 保留細節資訊
  - 總參數約 31M

## 訓練模型

### 快速開始

```bash
python train.py \
    --train_images_dir data/train/images \
    --train_masks_dir data/train/masks \
    --val_images_dir data/val/images \
    --val_masks_dir data/val/masks \
    --epochs 50 \
    --batch_size 8 \
    --learning_rate 1e-4
```

### 訓練功能

- **損失函數**: 支援 BCE Loss、Dice Loss、Combined Loss（推薦）
- **優化器**: 支援 Adam、SGD
- **學習率調度**: 可選的 ReduceLROnPlateau
- **評估指標**: IoU、Dice Score、Pixel Accuracy
- **自動保存**: 最新模型、最佳模型、每個 epoch 的檢查點
- **TensorBoard**: 自動記錄訓練日誌

詳細說明請參考 [TRAINING_GUIDE.md](TRAINING_GUIDE.md)

## 使用方式
（待完成）

---

## 開發進度

### ✅ 已完成

#### 1. 專案初始化
- U-Net 模型實作
- DataLoader 實作（支援同步增強）
- 評估指標（IoU、Dice、Pixel Accuracy）
- 訓練腳本

#### 2. Toy Dataset 建立
產生 50 張模擬天空場景的測試資料：
```bash
python generate_toy_dataset.py --output_dir data/toy --num_images 50
```

#### 3. Data Loader Sanity Check ✓
確保 DataLoader 輸出正確，為後續訓練排除資料問題。

**檔案配對檢查：**
```bash
python check_toy_dataset.py
```
- 確認每張 image 都有對應的 mask
- 統計配對數量與缺失清單

**DataLoader 輸出驗證：**
```bash
python sanity_check_dataloader.py
```
- 單張 image: (256, 256, 3), uint8
- 單張 mask: (256, 256), uint8, {0, 255}
- Batch image: torch.Size([B, 3, 256, 256]), float32, [0, 1]
- Batch mask: torch.Size([B, 1, 256, 256]), float32, {0.0, 1.0}

**視覺化驗收：**
```bash
python scripts/verify_toy_loader_visual.py
```
輸出到 `outputs/toy_loader_vis/`：
- `XXX_1_original.png` - 原始圖片
- `XXX_2_mask.png` - GT mask（白=天空）
- `XXX_3_overlay.png` - Mask 疊在圖片上
- `XXX_4_resized_overlay.png` - Resize 後的 overlay

### ⏳ 待完成
- [ ] 使用真實資料集（SkyFinder）訓練
- [ ] 預測腳本 (predict.py)
- [ ] 模型評估與比較
