# 訓練指南

## 快速開始

### 1. 準備資料集

確保您的資料集結構如下：

```
data/
├── train/
│   ├── images/
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │   └── ...
│   └── masks/
│       ├── img1.png
│       ├── img2.png
│       └── ...
└── val/
    ├── images/
    │   ├── img1.jpg
    │   └── ...
    └── masks/
        ├── img1.png
        └── ...
```

### 2. 基本訓練命令

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

### 3. 完整參數範例

```bash
python train.py \
    --train_images_dir data/train/images \
    --train_masks_dir data/train/masks \
    --val_images_dir data/val/images \
    --val_masks_dir data/val/masks \
    --epochs 100 \
    --batch_size 16 \
    --learning_rate 1e-4 \
    --weight_decay 1e-5 \
    --image_size 256 256 \
    --loss combined \
    --optimizer adam \
    --use_scheduler \
    --num_workers 0
```

## 參數說明

### 必需參數

- `--train_images_dir`: 訓練圖片資料夾路徑
- `--train_masks_dir`: 訓練 mask 資料夾路徑
- `--val_images_dir`: 驗證圖片資料夾路徑
- `--val_masks_dir`: 驗證 mask 資料夾路徑

### 訓練參數

- `--epochs`: 訓練輪數（預設: 50）
- `--batch_size`: 批次大小（預設: 8）
- `--learning_rate`: 學習率（預設: 1e-4）
- `--weight_decay`: 權重衰減/L2 正則化（預設: 1e-5）

### 模型參數

- `--image_size`: 圖片尺寸 [height width]（預設: 256 256）

### 損失函數和優化器

- `--loss`: 損失函數選擇
  - `bce`: BCEWithLogitsLoss（標準二進制交叉熵）
  - `dice`: Dice Loss（對類別不平衡較不敏感）
  - `combined`: BCE + Dice 組合（推薦，預設）
  
- `--optimizer`: 優化器選擇
  - `adam`: Adam 優化器（推薦，預設）
  - `sgd`: SGD 優化器

- `--use_scheduler`: 使用學習率調度器（ReduceLROnPlateau）

### 其他參數

- `--num_workers`: 資料載入進程數（Windows 建議設為 0）

## 訓練輸出

訓練過程中會自動：

1. **保存檢查點**：
   - `outputs/train_YYYYMMDD_HHMMSS/checkpoints/latest.pth`: 最新模型
   - `outputs/train_YYYYMMDD_HHMMSS/checkpoints/best.pth`: 最佳模型（IoU 最高）
   - `outputs/train_YYYYMMDD_HHMMSS/checkpoints/epoch_N.pth`: 每個 epoch 的模型

2. **TensorBoard 日誌**：
   - `outputs/train_YYYYMMDD_HHMMSS/logs/`: TensorBoard 日誌目錄
   - 查看日誌：`tensorboard --logdir outputs/train_YYYYMMDD_HHMMSS/logs`

## 監控訓練

### 使用 TensorBoard

```bash
tensorboard --logdir outputs/train_YYYYMMDD_HHMMSS/logs
```

然後在瀏覽器打開 `http://localhost:6006`

### 訓練日誌包含

- Loss（訓練和驗證）
- IoU（Intersection over Union）
- Dice Score
- Pixel Accuracy

## 評估指標說明

- **IoU (Intersection over Union)**: 預測區域與真實區域的重疊比例，範圍 [0, 1]，越高越好
- **Dice Score**: 類似 F1 分數，對類別不平衡較不敏感，範圍 [0, 1]，越高越好
- **Pixel Accuracy**: 正確預測的像素比例，範圍 [0, 1]，越高越好

## 訓練技巧

1. **批次大小**: 根據 GPU 記憶體調整，通常 8-16 較好
2. **學習率**: 從 1e-4 開始，如果損失不下降可嘗試 1e-3 或 5e-5
3. **損失函數**: 推薦使用 `combined`（BCE + Dice），通常效果最好
4. **學習率調度器**: 如果驗證損失長時間不下降，啟用 `--use_scheduler`
5. **資料增強**: 訓練時自動啟用，驗證時自動關閉

## 常見問題

### Q: 訓練時記憶體不足？

A: 減小 `--batch_size`（例如改為 4）或減小 `--image_size`（例如改為 128 128）

### Q: 訓練損失不下降？

A: 
- 檢查學習率是否合適（嘗試調整 `--learning_rate`）
- 確認資料集標註正確
- 嘗試使用 `--use_scheduler`

### Q: 驗證指標很低？

A:
- 確認驗證集與訓練集來自相同分佈
- 檢查是否有過擬合（訓練指標高但驗證指標低）
- 嘗試增加資料增強或正則化（`--weight_decay`）
