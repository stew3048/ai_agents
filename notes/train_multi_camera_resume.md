# train_multi_camera.py：中斷可續訓的訓練腳本

> 此腳本為專案**訓練腳本基底**，之後新增或修改訓練邏輯時應以此為範本，並保留「中斷可續訓」設計。

## 功能摘要

- 使用 `outputs/multi_camera_splits.json` 的 train / sanity / val 做 U-Net 訓練
- **中斷後重跑**：自動偵測未完成 run，載入 `latest.pth`，從下一 epoch 繼續
- 每 epoch：train loss/IoU、sanity IoU、val loss/IoU/Dice/PixelAcc，寫入 `training_log.csv`
- 依 val IoU 存 `best.pth`，每 epoch 存 `latest.pth`
- 訓練結束後用 `best.pth` 產生 val overlay（最好 / 最差各 5 張）

## 續訓邏輯（務必保留）

### 1. 選擇輸出目錄：續訓 vs 新 run

```
outputs/
  train_multi_camera_20260124_004036/   # 依 mtime 取「最新」
    checkpoints/
      latest.pth
      best.pth
    training_log.csv
    val_overlays/
```

- 掃描 `outputs/train_multi_camera_*`，取 **mtime 最新** 的目錄
- 若該目錄有 `training_log.csv`：
  - `completed_epochs = (CSV 行數 - 1)`，不含 header
  - 若 `completed_epochs < epochs` → **續訓**，`run_dir = 該目錄`
  - 否則 → **新 run**，建立 `outputs/train_multi_camera_YYYYMMDD_HHMMSS`

### 2. 續訓時的起始狀態

- `start_epoch = completed_epochs + 1`
- `best_val_iou = max(既有 CSV 的 val_iou)`
- 若存在 `checkpoints/latest.pth`：
  - `model.load_state_dict(checkpoint['model_state_dict'])`
  - `optimizer.load_state_dict(checkpoint['optimizer_state_dict'])`

### 3. 每 epoch 結束

- 該 epoch 的 metrics **追加**寫入 `training_log.csv`（`'a'` mode）
- `save_checkpoint(..., is_best=(val_iou > best_val_iou))`
  - 每次寫 `latest.pth`
  - 若 `is_best` 再寫 `best.pth`

### 4. checkpoint 格式

```python
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'val_iou': val_iou
}
```

## 使用方式

```bash
# 直接執行；中斷後再跑同一支，會自動續訓
python train_multi_camera.py
```

- 預設 `epochs=6`、`batch_size=2`、`image_size=(160,160)`、`lr=1e-4`
- 有 GPU 時自動 AMP；不足則改 CPU

## 相關檔案

- 主程式：`train_multi_camera.py`
- 規則：`.cursor/rules/train-script-base.mdc`
- 訓練總覽：`TRAINING_GUIDE.md`
