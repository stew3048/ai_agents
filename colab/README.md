# Colab 專屬訓練版本

本資料夾包含專為 Google Colab 環境設計的訓練和評估腳本，所有路徑和參數都可通過 CLI 指定，方便在不同環境中使用。

## 檔案結構

```
colab/
├── README.md                    # 本說明文件
├── train_all.py                 # 統一訓練入口（可選擇模型）
├── eval_all.py                  # 統一評估入口
├── create_splits.py             # 數據準備（參數化版本）
├── train_dl.py                  # DL baseline 訓練（參數化版本）
├── train_dino_mlp.py            # DINO-MLP 訓練（參數化版本）
├── train_dino_simple_cnn.py    # DINO-SimpleCNN 訓練（參數化版本）
├── train_dino_fpn.py            # DINO-FPN 訓練（參數化版本）
├── train_dino_hybrid.py         # DINO-Hybrid 訓練（參數化版本）
├── eval_dl.py                   # DL 評估（參數化版本）
├── eval_sam.py                  # SAM 評估（參數化版本）
├── eval_clipseg.py              # CLIPSeg 評估（參數化版本）
├── eval_dino_dl.py              # DINO-DL 評估（參數化版本）
└── colab_example.ipynb          # Colab notebook 範例
```

## 環境偵測與 Device 預設值

腳本會自動偵測運行環境：
- **本機環境**：預設使用 `cpu`（避免本機 GPU 問題）
- **Colab 環境**：預設使用 `cuda`（如果有 GPU 可用）

偵測方法：
1. 檢查環境變數 `COLAB_GPU`
2. 檢查當前工作目錄是否包含 `/content`

可通過 `--device` 參數覆蓋預設值：
- `--device cpu`：強制使用 CPU
- `--device cuda`：強制使用 GPU（如果可用）
- `--device auto`：自動偵測（有 GPU 用 GPU，否則用 CPU）

## 快速開始

### 1. 數據準備

```bash
# 步驟 1：建立 metadata
python colab/create_splits.py --step metadata \
    --data_dir data \
    --outputs_dir outputs

# 步驟 2：建立 splits（train/val/test）
python colab/create_splits.py --step splits \
    --data_dir data \
    --outputs_dir outputs \
    --metadata_file outputs/metadata_all_images.csv
```

### 2. 訓練模型

#### 訓練所有模型（預設）
```bash
# 本機運行（自動使用 CPU）
python colab/train_all.py

# Colab 運行（自動使用 GPU）
python colab/train_all.py \
    --batch_size 8 \
    --image_size 512 512 \
    --num_workers 2 \
    --outputs_dir /content/drive/MyDrive/outputs
```

#### 訓練特定模型
```bash
# 只訓練 DL baseline
python colab/train_all.py --models dl

# 訓練 DL 和 DINO-MLP
python colab/train_all.py --models dl,dino-mlp

# 單獨訓練一個模型
python colab/train_dl.py \
    --data_dir data \
    --outputs_dir outputs \
    --batch_size 8 \
    --image_size 512 512 \
    --device cuda
```

### 3. 評估模型

```bash
# 評估所有模型
python colab/eval_all.py

# 評估特定模型
python colab/eval_all.py --models dl,sam

# 單獨評估一個模型
python colab/eval_dl.py \
    --data_dir data \
    --outputs_dir outputs \
    --test_list outputs/test_list.txt \
    --image_size 512 512
```

## 參數說明

### train_all.py

**模型選擇**：
- `--models`：要訓練的模型（預設：`all`）
  - `all`：訓練所有模型
  - `dl`：DL baseline
  - `dino-mlp`, `dino-simplecnn`, `dino-fpn`, `dino-hybrid`：DINO 模型
  - 可多選：`--models dl,dino-mlp`（逗號分隔）

**路徑參數**：
- `--data_dir`：數據根目錄（預設：`data`）
- `--outputs_dir`：輸出目錄（預設：`outputs`）
- `--train_list`：train_list.txt 路徑（預設：`{outputs_dir}/train_list.txt`）
- `--val_list`：val_list.txt 路徑（預設：`{outputs_dir}/val_list.txt`）

**訓練參數**：
- `--batch_size`：批次大小（預設：2，Colab 可用 8-16）
- `--epochs`：訓練輪數（預設：10）
- `--learning_rate`：學習率（預設：1e-4）
- `--image_size`：影像尺寸（預設：256 256，可指定如 `512 512`）
- `--device`：設備（`cpu`/`cuda`/`auto`，預設：自動偵測環境）
- `--num_workers`：數據載入進程數（預設：本機 0，Colab 可用 2-4）

### eval_all.py

**模型選擇**：
- `--models`：要評估的模型（預設：`all`）
  - `all`：評估所有模型
  - `dl`, `sam`, `clipseg`, `dino-dl`：特定模型
  - 可多選：`--models dl,sam`（逗號分隔）

**路徑參數**：
- `--data_dir`：數據根目錄（預設：`data`）
- `--outputs_dir`：輸出目錄（預設：`outputs`）
- `--test_list`：test_list.txt 路徑（預設：`{outputs_dir}/test_list.txt`）
- `--checkpoint_dir`：checkpoint 目錄（預設：`{outputs_dir}`，自動尋找）
- `--output_csv`：輸出 CSV 路徑（預設：`{outputs_dir}/metrics_summary.csv`）

**評估參數**：
- `--image_size`：與訓練時一致的 image_size（預設：256 256）

### create_splits.py

**步驟選擇**：
- `--step`：執行步驟（`metadata` 或 `splits`）
  - `metadata`：建立 metadata_all_images.csv
  - `splits`：建立 train/val/test splits

**路徑參數**：
- `--data_dir`：數據根目錄（預設：`data`）
- `--outputs_dir`：輸出目錄（預設：`outputs`）
- `--metadata_file`：metadata CSV 路徑（預設：`{outputs_dir}/metadata_all_images.csv`）

## 使用範例

### 本機運行（預設值，自動使用 CPU）
```bash
python colab/train_all.py
```

### Colab 運行（自動偵測環境，預設使用 GPU）
```bash
python colab/train_all.py \
    --batch_size 8 \
    --image_size 512 512 \
    --num_workers 2 \
    --outputs_dir /content/drive/MyDrive/outputs
```

### 只訓練特定模型
```bash
python colab/train_all.py --models dl,dino-mlp
```

### 單獨訓練一個模型
```bash
python colab/train_dl.py \
    --data_dir /content/data \
    --outputs_dir /content/outputs \
    --image_size 512 512 \
    --batch_size 8 \
    --device cuda
```

## 注意事項

1. 所有腳本都會自動添加專案根目錄到 `sys.path`，確保可以 import `models`, `utils` 等模組
2. 路徑支援相對路徑和絕對路徑
3. 預設值適合本機運行，Colab 可通過 CLI 參數覆蓋
4. 訓練和評估時請確保 `image_size` 參數與訓練時一致

## Colab Notebook 範例

請參考 `colab_example.ipynb` 查看完整的 Colab 使用範例。
