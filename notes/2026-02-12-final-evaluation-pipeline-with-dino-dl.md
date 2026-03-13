# Sky Segmentation 最終評估流程：DINO → DL 整合

> 今日重點：執行統一評估流程 plan，整合 DINO → DL 四種 decoder 方案，建立 in-domain split，重新訓練所有模型，公平比較所有方法。  
> 日期：2026-02-12

---

## 一、今日目標

執行 **Sky Segmentation Final Evaluation Pipeline** plan，完成以下目標：

1. **建立統一的資料切分**（in-domain split：每個 camera 80/10/10）
2. **重新訓練所有模型**：
   - DL baseline (U-Net)
   - DINO → DL pipeline（四種 decoder 方案）
3. **公平評估所有方法**：
   - DL baseline
   - VLM → SAM pipeline
   - VLM → CLIPSeg pipeline
   - DINO → DL pipeline（四種方案）
4. **產出統一 metrics_summary.csv**，比較所有方法在相同 test_list.txt 上的表現

---

## 二、Plan 文件位置與內容摘要

**Plan 文件**：`C:\Users\yiching\.cursor\plans\sky_segmentation_final_evaluation_pipeline_7b22261e.plan.md`

### Plan 執行步驟

**Step 1: 建立 metadata.csv（所有影像）**
- 腳本：`scripts/create_metadata_for_in_domain_split.py`
- 輸出：`outputs/metadata_all_images.csv`
- 功能：掃描所有影像，產出 metadata（image_path, camera_id, filename_or_timestamp, brightness_estimation, sky_area_ratio, is_no_sky）

**Step 2: 方案 S - 每個 camera 做 80/10/10 split（時間序）**
- 腳本：`scripts/create_in_domain_splits.py`
- 輸出：`outputs/train_list.txt`、`outputs/val_list.txt`、`outputs/test_list.txt`（**鎖死**）
- 規則：每個 camera 依 filename 排序，前 80% → train，接下來 10% → val，最後 10% → test

**Step 3: 重新訓練 DL 模型與 DINO+DL 模型（使用新 split）**

**Step 3.1: 重新訓練 DL baseline (U-Net)**
- 腳本：`scripts/train_in_domain.py`
- 參數：batch_size=2, epochs=10, learning_rate=1e-4
- 輸出：`outputs/train_in_domain_YYYYMMDD_HHMMSS/checkpoints/best.pth`

**Step 3.2: 重新訓練 DINO → DL 模型（四種方案）**

按照從簡單到複雜的順序實作：

1. **方案4：SegFormer-like Decoder（MLP-based）**
   - 模型：`models/dino_mlp_decoder.py`
   - 訓練：`scripts/train_dino_mlp.py`
   - 架構：MLP Mixer + Linear Layers → ConvTranspose2d 上採樣
   - 特點：最輕量，適合快速驗證

2. **方案1：簡單 CNN Decoder**
   - 模型：`models/dino_simple_cnn_decoder.py`
   - 訓練：`scripts/train_dino_simple_cnn.py`
   - 架構：Conv2d 降維 + Upsample 逐步上採樣
   - 特點：簡單、參數少、訓練快

3. **方案2：FPN-like Decoder（多尺度特徵融合）**
   - 模型：`models/dino_fpn_decoder.py`
   - 訓練：`scripts/train_dino_fpn.py`
   - 架構：多尺度特徵提取（P5/P4/P3/P2）→ 融合 → 降維
   - 特點：多尺度資訊，適應不同大小的物體

4. **方案6：混合方案（DINO + 原圖細節）**
   - 模型：`models/dino_hybrid_decoder.py`
   - 訓練：`scripts/train_dino_hybrid.py`
   - 架構：DINO features + 淺層 CNN features → Concat → 簡單 CNN Decoder
   - 特點：結合全局語義與像素細節，預期最佳表現

**共同設定**：
- DINO Feature Extractor：DINOv2 ViT-B/14，**凍結（frozen）**，只當 feature extractor
- 訓練參數：batch_size=2, epochs=10, learning_rate=1e-4, loss=BCE+Dice, optimizer=Adam
- 資料：使用相同的 `outputs/train_list.txt` 和 `outputs/val_list.txt`

**Step 4: 評估 DL baseline（使用固定 test_list.txt）**
- 腳本：`scripts/eval_dl_final.py`
- 輸入：`outputs/test_list.txt`（固定）、DL baseline checkpoint
- 輸出：追加到 `outputs/metrics_summary.csv`（method='DL'）

**Step 5: 評估 VLM → SAM pipeline（使用相同 test_list.txt）**
- 腳本：`scripts/eval_sam_final.py`
- 輸入：`outputs/test_list.txt`（固定）、SAM 2.0 模型
- 輸出：追加到 `outputs/metrics_summary.csv`（method='SAM'）

**Step 6: 評估 CLIPSeg（使用相同 test_list.txt）**
- 腳本：`scripts/eval_clipseg_final.py`
- 輸入：`outputs/test_list.txt`（固定）、CLIPSeg 模型
- 輸出：追加到 `outputs/metrics_summary.csv`（method='CLIPSeg'）

**Step 7: 評估 DINO → DL pipeline（四種方案，使用相同 test_list.txt）**
- 腳本：`scripts/eval_dino_dl_final.py`
- 輸入：`outputs/test_list.txt`（固定）、四個方案的 best checkpoints
- 輸出：追加到 `outputs/metrics_summary.csv`（method='DINO-MLP', 'DINO-CNN', 'DINO-FPN', 'DINO-Hybrid'）

**Step 8: 產出統一 metrics_summary.csv**
- 腳本：`scripts/aggregate_final_metrics.py`
- 輸出：`outputs/metrics_summary.csv`
- 格式：method, camera_id, num_images, IoU_mean, FP_mean, FN_mean, FP_rate_nosky

---

## 三、DINO → DL 四種方案架構說明

### 方案4：SegFormer-like Decoder（MLP-based）

```
DINO features (37×37×768)
    ↓
[ MLP Mixer / Linear Layers ]
    ├─→ Linear(768→256) → 37×37×256
    ├─→ LayerNorm + GELU
    └─→ Linear(256→256) → 37×37×256
    ↓
[ 上採樣模組 ]
    ├─→ ConvTranspose2d(256→128, stride=2) → 74×74×128
    ├─→ ConvTranspose2d(128→64, stride=2) → 148×148×64
    └─→ ConvTranspose2d(64→1, stride=2) → 296×296×1
    ↓
Resize → 256×256×1
```

**特點**：輕量（MLP 參數少）、適合 Transformer 特徵、訓練快

---

### 方案1：簡單 CNN Decoder

```
DINO features (37×37×768)
    ↓
Conv2d(768→256, kernel=1) → 37×37×256
    ↓
Upsample (×2) → 74×74×256
    ↓
Conv2d(256→128, kernel=3, padding=1) → 74×74×128
    ↓
Upsample (×2) → 148×148×128
    ↓
Conv2d(128→64, kernel=3, padding=1) → 148×148×64
    ↓
Upsample (×2) → 296×296×64
    ↓
Conv2d(64→1, kernel=3, padding=1) → 296×296×1
    ↓
Resize → 256×256×1
```

**特點**：簡單、參數少、訓練快

---

### 方案2：FPN-like Decoder（多尺度特徵融合）

```
DINO features (37×37×768)
    ↓
Conv2d(768→256, kernel=1) → 37×37×256
    ↓
[ 多尺度特徵提取 ]
    ├─→ P5: 37×37×256  (原始尺寸)
    ├─→ Upsample(×2) → P4: 74×74×256
    ├─→ Upsample(×2) → P3: 148×148×256
    └─→ Upsample(×2) → P2: 296×296×256
    ↓
[ 融合多尺度特徵 ]
把所有尺度上採樣到最大尺寸（296×296），然後 concat
    ↓
Concat → 296×296×1024
    ↓
[ 融合與降維 ]
Conv2d(1024→256) → Conv2d(256→128) → Conv2d(128→1)
    ↓
Resize → 256×256×1
```

**特點**：多尺度資訊，適應不同大小的物體，比方案一表現更好

---

### 方案6：混合方案（DINO + 原圖細節）

```
[ 影像 256×256 ]
    ↓
    ├─→ DINO → Patch features (37×37×768)  [ 全局語義 ]
    │
    └─→ 淺層 CNN → Features (256×256×64)  [ 像素細節 ]
         ↓
    [ DINO features 上採樣到 256×256×256 ]
         ↓
    [ Concat(DINO features, CNN features) ] → 256×256×320
         ↓
    [ 簡單 CNN Decoder ]
         ├─→ Conv(320→128) → 256×256×128
         ├─→ Conv(128→64) → 256×256×64
         └─→ Conv(64→1) → 256×256×1
```

**特點**：結合 DINO 的全局語義與 CNN 的像素細節，預期最佳表現

---

## 四、重要原則

1. **test_list.txt 鎖死**：產出後不得再修改；所有方法必須用同一份 test_list.txt
2. **Deterministic split**：依 filename 排序，不隨機，確保可重現
3. **公平比較**：DL / SAM / CLIPSeg / DINO+DL（四種）都用相同 test_list.txt，指標計算方式一致
4. **No-sky 特殊處理**：no-sky camera 只看 FP_rate_nosky（pred sky area > 0 的比例），IoU/FN 不適用
5. **DINO 凍結**：所有 DINO → DL 方案中，DINO feature extractor 都凍結（frozen），只訓練 decoder
6. **訓練參數統一**：所有方法使用相同的訓練參數（batch_size=2, epochs=10, learning_rate=1e-4）

---

## 五、執行狀態

**更新時間**：2026-02-12 23:56

### Step 1: 建立 metadata.csv ✅ **已完成**

- **狀態**：✅ 已完成
- **輸出檔案**：`outputs/metadata_all_images.csv`
- **統計**：3170 張影像，9 個 camera
- **完成時間**：2026-02-12（之前已執行）

### Step 2: 建立 in-domain splits ✅ **已完成**

- **狀態**：✅ 已完成
- **輸出檔案**：
  - `outputs/train_list.txt` - 2535 張
  - `outputs/val_list.txt` - 315 張
  - `outputs/test_list.txt` - 320 張（**已鎖死**）
  - `outputs/in_domain_splits_summary.json` - 統計資訊
- **完成時間**：2026-02-12 23:53

### Step 3: 訓練階段 🔄 **執行中**

#### Step 3.1: 訓練 DL baseline (U-Net) 🔄 **執行中**

- **狀態**：🔄 執行中（使用 CPU）
- **腳本**：`scripts/train_in_domain.py`
- **開始時間**：2026-02-12 23:55:48
- **當前進度**：Epoch 1/10，約 13/1268 batches（約 1%）
- **訓練參數**：
  - Batch size: 2
  - Epochs: 10
  - Learning rate: 1e-4
  - 設備: CPU（強制）
  - 混合精度: 關閉
- **輸出目錄**：`outputs/train_in_domain_20260212_235623/`
- **預估完成時間**：
  - 每個 batch 約 12-15 秒（CPU 模式）
  - 1268 batches × 10 epochs = 12,680 batches
  - 預估總時間：**約 42-53 小時**（CPU 模式較慢）
  - 預估完成：**約 2026-02-14 18:00-05:00**

#### Step 3.2: 訓練 DINO → DL（四種方案） ⏳ **等待中**

**批次執行腳本**：`scripts/run_all_training.py`
- **功能**：依序執行所有訓練，即使某個失敗也會繼續
- **強制使用 CPU**：避免 GPU 問題
- **記錄檔案**：`outputs/training_batch_log.txt`

**執行順序**：

1. **方案4：DINO-MLP** ⏳ **等待中**
   - 腳本：`scripts/train_dino_mlp.py`
   - 狀態：等待 Step 3.1 完成
   - 預估時間：約 42-53 小時（CPU 模式）

2. **方案1：DINO-CNN** ⏳ **等待中**
   - 腳本：`scripts/train_dino_simple_cnn.py`
   - 狀態：等待前一個完成
   - 預估時間：約 42-53 小時（CPU 模式）

3. **方案2：DINO-FPN** ⏳ **等待中**
   - 腳本：`scripts/train_dino_fpn.py`
   - 狀態：等待前一個完成
   - 預估時間：約 42-53 小時（CPU 模式）

4. **方案6：DINO-Hybrid** ⏳ **等待中**
   - 腳本：`scripts/train_dino_hybrid.py`
   - 狀態：等待前一個完成
   - 預估時間：約 42-53 小時（CPU 模式）

**總預估時間**：
- 5 個訓練 × 42-53 小時 = **約 210-265 小時**（約 8.75-11 天）
- **注意**：CPU 模式訓練非常慢，建議：
  - 讓訓練在背景執行
  - 明天檢查進度
  - 考慮是否需要調整參數（例如減少 epochs 或 batch size）

### Step 4-7: 評估階段 ⏳ **等待中**

- **狀態**：⏳ 等待 Step 3 完成
- **待執行腳本**：
  - `scripts/eval_dl_final.py` - 評估 DL baseline
  - `scripts/eval_sam_final.py` - 評估 SAM
  - `scripts/eval_clipseg_final.py` - 評估 CLIPSeg
  - `scripts/eval_dino_dl_final.py` - 評估 DINO → DL（四種方案）
- **預估時間**：每個評估約 30 分鐘 - 2 小時

### Step 8: 產出統一 metrics_summary.csv ⏳ **等待中**

- **狀態**：⏳ 等待 Step 4-7 完成
- **腳本**：`scripts/aggregate_final_metrics.py`
- **預估時間**：約 5 分鐘

---

## 六、已建立的腳本與模型

### 資料準備腳本 ✅
- ✅ `scripts/create_metadata_for_in_domain_split.py`
- ✅ `scripts/create_in_domain_splits.py`

### 訓練腳本 ✅
- ✅ `scripts/train_in_domain.py` - DL baseline
- ✅ `scripts/train_dino_mlp.py` - DINO-MLP
- ✅ `scripts/train_dino_simple_cnn.py` - DINO-CNN
- ✅ `scripts/train_dino_fpn.py` - DINO-FPN
- ✅ `scripts/train_dino_hybrid.py` - DINO-Hybrid
- ✅ `scripts/run_all_training.py` - 批次執行腳本（依序執行，強制 CPU）

### 評估腳本 ✅
- ✅ `scripts/eval_dl_final.py` - 評估 DL baseline
- ✅ `scripts/eval_sam_final.py` - 評估 SAM
- ✅ `scripts/eval_clipseg_final.py` - 評估 CLIPSeg
- ✅ `scripts/eval_dino_dl_final.py` - 評估 DINO → DL（四種方案）
- ✅ `scripts/aggregate_final_metrics.py` - 合併 metrics

### 模型定義 ✅
- ✅ `models/dino_mlp_decoder.py` - DINO-MLP Decoder
- ✅ `models/dino_simple_cnn_decoder.py` - DINO-CNN Decoder
- ✅ `models/dino_fpn_decoder.py` - DINO-FPN Decoder
- ✅ `models/dino_hybrid_decoder.py` - DINO-Hybrid Decoder

---

## 七、重要注意事項

1. **CPU 訓練模式**：所有訓練強制使用 CPU，避免 GPU 問題
2. **訓練時間**：CPU 模式訓練非常慢，每個模型約需 42-53 小時
3. **批次執行**：使用 `scripts/run_all_training.py` 依序執行所有訓練
4. **錯誤處理**：即使某個訓練失敗，其他訓練仍會繼續
5. **記錄檔案**：`outputs/training_batch_log.txt` 記錄所有訓練的執行狀況
6. **test_list.txt 已鎖死**：不得再修改，所有評估必須使用同一份

---

## 八、參考資料

- **Plan 文件**：`C:\Users\yiching\.cursor\plans\sky_segmentation_final_evaluation_pipeline_7b22261e.plan.md`
- **架構討論日誌**：`notes/2026-02-11-dino-dl-architecture-discussion.md`
- **DINO prompt 腳本**：`scripts/dino_sky_prompt.py`（參考 DINO 載入方式）
- **訓練腳本基底**：`train_multi_camera.py`（參考訓練流程）

---

## 九、執行記錄

### 2026-02-12 23:56 更新

**已完成**：
- ✅ Step 1: 建立 metadata.csv（3170 張影像）
- ✅ Step 2: 建立 in-domain splits（train: 2535, val: 315, test: 320）
- ✅ 所有訓練與評估腳本已建立
- ✅ 批次執行腳本已建立並啟動

**執行中**：
- 🔄 Step 3.1: DL baseline 訓練中（Epoch 1/10，約 1% 進度）
- ⏳ Step 3.2: DINO → DL 四種方案等待中

**預估時間**：
- 當前訓練（DL baseline）：約 42-53 小時
- 所有訓練完成：約 210-265 小時（8.75-11 天）

**記錄檔案**：
- `outputs/training_batch_log.txt` - 批次執行記錄
- `outputs/train_in_domain_20260212_235623/` - DL baseline 訓練輸出

---

*記錄日期：2026-02-12*
*最後更新：2026-02-12 23:56*
