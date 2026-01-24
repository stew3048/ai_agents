# Test 評估流程（Multi-Camera 訓練完成後）

> 訓練完成後，用 **best checkpoint** 在 **Test set** 上評估，並產出所有報表與 overlay。  
> **之後要做 Test 時，只要說「做 Test 評估」或「照 eval_test_procedure 跑 Test」**，即依本文件執行，不需重新給指令。

---

## 0. 執行步驟（一頁摘要）

| 步驟 | 動作 |
|------|------|
| 1 | 執行 `python eval_multi_camera_test.py`（或加 `--checkpoint` 指定 best.pth） |
| 2 | 腳本會自動：算整體指標（含 FP/FN）、輸出 overlay 最好/最差各 5、印出最差 5 的 FP/FN |
| 3 | （可選）依 §4 寫入當次 run 的泛化測試筆記 |

---

## 1. 腳本與指令

| 項目 | 說明 |
|------|------|
| **腳本** | `eval_multi_camera_test.py` |
| **預設指令** | `python eval_multi_camera_test.py` |
| **指定 checkpoint** | `python eval_multi_camera_test.py --checkpoint outputs/train_multi_camera_XXX/checkpoints/best.pth` |
| **Checkpoint 來源** | 未指定時，自動取最新的 `outputs/train_multi_camera_*/checkpoints/best.pth` |
| **Splits** | 預設 `outputs/multi_camera_splits.json`，可用 `--splits` 覆寫 |
| **Image size** | 預設 `160 160`（與訓練一致），可用 `--image_size 160 160` 覆寫 |

---

## 2. Test 要做的事（完整清單）

腳本 `eval_multi_camera_test.py` 會**一次完成**下列所有項目；之後不需再個別下指令。

- **整體 Test 指標**：IoU、Dice、Pixel Acc、**FP Rate、FN Rate**、Loss（腳本自動）  
- **Overlay**：**最好 5 張**、**最差 5 張** 存到 `{run_dir}/test_overlays/`，檔名含 IoU（腳本自動）  
- **終端印出**：整體指標、最好 5 的 IoU、**最差 5 的 IoU + FP Rate + FN Rate**（腳本自動）  
- **筆記**：依 §4 寫泛化測試筆記—前提、Val/Test 結果、失敗情境、輸出檔與指令（手動）  

### 2.1 整體 Test 指標

| 指標 | 說明 |
|------|------|
| Test IoU | 整體 IoU |
| Test Dice | 整體 Dice |
| Test Pixel Acc | 整體 Pixel Accuracy |
| **Test FP Rate** | **非天空**像素中，被誤判為天空的比例（FP / (TN+ε)） |
| **Test FN Rate** | **天空**像素中，被漏檢的比例（FN / (TP+ε)） |
| Test Loss | BCEWithLogitsLoss |

### 2.2 Overlay 輸出

| 項目 | 路徑與命名 |
|------|------------|
| **最好 5 張** | `{run_dir}/test_overlays/best_1_iou_0.xxxx.png` ~ `best_5_iou_0.xxxx.png` |
| **最差 5 張** | `{run_dir}/test_overlays/worst_1_iou_0.xxxx.png` ~ `worst_5_iou_0.xxxx.png` |

- Overlay 配色：**藍 = GT**，**紅 = 預測**（與 val overlay 相同）。
- 依每張圖 **IoU** 排序，取最高 5 張與最低 5 張。

### 2.3 最差 5 張的 FP / FN

- 終端會印出最差 5 張的 **IoU、FP Rate、FN Rate**，便於分析失敗型態（漏檢 vs 誤判）。

---

## 3. 輸出位置

- **Overlay 目錄**：`outputs/train_multi_camera_{timestamp}/test_overlays/`
- 與該 run 的 `checkpoints/best.pth`、`val_overlays/`、`training_log.csv` 同一 run 目錄下。

---

## 4. 日誌／筆記要記錄的內容

做完 Test 評估後，建議在 **跨 camera 泛化測試** 的 note（如 `notes/2026-01-24_multi-camera_cross-camera_generalization.md`）中寫入以下全部項目。

### 4.1 前提

- **Train / Val / Test 張數**：Train、Sanity、Val、Test 各幾張，以及 Val / Test 來自哪個 camera。
- **各 Camera 的白天／夜晚比例**：從 `outputs/camera_inventory.csv` 取  
  `night_ratio`、`mean_brightness`，表格列出：Camera、張數、夜晚比例、白天／傍晚／黃昏等、平均亮度。

### 4.2 驗證（Val）結果

- 各 Epoch 的：Train Loss/IoU、Sanity IoU、Val Loss、**Val IoU、Val Dice、Val Pixel Acc**。
- 最佳 Val：Epoch、Val IoU、Val Dice、Val Pixel Acc、Val Loss。
- Val Overlay 路徑：`val_overlays/`，最好 5、最差 5 的檔名範例。  
- （若日後有算）Val 的 FP Rate、FN Rate。

### 4.3 測試（Test）結果

- **整體**：Test IoU、Dice、Pixel Acc、**FP Rate、FN Rate**、Test Loss。
- **Val vs Test 比較表**：IoU、Dice、Pixel Acc、FP Rate、FN Rate 的落差。
- **最好 5 張**：每張 IoU。
- **最差 5 張**：每張 IoU、**FP Rate、FN Rate**。
- Test Overlay 路徑：`test_overlays/`，最好 5、最差 5 的檔名範例。

### 4.4 Val 與 Test 的「訓練不好的情境」

- **Val**：依 camera 的 night_ratio、亮度，說明易失敗情境（如：灰雲、低對比、黃昏／傍晚、多雲／霧）；最差 5 張 IoU 範圍。
- **Test**：  
  - **Domain shift**：Test camera 的夜晚比例、亮度與 Train/Val 的差異。  
  - **表現不好的情境**：夜晚、傍晚／黃昏、厚重灰雲、霧、低對比等。  
  - **極端失敗**：最差一張的 IoU、FN（及 FP），推測情境（如夜晚、厚重灰雲／霧）。

### 4.5 輸出檔與指令

- 訓練日誌、best.pth、val_overlays、test_overlays 之路徑。
- 重新跑 Test 評估的指令：`python eval_multi_camera_test.py`（及可選的 `--checkpoint`）。

---

## 5. 參考範例

- **完整範例**：`notes/2026-01-24_multi-camera_cross-camera_generalization.md`  
  含：前提、Val/Test 結果、FP/FN、overlay、失敗情境、domain shift、極端失敗、輸出檔與指令。

---

## 6. 與訓練流程的關係

| 階段 | 腳本 | 產出 |
|------|------|------|
| 訓練 | `train_multi_camera.py` | `best.pth`、`latest.pth`、`training_log.csv`、**val_overlays/**（最好/最差各 5） |
| **Test 評估** | **`eval_multi_camera_test.py`** | **Test 整體指標（含 FP/FN）**、**test_overlays/**（最好/最差各 5）、最差 5 的 FP/FN |

訓練結束後，跑一次 `eval_multi_camera_test.py` 即可完成上述所有 Test 項目；再依 §4 寫入當次 run 的泛化測試筆記。

---

*建立日期：2026-01-24*
