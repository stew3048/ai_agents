# 評估結論、改善方向、Val vs Test night 分析差別與用途

**日期**：2026-01-25  
**Checkpoint**：`outputs/train_multi_camera_20260124_004036/checkpoints/best.pth` (Epoch 3, Val IoU 0.8616)

---

## 一、本次評估結果摘要

### Val（camera 9483，100 張）

| 組別      | n   | IoU    | Dice   | PixelAcc | FP rate | FN rate |
|-----------|-----|--------|--------|----------|---------|---------|
| overall   | 100 | 0.8625 | 0.9262 | 0.9355   | 0.0566  | 0.0746  |
| night-ish | 20  | 0.7451 | 0.8539 | 0.8885   | 0.0005  | **0.2545** |
| non-night | 80  | 0.8894 | 0.9415 | 0.9473   | 0.0706  | 0.0296  |

- **luma 閾值**：T=0.10（linear），night-ish=20，non-night=80  
- **Overlay**：`outputs/val_9483_errors/night-ish/`、`outputs/val_9483_errors/non-night/`（各最差 10 張）

### Test（camera 10870，94 張，unseen camera）

| 組別 | n  | IoU (mean±std) | Dice (mean±std) | PixelAcc (mean±std) | FP rate (mean±std) | FN rate (mean±std) |
|------|----|----------------|-----------------|---------------------|--------------------|--------------------|
| night | 58 | **0.6299 ± 0.0755** | 0.7704 ± 0.0562 | 0.8882 ± 0.0282 | 0.0477 ± 0.0224 | **0.2901 ± 0.0604** |
| day  | 36 | 0.8135 ± 0.1466 | 0.8871 ± 0.1302 | 0.9382 ± 0.0501 | 0.0742 ± 0.0529 | 0.0274 ± 0.1477 |

- **luma 閾值**：T=0.20（linear），night=58，day=36  
- **Overlay**：`outputs/test_10870_errors/night/`、`outputs/test_10870_errors/day/`（各最差 10 張）

---

## 二、結論

1. **夜間／低亮度明顯較差**
   - **Val**：night-ish IoU 0.7451 vs non-night 0.8894（差約 0.14）；night-ish **FN rate 0.25**，漏檢嚴重。
   - **Test**：night IoU 0.63±0.08 vs day 0.81±0.15；night **FN rate 0.29±0.06**，漏檢更嚴重。
   - 低亮度下模型易**漏檢天空**（藍色 FN 多），推測與訓練集 night 樣本少、低亮度特徵不足有關。

2. **跨 camera 泛化有落差**
   - Val（9483）overall IoU 0.86，Test（10870）在 night 組掉到 0.63，顯示**未見過的 10870 + 夜間**是雙重弱點。
   - Day 組 Test 0.81±0.15，與 Val non-night 0.89 仍有距離，10870 的場景／鏡頭特性與 9483 有差異。

3. **Overlay 觀察（白天、晚上都有）**
   - **夜間最差 case**：藍色（FN）多，天空邊緣、薄雲、低對比處漏檢；少數紅色（FP）在建築／地景邊界。
   - **白天最差 case**：除邊緣與複雜遮擋外，Test day 有極端低 IoU（如 0.1、0.5），可能是 10870 特有場景（逆光、強陰影、特殊構圖），值得單獨排查。

---

## 三、改善方向

1. **資料面**
   - **增加夜間／低亮度樣本**：從更多 camera 納入 mean_luma 低、清晨／黃昏／夜間資料，平衡 night 比例。
   - **亮度／對比 augmentation**：若暫不擴充資料，可做 gamma、對比、亮度下調，讓模型多見「變暗」版本。
   - **10870 的少量加入**：若允許，可從 10870 取少許（如 10–20 張）加入 train，觀察 Test night 是否提升（需注意過擬 10870）。

2. **訓練與選模型**
   - **納入 night 指標選 checkpoint**：例如 `0.5*val_iou + 0.5*night_ish_iou`，或設 night_ish_iou 下限，避免只優化 overall。
   - **Loss 加權**：對 luma&lt;T 的樣本提高 loss 權重，強制多學夜間。
   - **Early stopping**：可同時看 val overall 與 val night-ish，避免 overall 升、night 崩。

3. **推理與後處理（若可行）**
   - **亮度自適應**：推論時若偵測到 mean_luma&lt;T，可考慮輕微 pre-processing（如小幅提亮、對比）或使用專用 head／threshold，需實驗驗證。
   - **集成**：若有「偏向夜間」的模型，可依 luma 切換或加權融合。

4. **分析與監控**
   - **定期跑 Val / Test 的 night 分解**：每次換資料或改架構後跑 `eval_val_only`、`eval_test_night_day`，追蹤 night 的 IoU、FN。
   - **Overlay 覆蓋**：白天、晚上最差 10 張都留 overlay，便於發現新的 failure mode（如 10870 的特定場景）。

---

## 四、Val 做 night/non-night 與 Test 做 night/day 的差別與用途

### 1. Val（9483）night-ish / non-night

| 項目 | 說明 |
|------|------|
| **資料** | 與訓練同一 camera（9483），僅 split 不同；night 比例通常不高（本次 20/100）。 |
| **用途** | **訓練期監控、選 checkpoint、early stopping**；發現「val 是否偏易、是否缺 night」的偏差。 |
| **產出** | overall / night-ish / non-night 的 IoU、Dice、PixelAcc、FP、FN；`val_9483_grouped_metrics.csv`；可選 **overlay（各組最差 10）** 做案例檢查。 |
| **時機** | 可每 epoch 或定期離線跑 `eval_val_only`；不一定要每輪都產 overlay，依需要。 |
| **解讀** | 若 night-ish 遠差於 non-night、且 night-ish 張數少，代表 val 偏亮、對夜間參考性有限，需從資料或指標上補強。 |

### 2. Test（10870）night / day

| 項目 | 說明 |
|------|------|
| **資料** | **Unseen camera（10870）**，與 train/val 的 9483 等不同；10870 的 night 比例常較高（本次 58/94）。 |
| **用途** | **最終泛化評估、寫報告／論文**；評估跨 camera、跨亮度（日夜）的真實表現；分析 **domain shift、night 弱點**。 |
| **產出** | 各組 **mean±std**（IoU、Dice、PixelAcc、FP、FN）；`test_10870_grouped_metrics.csv`；**各組最差 10 張 overlay**（day、night 都要）做案例與 failure 分析。 |
| **時機** | 訓練結束、換資料或架構後；不宜用於挑 checkpoint，避免 test 洩漏。 |
| **解讀** | night 明顯差、FN 高 → 夜間與跨 camera 是瓶頸；day 有極端低 IoU → 可能為 10870 特殊場景，需 overlay 與個案排查。 |

### 3. 兩者對照

| | Val night-ish / non-night | Test night / day |
|---|---------------------------|------------------|
| **角色** | 訓練流程內部的**監控與選模** | 對外的**泛化與報告** |
| **Camera** | 與 train 同源（9483） | 未見過（10870） |
| **Overlay** | 可選，偏除錯與 sanity | **建議必做**，尤其 day / night 最差 case |
| **指標形式** | 各組 pooled 單值即可 | 各組 **mean±std** 較合適（張數、分佈差異大） |
| **T 與組名** | T=0.10，night-ish / non-night | T=0.20，night / day（可依需求調成一致） |

**簡言之**：  
- **Val 的 night-ish / non-night**：幫你**選模型、調參、發現 val 偏亮或 night 不足**，overlay 輔助檢查。  
- **Test 的 night / day**：用來**報告泛化、分析跨 camera 與日夜弱點**，overlay（白天、晚上）是說明與改進的依據。

---

## 五、Overlay 與輸出路徑一覽

- **Val（9483）**  
  - `outputs/val_9483_grouped_metrics.csv`  
  - `outputs/val_9483_errors/night-ish/`、`outputs/val_9483_errors/non-night/`（各最差 10 張）

- **Test（10870）**  
  - `outputs/test_10870_grouped_metrics.csv`  
  - `outputs/test_10870_errors/night/`、`outputs/test_10870_errors/day/`（各最差 10 張）

Overlay 顏色：**藍**=漏檢（FN），**紅**=誤判（FP），**紫/粉紫**=正確（TP）。
