# 2026-01-25 Val / Test 依亮度分組分析（night-ish / non-night、night / day）

> 使用 `scripts/eval_val_only.py` 與 `scripts/eval_test_night_day.py`，以 **linear mean_luma** 將 Val (9483) 分為 night-ish / non-night、Test (10870) 分為 night / day，產出 overall 與分組指標、**各組最差 10 張 overlay**；並撰寫結論、改善方向、Val vs Test 差別與用途。  
> Checkpoint 沿用 2026-01-24 的 best.pth（Epoch 3）。

---

## 0. 前提

- **Val**：Camera **9483**，100 張（與 2026-01-24 相同 splits）。
- **Test**：Camera **10870**，94 張（unseen）。
- **Checkpoint**：`outputs/train_multi_camera_20260124_004036/checkpoints/best.pth`（Epoch 3, Val IoU 0.8616）。
- **分組規則**：**mean_luma（linear）**：image sRGB [0,1] 先 sRGB→linear，再 `0.2126*R + 0.7152*G + 0.0722*B` 取整張平均；night-ish / night = luma < T，non-night / day = luma ≥ T。T：Val 預設 **0.10**，Test 預設 **0.20**；若 night-ish / night < 10 張則改用 p25。  
  - Val：**T=0.10**，night-ish=20、non-night=80。  
  - Test：**T=0.20**，night=58、day=36。

---

## 1. 設定摘要

| 項目 | 內容 |
|------|------|
| **Val 腳本** | `scripts/eval_val_only.py`（含 night-ish / non-night 最差 10 overlay） |
| **Test 腳本** | `scripts/eval_test_night_day.py`（含 night / day 最差 10 overlay） |
| **Checkpoint** | `outputs/train_multi_camera_20260124_004036/checkpoints/best.pth` |
| **Val** | 9483，100 張；T=0.10，night-ish 20、non-night 80 |
| **Test** | 10870，94 張；T=0.20，night 58、day 36 |
| **mean_luma** | sRGB→linear 後再算 luma（避免 sRGB 直接算高估暗部） |

---

## 2. 驗證集（Val）依亮度分組結果

### 2.1 overall / night-ish / non-night 摘要（T=0.10，night-ish=20、non-night=80）

|            | IoU    | Dice   | PixelAcc | FP rate | FN rate |
|------------|--------|--------|----------|---------|---------|
| **overall**   | 0.8625 | 0.9262 | 0.9355   | 0.0566  | 0.0746  |
| **night-ish** | 0.7451 | 0.8539 | 0.8885   | 0.0005  | **0.2545** |
| **non-night** | 0.8894 | 0.9415 | 0.9473   | 0.0706  | 0.0296  |

### 2.2 明細與 Overlay

- **明細**：`outputs/val_9483_grouped_metrics.csv`（`filename`, `luma`, `group`, `iou`, `fp`, `fn`）
- **Overlay（各組最差 10 張）**：`outputs/val_9483_errors/night-ish/`、`outputs/val_9483_errors/non-night/`  
  - 顏色：藍=FN（漏檢）、紅=FP（誤判）、紫/粉紫=TP

---

## 3. 測試（Test）— eval_test_night_day（camera 10870）

本日已跑 `scripts/eval_test_night_day.py`，以 T=0.20 將 Test 分為 **night（58 張）**、**day（36 張）**。

### 3.1 各組 mean ± std

| 組別 | n  | IoU (mean±std) | Dice (mean±std) | PixelAcc (mean±std) | FP rate (mean±std) | FN rate (mean±std) |
|------|----|----------------|-----------------|---------------------|--------------------|--------------------|
| **night** | 58 | **0.6299 ± 0.0755** | 0.7704 ± 0.0562 | 0.8882 ± 0.0282 | 0.0477 ± 0.0224 | **0.2901 ± 0.0604** |
| **day**   | 36 | 0.8135 ± 0.1466 | 0.8871 ± 0.1302 | 0.9382 ± 0.0501 | 0.0742 ± 0.0529 | 0.0274 ± 0.1477 |

### 3.2 明細與 Overlay

- **明細**：`outputs/test_10870_grouped_metrics.csv`
- **Overlay（各組最差 10 張）**：`outputs/test_10870_errors/night/`、`outputs/test_10870_errors/day/`  
  - 顏色：藍=FN、紅=FP、紫/粉紫=TP

---

## 4. 觀察與分析

### 4.1 Val：夜晚 (night-ish) 明顯比日間 (non-night) 差

|  | night-ish (20 張) | non-night (80 張) | 差距 |
|---|------------------|-------------------|------|
| **IoU** | 0.745 | 0.889 | **−0.14** |
| **Dice** | 0.854 | 0.942 | −0.09 |
| **PixelAcc** | 0.889 | 0.947 | −0.06 |
| **FN rate** | **0.255** | 0.030 | **約 8 倍** |
| **FP rate** | 0.0005 | 0.071 | 日間較高 |

- 整體表現：night-ish 的 IoU 掉一大截，Dice、PixelAcc 也較差。
- 錯誤型態：night-ish 的 **FN（天空漏檢）** 特別高，FP 極低；non-night 則是 FP、FN 都有，且 FP 較明顯。
- **Overlay**：`val_9483_errors/night-ish/`、`non-night/` 最差 10 張可見夜間藍色（FN）多、白天紅/紫並存。

### 4.2 Test（10870）：night 更差、跨 camera 泛化有落差

- **night** IoU **0.63±0.08**、FN **0.29±0.06**，較 Val night-ish 更差；**unseen 10870 + 夜間** 為雙重弱點。
- **day** IoU 0.81±0.15，與 Val non-night 0.89 有距離；少數極端低 IoU（如 0.1、0.5）可能為 10870 逆光、強陰影等特殊場景。
- **Overlay**：`test_10870_errors/night/`、`day/` 最差 10 張可供 failure 分析。

### 4.3 Overall 容易被日間主導

- Val 裡 **80% 是 non-night、20% 是 night-ish**；Test 的 10870 則 **night 佔 58/94**，night 比例高。
- 若只信 overall，會**高估**模型在 **低亮度／夜晚** 的表現。

### 4.4 夜晚：漏檢天空 (FN) 是主要問題

- 從 `val_9483_grouped_metrics.csv`、overlay：night-ish **fp≈0，fn 高**；低亮度下模型 **少預測天空**（under-segment）。
- 可能原因：訓練以日間為多，夜晚／低亮度樣本少；低亮度紋理、對比變弱，日間特徵不好遷移。

### 4.5 小結

- **Val / Test 皆顯示：夜間 IoU 明顯較差、FN 高；Test 在 unseen 10870 + night 尤甚。**
- 把 **night / non-night（Val）、night / day（Test）分開看** 是合理做法；overlay（白天、晚上）有助發現 failure mode。
- **結論、改善方向、Val vs Test 差別與用途** 詳見：`notes/2026-01-25_eval_conclusions_val_test_night_day.md`。

---

## 5. 下一步改善方向

- **加夜晚／低亮度資料**（或亮度／對比 augmentation），對壓低 night 的 FN 最有幫助。
- **Loss 或選 best 的指標**：納入 night 相關指標（如 `0.5*val_iou + 0.5*night_ish_iou`），或對 luma&lt;T 樣本提高 loss 權重。
- **定期跑 Val / Test 的 night 分解**（`eval_val_only`、`eval_test_night_day`），overlay 涵蓋白天、晚上最差 case。
- 完整條列見：`notes/2026-01-25_eval_conclusions_val_test_night_day.md` § 三、改善方向。

---

## 6. 輸出檔與指令

| 項目 | 路徑／指令 |
|------|------------|
| Val 分組明細 | `outputs/val_9483_grouped_metrics.csv` |
| Val overlay（night-ish / non-night 各最差 10） | `outputs/val_9483_errors/night-ish/`、`outputs/val_9483_errors/non-night/` |
| Test 分組明細 | `outputs/test_10870_grouped_metrics.csv` |
| Test overlay（night / day 各最差 10） | `outputs/test_10870_errors/night/`、`outputs/test_10870_errors/day/` |
| 結論、改善方向、Val vs Test 差別與用途 | `notes/2026-01-25_eval_conclusions_val_test_night_day.md` |
| 重新跑 Val 一次性分析 | `python scripts/eval_val_only.py` |
| 重新跑 Test 日夜分組分析 | `python scripts/eval_test_night_day.py` |
| 指定 checkpoint | `--checkpoint outputs/train_multi_camera_20260124_004036/checkpoints/best.pth`（兩腳本皆支援） |

---

## 7. 今日結論

> Val 與 Test 皆顯示 **夜間／低亮度 IoU 明顯較差、FN 高**（Val night-ish FN≈0.25、Test night FN≈0.29）；**unseen 10870 + 夜間** 為雙重弱點。本日已完成 **Val / Test 的 night–non-night、night–day 分組、各組最差 10 張 overlay**（白天、晚上都有），以及 **結論、改善方向、Val vs Test 差別與用途** 之整理（`notes/2026-01-25_eval_conclusions_val_test_night_day.md`）；宜把 night 分開觀察、增補低亮度資料或納入 night 相關指標做 best 挑選。

---

*紀錄日期：2026-01-25*
