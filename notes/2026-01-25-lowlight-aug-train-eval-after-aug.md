# 2026-01-25 低光 Augmentation 訓練與 Test 10870 改前／改後評估

> 在 train 加入低光／低對比 augmentation（`_TRAIN_AUG_LOWLIGHT`），以 best checkpoint 跑 `eval_test_after_aug.py`，產出 Test 10870 的 night / day / dusk 分組與**改前 vs 改後**對照表。

---

## 0. 前提：資料規模與各 Camera 白天／夜晚比例

### 0.1 Train / Val / Test 張數

沿用 `outputs/multi_camera_splits.json` 與 `outputs/camera_inventory.csv`（未重新跑 camera_inventory）。

| Split | 張數 | 說明 |
|-------|------|------|
| **Train** | 208 | 10066(45) + 9291(73) + 9112(90)，各 camera 前 90% |
| **Sanity** | 24 | 同上三台，各 camera 後 10% |
| **Val** | 100 | Camera **9483** 全部（unseen） |
| **Test** | 94 | Camera **10870** 全部（unseen） |

### 0.2 各 Camera 的白天／夜晚比例（camera_inventory）

**注意**：`camera_inventory.csv` 中的 `night_ratio` 原本使用錯誤的計算方式（sRGB 空間，閾值 50），已於 2026-01-26 修正為與訓練/評估一致的計算方式（linear RGB luma，閾值 0.20）。以下為修正後的數值（需重新執行 `camera_inventory.py` 以更新）：

| Camera | 張數 | 夜晚比例 (night_ratio) | 平均亮度 (mean_brightness) |
|--------|------|------------------------|----------------------------|
| 10066 | 50 | ~2% | ~137 |
| 9291 | 82 | ~53% | ~74 |
| 9112 | 100 | ~44% | ~78 |
| **9483** (Val) | 100 | ~16% | ~130 |
| **10870** (Test) | 94 | **~48%** | **~72** |

**計算方式**：
- sRGB → linear RGB 轉換
- luma = 0.2126*R_lin + 0.7152*G_lin + 0.0722*B_lin (ITU-R BT.709)
- night 判斷：luma < 0.20（與訓練/評估一致）

Test 的 10870 夜晚比例最高、亮度最低，與改前日誌結論一致：night FN 高、需低光相關改善。

---

## 1. 設定摘要

| 項目 | 內容 |
|------|------|
| **Splits** | `outputs/multi_camera_splits.json` |
| **Train cameras** | 10066, 9291, 9112 |
| **Val camera** | 9483 |
| **Test camera** | 10870 |

### 訓練參數

| 參數 | 值 |
|------|-----|
| Epochs | 6 |
| Batch size | 2 |
| Image size | 160×160 |
| Learning rate | 1e-4 |
| Loss | BCEWithLogitsLoss |
| Optimizer | Adam |
| AMP | on（GPU） |
| **Train transform** | **transform=True**：低光 / 低對比 aug（見下） |

### Train Augmentation（`utils/dataset._TRAIN_AUG_LOWLIGHT`）

僅在 **train、transform=True** 時套用；val / sanity / test 皆 **transform=False**。

| 項目 | 設定 |
|------|------|
| RandomBrightnessContrast | brightness_limit=(-0.45, 0.10), contrast_limit=(-0.45, 0.20), p=0.75 |
| RandomGamma | gamma_limit=(60, 140)（即 gamma 0.6~1.4），p=0.55 |
| GaussianBlur | blur_limit=(3, 5), p=0.25 |
| GaussNoise | var_limit=(5.0, 25.0), p=0.20 |

- **修正**：albumentations `RandomGamma` 的 `gamma_limit` 需 int ≥1，故由 `(0.6, 1.4)` 改為 `(60, 140)`。

### Checkpoint 與評估腳本

- **Run 目錄**：`outputs/train_multi_camera_20260125_222612/`
- **Best checkpoint**：`checkpoints/best.pth`（Epoch 5，依 **Val IoU** 選取）
- **Test 評估**：`scripts/eval_test_after_aug.py`（T=0.20 night/day，dusk 為 p25≤luma<p50）

---

## 2. 驗證（Val）結果

Val 來自 **9483**，`transform=False`。

### 2.1 各 Epoch Val 指標（含 Val Night IoU / FN）

**說明**：本 run 當時腳本只計算 **Val Night**（luma&lt;0.10）IoU/FN，**未計算 Val Day**（luma≥0.10）IoU/FN，故下表與 2.2 僅有 night。`train_multi_camera.py` 已改為 `evaluate_val_night_day_metrics`，之後 runs 會一併輸出 **Val Day IoU / Val Day FN** 並寫入 `training_log.csv`。

| Epoch | Train Loss | Train IoU | Sanity IoU | Val IoU | Val Dice | Val Pixel Acc | Val FP | Val FN | **Val Night IoU** | **Val Night FN** |
|-------|------------|-----------|------------|---------|----------|---------------|--------|--------|-------------------|------------------|
| 1 | 0.4657 | 0.6409 | 0.6734 | 0.7584 | 0.8550 | 0.8769 | 0.1283 | 0.1164 | 0.7623 | 0.1664 |
| 2 | 0.3125 | 0.7699 | 0.6691 | 0.5810 | 0.7150 | 0.8057 | 0.0590 | 0.3686 | 0.8370 | 0.0000 |
| 3 | 0.2461 | 0.8240 | 0.7869 | 0.7024 | 0.8240 | 0.8692 | 0.0063 | 0.2913 | 0.7463 | 0.2460 |
| 4 | 0.2118 | 0.8417 | 0.7272 | 0.8617 | 0.9252 | 0.9293 | 0.1200 | 0.0072 | 0.8953 | 0.0100 |
| **5** | **0.1816** | **0.8698** | 0.8486 | **0.8873** | **0.9400** | **0.9449** | 0.0850 | 0.0166 | **0.8710** | **0.0635** |
| 6 | 0.1655 | 0.8747 | 0.7442 | 0.8109 | 0.8951 | 0.8993 | 0.1609 | 0.0231 | 0.8201 | 0.1091 |

### 2.2 最佳 Val（Epoch 5）

| 指標 | 數值 |
|------|------|
| **Val IoU** | **0.8873** |
| Val Dice | 0.9400 |
| Val Pixel Acc | 0.9449 |
| Val Night IoU (luma<0.10) | 0.8710 |
| Val Night FN (luma<0.10) | 0.0635 |

### 2.3 Val Overlay

- **路徑**：`outputs/train_multi_camera_20260125_222612/val_overlays/`
- **val_best_worst_metrics.json**：best_5 / worst_5 的 source_id、iou、fp_rate、fn_rate、orig_path

### 2.4 為何 Val Night FN 會亂跳？與 aug 的關係

Val Night FN 在 6 個 epoch 的變化：0.17 → **0.00** → **0.25** → 0.01 → 0.06 → 0.11，起伏很大。可能原因如下。

1. **Val 本身沒有做 aug**  
   Val / sanity / test 皆 `transform=False`，只有 **train** 做低光 aug。因此 Val 影像沒被 augmentation 直接動到，Val night FN 的跳動**不是**「val 被 aug 污染」造成的。

2. **Aug 的間接影響：每個 epoch 的模型不同**  
   Train 用強低光 aug，每個 epoch 學到的決策邊界都不一樣；同一份 Val（固定、未 aug）在不同 epoch 的模型上，表現自然會變。  
   若某 epoch 在「多預測一點天空」的方向優化，Val night 的 FN 可能暫時變很低（甚至 0）；下一 epoch 偏「少預測一點」，FN 又拉高。所以 **aug 透過「改變每個 epoch 的模型」間接造成 Val 指標上下**。

3. **Val night-ish 樣本數少，pooled 指標容易大動**  
   Val 9483 共 100 張，以 T=0.10 切 night-ish 時約 **20 張**（見 `notes/2026-01-25_eval_conclusions_val_test_night_day.md`）。  
   在只有 ~20 張上算 pooled FN：只要少數幾張從「漏很多」變成「漏很少」（或反過來），FN rate 就會從 0.25 掉到 0.01 或從 0.01 飆到 0.25。  
   相較之下，**Val day（luma≥0.10）約 80 張**，樣本多，若日後一併輸出 **Val Day IoU / Val Day FN**，預期會比 Val Night 穩定；若只有 night 跳、day 相對平，可支持「主要是 night 樣本少＋模型在 night 上不穩定」的解釋。

4. **之後的 runs**  
   `train_multi_camera.py` 已改為輸出 **Val Day IoU / Val Day FN** 並寫入 CSV，之後可對照 night 與 day 的走勢，區分是「整體不穩」還是「只有 night 不穩」。

---

## 3. 測試（Test）結果：改前 vs 改後

Test 來自 **10870**，以 **best.pth（Epoch 5）** 跑 `eval_test_after_aug.py`。  
**改前** baseline：2026-01-25 無 aug（T=0.20 night/day），見 `notes/2026-01-25_eval_conclusions_val_test_night_day.md`。

### 3.1 改前 vs 改後對照表

| 指標 | 改前 | 改後 |
|------|------|------|
| **overall test IoU** | 0.6960 | 0.5979 |
| **night test IoU** | 0.6299 | 0.5668 |
| **night FN** | 0.2901 | **0.0289** |
| **day test IoU** | 0.8135 | 0.6602 |
| **dusk IoU** | — | 0.5709 |
| **dusk FN** | — | 0.0313 |

### 3.2 Test 分組（改後）

- **luma**：T=0.20，night=58，day=36；dusk（p25~p50）：n=23，p25=0.0091，p50=0.0687。

| 組別 | n | IoU (mean±std) | Dice (mean±std) | PixelAcc (mean±std) | FP rate (mean±std) | FN rate (mean±std) |
|------|---|----------------|-----------------|---------------------|--------------------|--------------------|
| night | 58 | 0.5668 ± 0.0294 | 0.7231 ± 0.0229 | 0.8027 ± 0.0204 | 0.2579 ± 0.0270 | **0.0289 ± 0.0108** |
| day | 36 | 0.6602 ± 0.0447 | 0.7945 ± 0.0326 | 0.8623 ± 0.0284 | 0.1858 ± 0.0423 | 0.0038 ± 0.0192 |
| dusk (pooled) | 23 | 0.5709 | — | — | 0.2506 | 0.0313 |

### 3.3 成功判準（eval 腳本內）

- **night FN 明顯下降（0.30→≤0.22）**：0.2901 → **0.0289** ✓ 達成
- **night IoU 上升（0.63→≥0.70）**：0.6299 → 0.5668 ✗ 未達成
- **day IoU 掉 ≤0.03 可接受**：0.8135 → 0.6602（約 -0.15）✗ 超出可接受範圍

### 3.4 Test 產出

- **CSV**：`outputs/test_10870_after_aug_metrics.csv`（filename, luma, group, iou, fp, fn）
- **Overlay**：
  - `outputs/test_10870_errors/night_after_aug/`：night 最差 IoU 10 + 最差 FN 10
  - `outputs/test_10870_errors/dusk_after_aug/`：dusk 最差 IoU 10

---

## 4. 觀察與分析

1. **night FN 大幅改善**  
   低光 aug 明顯壓低 Test 10870 的 **night FN（0.29→0.03）**，漏檢大幅減少，符合「多見變暗、低對比」的預期。

2. **night IoU 略降、day IoU 明顯下降**  
   - night IoU：0.63→0.57，未達「≥0.70」目標。  
   - day IoU：0.81→0.66，掉約 0.15，超出「≤0.03」可接受範圍。  
   推測：aug 過強或機率偏高，導致對「正常亮度／白天」的擬合變差；或選 best 只看 Val IoU，Val 以 9483 為主、與 10870 日間分布有差。

3. **overall test IoU 下降**  
   0.6960→0.5979，與 day 大掉、night 略降一致；night FN 的收益不足以抵銷 day 與整體 IoU 的損失。

4. **Val vs Test**  
   Val IoU 0.8873、Val Night FN 0.0635 均佳， Test 的 night FN 0.0289 也佳，但 **Test day 與 overall 崩得比 Val 多**，顯示 10870 日間或跨 camera 的泛化在目前 aug 下受損。

---

## 5. 下一步改善方向

1. **減弱或收斂 aug**  
   - 調低 `RandomBrightnessContrast`、`RandomGamma` 的強度與 `p`，或縮小 `brightness_limit` / `contrast_limit` 的負向範圍，減少對「正常／白天」的扭曲。  
   - 考慮 **只對 luma 偏低** 的 train 樣本做低光 aug（需在 Dataset 依 luma 條件套用）。

2. **選 checkpoint 時納入 day / overall**  
   - 若有 val 的 day-like 子集或 proxy，可一併納入選 best，避免僅優化 overall Val IoU 而犧牲 Test day。

3. **GaussNoise 參數**  
   - 若出現 `var_limit` 的 deprecation 或警告，改為 albumentations 新參數（如 `var_limit` 對應之新名稱）以消除 warning。

4. **dusk 與 night 的 overlay**  
   - 已產出 `night_after_aug`、`dusk_after_aug` 最差 case，可對照 failure mode 再微調 aug 或資料。

---

## 6. 輸出檔與指令

| 項目 | 路徑／指令 |
|------|------------|
| 訓練日誌 | `outputs/train_multi_camera_20260125_222612/training_log.csv` |
| Best checkpoint | `outputs/train_multi_camera_20260125_222612/checkpoints/best.pth` |
| Val overlays | `outputs/train_multi_camera_20260125_222612/val_overlays/` |
| Test 10870 分組 CSV | `outputs/test_10870_after_aug_metrics.csv` |
| Test 10870 overlay | `outputs/test_10870_errors/night_after_aug/`、`dusk_after_aug/` |
| 重新跑 Test 評估 | `python scripts/eval_test_after_aug.py --checkpoint outputs/train_multi_camera_20260125_222612/checkpoints/best.pth` |
| Camera inventory | `outputs/camera_inventory.csv`（沿用） |

---

## 7. 今日結論

> 加入低光／低對比 train augmentation 後，**Test 10870 的 night FN 由 0.29 降至 0.03**，漏檢明顯改善；但 **night IoU 略降、day IoU 與 overall IoU 明顯下降**，顯示目前 aug 強度或範圍對白天與整體泛化不利。下一步宜**減弱 aug、或僅對低 luma 樣本做低光 aug**，並在選 checkpoint 時兼顧 day / overall，以在維持 night FN 增益的同時，減少對 10870 日間的損傷。

---

*紀錄日期：2026-01-25*
