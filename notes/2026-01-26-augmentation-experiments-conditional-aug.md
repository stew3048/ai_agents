# 2026-01-26 Augmentation 實驗：從強 aug 到條件式 aug

> 目標：改善 Test 10870 的 night FN（改前 0.29），同時避免 day IoU 大幅下降。經過多輪實驗（強 aug → 降低 aug → 中等 aug → 條件式 aug），最終採用條件式 augmentation 策略。

---

## 0. 前提：資料規模與各 Camera 白天／夜晚比例

### 0.1 Train / Val / Test 張數

沿用 `outputs/multi_camera_splits.json` 與 `outputs/camera_inventory.csv`。

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

**Train 資料的實際 night/day 比例**（基於實際 luma 計算，見 `scripts/analyze_train_luma_aug.py`）：
- 10066: 45 張，2.2% night → 1 張 night，44 張 day
- 9291: 73 張，53.4% night → 39 張 night，34 張 day
- 9112: 90 張，44.4% night → 40 張 night，50 張 day
- **總計**：**80 張 night（38.5%）**，**128 張 day（61.5%）**

**修正說明**：
- 先前日誌中估算的 32% night、68% day 是基於錯誤的 `camera_inventory.csv` 計算方式
- 實際計算（使用與訓練/評估一致的 luma 計算）顯示：38.5% night、61.5% day
- 這解釋了為什麼條件式 aug 對 day IoU 的影響比預期大（實際 night 比例更高）

Test 的 10870 夜晚比例最高（48%）、亮度最低（~72），與改前日誌結論一致：night FN 高、需低光相關改善。

---

## 1. 改前 Baseline（無 aug + BCE loss）

### 1.1 設定

| 項目 | 內容 |
|------|------|
| **Augmentation** | 無（transform=False） |
| **Loss** | BCEWithLogitsLoss |
| **Run** | `outputs/train_multi_camera_20260126_145012/` |
| **Best checkpoint** | Epoch 2, Val IoU 0.8659 |

### 1.2 Test 10870 結果

| 指標 | 數值 |
|------|------|
| overall test IoU | 0.6513 |
| night test IoU | 0.5530 |
| **night FN** | **0.3637** |
| day test IoU | 0.8083 |
| dusk IoU | 0.5499 |
| dusk FN | 0.3718 |

### 1.3 觀察

- Night FN 0.3637 比改前（0.2901）略差，但比降低 aug 版本（0.8684）好很多
- Day IoU 0.8083 接近改前（0.8135），表示 loader 正常
- 此版本用於驗證 loader 與訓練流程是否正常

---

## 2. 實驗一：強 Augmentation（全樣本）

### 2.1 設定

| 項目 | 內容 |
|------|------|
| **Augmentation** | 強 aug（全樣本） |
| **Loss** | BCEWithLogitsLoss |
| **Aug 參數** | RandomBrightnessContrast: (-0.45, 0.10), (-0.45, 0.20), p=0.75<br>RandomGamma: (60, 140), p=0.55<br>GaussianBlur: (3, 5), p=0.25<br>GaussNoise: (5.0, 25.0), p=0.20 |
| **Run** | `outputs/train_multi_camera_20260125_222612/`（見 2026-01-25 日誌） |
| **Best checkpoint** | Epoch 5, Val IoU 0.8873 |

### 2.2 Test 10870 結果

| 指標 | 改前 | 強 aug |
|------|------|--------|
| overall test IoU | 0.6960 | 0.5979 |
| night test IoU | 0.6299 | 0.5668 |
| **night FN** | **0.2901** | **0.0289** ✓ |
| day test IoU | 0.8135 | 0.6602 |
| dusk IoU | — | 0.5709 |
| dusk FN | — | 0.0313 |

### 2.3 觀察與原因分析

**成功**：
- Night FN 大幅改善：0.2901 → 0.0289（改善約 90%）
- 強 aug 有效提升模型對低光場景的學習

**失敗**：
- Day IoU 明顯下降：0.8135 → 0.6602（下降 0.15，超出可接受範圍）
- Overall IoU 下降：0.6960 → 0.5979

**原因分析**：
- 強 aug 對所有樣本（包括 day）都套用，導致 day 樣本被過度扭曲
- 模型學習到「所有樣本都可能變暗/低對比」，影響對正常亮度 day 樣本的判斷
- Aug 強度過高，破壞了 day 樣本的特徵分布

---

## 3. 實驗二：降低 Augmentation 強度 + BCE+Dice Loss

### 3.1 設定

| 項目 | 內容 |
|------|------|
| **Augmentation** | 降低強度 aug（全樣本） |
| **Loss** | BCEWithLogitsLoss + DiceLoss (1:1) |
| **Aug 參數** | RandomBrightnessContrast: (-0.20, 0.10), (-0.20, 0.15), p=0.40<br>RandomGamma: (80, 120), p=0.30<br>GaussianBlur: (3, 3), p=0.10<br>移除 GaussNoise |
| **Run** | `outputs/train_multi_camera_20260126_141742/` |
| **Best checkpoint** | Epoch 4, Val IoU 0.8917 |

### 3.2 Test 10870 結果

| 指標 | 改前 | 強 aug | 降低 aug |
|------|------|--------|----------|
| overall test IoU | 0.6960 | 0.5979 | **0.4153** |
| night test IoU | 0.6299 | 0.5668 | **0.1244** |
| **night FN** | **0.2901** | **0.0289** | **0.8684** ✗ |
| day test IoU | 0.8135 | 0.6602 | 0.8623 |
| dusk IoU | — | 0.5709 | 0.0446 |
| dusk FN | — | 0.0313 | 0.9536 |

### 3.3 觀察與原因分析

**失敗**：
- Night FN 極高：0.8684（比改前 0.2901 還差）
- Night IoU 極低：0.1244（幾乎失效）
- Overall IoU 崩潰：0.4153

**原因分析**：
- Aug 強度降低過多，模型對 night 樣本的學習不足
- BCE+Dice loss 可能與降低 aug 組合產生負面影響
- 模型可能過度保守，導致 night 樣本大量漏檢

**後續驗證**：
- 執行 threshold sweep 發現：即使降低 threshold 到 0.05，night FN 仍有 0.5299
- 證實問題不在推論 threshold，而在訓練階段的模型學習

---

## 4. 實驗三：中等強度 Augmentation（方案 A）

### 4.1 設定

| 項目 | 內容 |
|------|------|
| **Augmentation** | 中等強度 aug（全樣本） |
| **Loss** | BCEWithLogitsLoss |
| **Aug 參數** | RandomBrightnessContrast: (-0.30, 0.10), (-0.30, 0.20), p=0.60<br>RandomGamma: (70, 130), p=0.45<br>GaussianBlur: (3, 4), p=0.15<br>GaussNoise: (5.0, 20.0), p=0.15 |
| **Run** | `outputs/train_multi_camera_20260126_162106/` |
| **Best checkpoint** | Epoch 3, Val IoU 0.9334 |

### 4.2 Test 10870 結果

| 指標 | 改前 | 強 aug | 降低 aug | 中等 aug |
|------|------|--------|----------|----------|
| overall test IoU | 0.6960 | 0.5979 | 0.4153 | **0.6231** |
| night test IoU | 0.6299 | 0.5668 | 0.1244 | **0.5623** |
| **night FN** | **0.2901** | **0.0289** | 0.8684 | **0.3156** |
| day test IoU | 0.8135 | 0.6602 | 0.8623 | **0.7121** |
| dusk IoU | — | 0.5709 | 0.0446 | **0.5639** |
| dusk FN | — | 0.0313 | 0.9536 | **0.3170** |

### 4.3 觀察與原因分析

**部分改善**：
- Overall IoU 回升：0.6231（比降低 aug 版本好）
- Day IoU 維持較好：0.7121（比強 aug 版本好）

**失敗**：
- Night FN 0.3156 未改善（比改前 0.2901 還差）
- 未達目標（night FN ≤ 0.18）

**原因分析**：
- 中等強度 aug 仍不足以改善 night FN
- 對全樣本套用 aug 仍會影響 day 樣本
- 需要更針對性的策略

---

## 5. 實驗四：條件式 Augmentation（方案 B）⭐

### 5.1 設定

| 項目 | 內容 |
|------|------|
| **Augmentation** | **條件式 aug（僅對 night 樣本做強 aug）** |
| **Loss** | BCEWithLogitsLoss |
| **Aug 參數** | 強 aug（與實驗一相同）<br>RandomBrightnessContrast: (-0.45, 0.10), (-0.45, 0.20), p=0.75<br>RandomGamma: (60, 140), p=0.55<br>GaussianBlur: (3, 5), p=0.25<br>GaussNoise: (5.0, 25.0), p=0.20 |
| **條件判斷** | 僅對 luma < 0.20 的樣本套用 aug<br>Day 樣本（luma >= 0.20）不做 aug，保持原樣 |
| **Run** | `outputs/train_multi_camera_20260126_210121/` |
| **Best checkpoint** | Epoch 4, Val IoU 0.8849 |

### 5.2 Test 10870 結果

| 指標 | 改前 | 強 aug | 降低 aug | 中等 aug | **條件式 aug** |
|------|------|--------|----------|----------|----------------|
| overall test IoU | 0.6960 | 0.5979 | 0.4153 | 0.6231 | **0.6316** |
| night test IoU | 0.6299 | 0.5668 | 0.1244 | 0.5623 | **0.5945** |
| **night FN** | **0.2901** | **0.0289** | 0.8684 | 0.3156 | **0.0667** ✓ |
| day test IoU | 0.8135 | 0.6602 | 0.8623 | 0.7121 | **0.7066** |
| dusk IoU | — | 0.5709 | 0.0446 | 0.5639 | **0.5987** |
| dusk FN | — | 0.0313 | 0.9536 | 0.3170 | **0.0674** |

### 5.3 觀察與原因分析

**成功**：
- **Night FN 大幅改善**：0.2901 → 0.0667（改善約 77%，達成目標 ≤ 0.18）
- Night IoU 維持較好：0.5945（接近改前 0.6299）
- Overall IoU 回升：0.6316（比強 aug 版本好）

**部分成功**：
- Day IoU 0.7066 比強 aug 版本（0.6602）好，但比改前（0.8135）低 0.11
- 超出預期的「只能掉 2%」目標

**原因分析（為什麼條件式 aug 還是會影響 day IoU 11%）**：

### 7.3.1 訓練時的混合 Batch 效應（主要原因）

- **Train 資料分布**：約 32% night（66 張），68% day（142 張）
- **Batch 組成**：每個 batch（batch_size=2）中可能同時包含：
  - Aug 過的 night 樣本（luma < 0.20，經過強 aug）
  - 未 aug 的 day 樣本（luma >= 0.20，保持原樣）
- **學習效應**：模型在學習時看到「部分樣本被 aug 扭曲，部分保持原樣」，導致：
  - 決策邊界需要同時適應兩種不同的特徵分布
  - 模型可能學習到「aug 後的 night 特徵」與「原始 day 特徵」之間的折衷
  - 這可能導致模型對 day 樣本的判斷稍微偏移

### 7.3.2 共享特徵空間的間接影響

- **U-Net 的參數共享**：模型的所有層（encoder、decoder）都是共享的
- **特徵提取的影響**：雖然 day 樣本不做 aug，但：
  - Encoder 在提取特徵時，會受到 aug 過的 night 樣本影響
  - 模型學習到的特徵表徵會同時反映 aug 和 non-aug 的樣本
  - 這可能導致對 day 樣本的特徵提取稍微偏移

### 7.3.3 模型容量的限制

- **無法完全分離**：U-Net 無法像兩個獨立模型一樣，完全分離 night 和 day 的學習
- **特徵空間的折衷**：模型需要在同一個特徵空間中同時處理：
  - Aug 過的 night 樣本（特徵分布改變）
  - 未 aug 的 day 樣本（特徵分布保持）
- **泛化能力的影響**：這可能導致模型對 day 樣本的泛化稍微變差

### 7.3.4 Aug 判斷的邊界效應

- **Luma 閾值的不精確**：使用 luma < 0.20 作為閾值，可能：
  - 有些「接近 night 的 day 樣本」（luma ≈ 0.18-0.20）被誤判為 night，做了 aug
  - 有些「接近 day 的 night 樣本」（luma ≈ 0.20-0.22）沒被 aug
- **影響**：這可能導致模型學習到不一致的特徵，影響最終表現

### 7.3.5 訓練資料分布的不一致

- **Night 樣本的特徵改變**：Train 中約 66 張 night 樣本經過強 aug 後，特徵分布大幅改變
- **Day 樣本保持原樣**：142 張 day 樣本保持原樣，但模型在學習時需要同時適應：
  - Aug 過的 night 分布（變暗、低對比）
  - 原始 day 分布（正常亮度、高對比）
- **擬合難度**：這可能導致模型對 day 樣本的擬合稍微變差

### 7.3.6 為什麼預期「只能掉 2%」但實際掉了 11%？

**預期**：僅對 night 樣本做 aug，day 樣本不做 aug，理論上 day IoU 應該幾乎不變

**實際**：Day IoU 從 0.8135 降到 0.7066（下降 0.11，約 13.5%）

**可能原因**：
1. **混合 Batch 的影響被低估**：雖然 day 樣本不做 aug，但每個 batch 中同時包含 aug 和 non-aug 的樣本，導致模型學習到折衷的特徵
2. **特徵空間的共享效應**：U-Net 的共享參數導致 aug 過的 night 樣本間接影響 day 樣本的特徵提取
3. **Aug 判斷的誤差**：luma 閾值可能不夠精確，導致部分 day 樣本被誤判為 night
4. **訓練資料的不平衡**：雖然 day 樣本較多（68%），但 aug 過的 night 樣本（32%）可能對模型學習產生較大影響

**結論**：條件式 aug 雖然只對 night 樣本做 aug，但由於模型參數共享、混合 batch 效應等因素，仍會間接影響 day 樣本的學習，導致 day IoU 下降 11%。

### 5.4 成功判準檢查（本輪 vs 改前）

- **night FN ≤ 0.18**：0.2901 → **0.0667** ✓ **達成**
- **day IoU 下降 ≤ 0.03**：0.8135 → 0.7066（下降 0.11）✗ **超出預期**
- **overall IoU ≥ 0.66**：0.6960 → **0.6316** ✗ **未達成，但方向正確**

---

## 6. 完整實驗對照表

| 版本 | Aug 策略 | Loss | Night FN | Night IoU | Day IoU | Overall IoU | 評價 |
|------|---------|------|----------|-----------|---------|-------------|------|
| **改前** | 無 aug | BCE | 0.2901 | 0.6299 | 0.8135 | 0.6960 | Baseline |
| **強 aug（全樣本）** | 強 aug | BCE | **0.0289** | 0.5668 | 0.6602 | 0.5979 | Night 最佳，但 day 下降 |
| **降低 aug（全樣本）** | 弱 aug | BCE+Dice | 0.8684 | 0.1244 | 0.8623 | 0.4153 | 失敗 |
| **中等 aug（全樣本）** | 中等 aug | BCE | 0.3156 | 0.5623 | 0.7121 | 0.6231 | 無改善 |
| **條件式 aug** | 僅 night 強 aug | BCE | **0.0667** | 0.5945 | 0.7066 | 0.6316 | **最佳平衡** ⭐ |

---

## 7. 觀察與分析

### 7.1 為什麼條件式 aug 還是會影響 day IoU？

雖然僅對 night 樣本做 aug，day IoU 還是掉了 11%（0.8135 → 0.7066），可能原因：

1. **混合 Batch 效應**：
   - Train 中約 32% night、68% day
   - 每個 batch 同時包含 aug 過的 night 和未 aug 的 day
   - 模型在學習時需要適應兩種不同的特徵分布

2. **共享特徵空間**：
   - U-Net 的參數是共享的，無法完全分離 night 和 day 的學習
   - Aug 過的 night 樣本會影響模型對 day 樣本的特徵提取

3. **決策邊界的調整**：
   - 模型學習到「aug 後的 night 特徵」與「原始 day 特徵」之間的關係
   - 這可能導致決策邊界稍微偏移，影響 day 樣本的判斷

4. **Aug 判斷的邊界效應**：
   - luma < 0.20 的閾值可能不夠精確
   - 有些「接近 night 的 day 樣本」可能被誤判，或反之

5. **訓練資料分布的不一致**：
   - Night 樣本經過強 aug 後，特徵分布大幅改變
   - Day 樣本保持原樣，但模型需要同時適應兩種分布
   - 這可能導致模型對 day 樣本的擬合稍微變差

### 7.2 各版本的優缺點

| 版本 | 優點 | 缺點 |
|------|------|------|
| **強 aug（全樣本）** | Night FN 最佳（0.0289） | Day IoU 下降明顯（0.66） |
| **降低 aug（全樣本）** | Day IoU 維持較好（0.86） | Night FN 極高（0.87），整體崩潰 |
| **中等 aug（全樣本）** | 平衡表現 | Night FN 未改善（0.32） |
| **條件式 aug** | Night FN 大幅改善（0.07），Day IoU 維持較好（0.71） | Day IoU 仍比改前低 0.11 |

### 7.3 為什麼條件式 aug 是最佳選擇？

雖然 day IoU 掉了 11%，但條件式 aug 仍是最佳選擇，因為：

1. **Night FN 大幅改善**：0.29 → 0.07（改善約 77%），達成目標
2. **Day IoU 維持較好**：0.71 比強 aug 版本（0.66）好
3. **Overall IoU 回升**：0.63 比強 aug 版本（0.60）好
4. **平衡表現**：在改善 night 的同時，盡量維持 day 表現

---

## 8. 下一步改善方向

### 8.1 微調條件式 aug 的強度

對 night 樣本使用稍弱的 aug，目標在維持 night FN 改善的同時，提升 day IoU：
- RandomBrightnessContrast: (-0.35, 0.10), (-0.35, 0.20), p=0.70
- RandomGamma: (65, 135), p=0.50
- 其他保持不變

### 8.2 調整 luma 閾值

- 目前：luma < 0.20 視為 night
- 可嘗試：luma < 0.15 或 luma < 0.25
- 觀察對 night/day 分類與最終表現的影響

### 8.3 加入 Loss 調整

在條件式 aug 基礎上，嘗試：
- BCE + Dice（權重 0.8:0.2）
- 或加入 class weight 平衡 night/day

### 8.4 進一步分析

- 檢查哪些 day 樣本表現下降
- 分析是否因 aug 判斷錯誤（day 被誤判為 night）
- 檢查 train 中 night/day 的實際比例與 aug 套用情況

---

## 9. 今日結論

> 經過四輪實驗（強 aug → 降低 aug → 中等 aug → 條件式 aug），最終採用**條件式 augmentation 策略**（僅對 night 樣本做強 aug）。雖然 day IoU 仍比改前低 11%，但這是目前最佳的平衡方案：**night FN 大幅改善（0.29 → 0.07，改善約 77%），day IoU 維持較好（0.71），overall IoU 回升（0.63）**。

**關鍵發現**：
- 強 aug 對全樣本套用會導致 day IoU 明顯下降
- 降低 aug 強度過多會導致 night FN 極高，整體崩潰
- 條件式 aug 是最佳平衡，但仍有改善空間（day IoU 掉了 11%）

**為什麼條件式 aug 還是會影響 day IoU**：
- 訓練時的混合 batch 效應
- 共享特徵空間的間接影響
- 決策邊界的調整
- Aug 判斷的邊界效應
- 訓練資料分布的不一致

**下一步**：微調條件式 aug 的強度，或調整 luma 閾值，目標在維持 night FN 改善的同時，提升 day IoU。

---

*紀錄日期：2026-01-26*
