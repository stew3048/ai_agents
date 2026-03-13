# 2026-01-28 建立情境診斷資料集與標註

> 目標：重新調整 sky segmentation 方向，從模型 tuning 轉向「理解不同方法在不同情境下為什麼會成功/失敗」。建立「情境診斷用資料集」與「基礎標註 CSV」，用於比較 DL 模型 vs VLM/SAM 在不同情境的行為差異。

---

## 0. 背景與目的

### 0.1 研究方向調整

**原本方向**：繼續 tuning 模型，提升 IoU 指標

**新方向**：
1. 理解天空有很多情境
2. 理解有不同的模型
3. 建立方法論：什麼樣的情境適合什麼模型（選模策略）

### 0.2 目標產出

產生兩份資料清單與標註工具：

**A. Hold-out Test（嚴格泛化）**
- 僅使用 camera_id = 10870
- 全部影像都列出（94 張）
- 用於評估 unseen camera 的泛化（light/night 等）

**B. Diagnostic Scenario Set（情境理解用）**
- 從所有 camera（train / val / test）中抽樣
- 每個 camera 抽大約 10–20 張
- 總量控制在 80–120 張左右
- **重要**：這不是正式 test，不需要避免 train 影像
- 目的：方法理解與 failure mode 分析，不是公平測試
- 補齊多場景 / 遮擋 / 海天一色 / 不同光照

---

## 1. 資料集 Schema 定義

### 1.1 CSV 欄位結構

從 `data/test_data/test_metadata.csv` 定義的 schema：

| 欄位 | 說明 | 自動/手動 |
|------|------|-----------|
| `camera_id` | 相機 ID | 自動 |
| `image_id` | 圖片 ID | 自動 |
| `path` | 圖片路徑 | 自動 |
| `has_sky` | 是否有天空 | 自動（從 mask 判斷） |
| `sea_sky_confusable` | 海天混淆 | 手動 |
| `scene(sea\|urban\|forest\|other)` | 場景類型 | 手動（預設 unknown） |
| `occlusion(none\|partial\|heavy)` | 遮擋程度 | 手動（預設 unknown） |
| `weather(clear\|cloudy\|rain\|fog\|snow\|unknown)` | 天氣 | 自動（嘗試粗略判斷） |
| `light (day\|dusk\|night)` | 光線條件 | 自動（mean_luma_linear） |
| `pred_sky_area_ratio` | 天空區域比例 | 自動（從 mask 計算） |
| `notes` | 備註 | 手動 |

### 1.2 自動標註邏輯

**Light 判斷**（使用 `mean_luma_linear`）：
- `night`: luma < 0.15
- `dusk`: 0.15 <= luma < 0.30
- `day`: luma >= 0.30

**Weather 判斷**（基於圖像特徵）：
- `clear`: 高亮度、高對比度、高飽和度
- `cloudy`: 中等亮度、低飽和度、灰色區域多
- `fog`: 中等亮度、低對比度、邊緣模糊
- `rain`: 低亮度、可能有反光
- `snow`: 高亮度、大量白色區域
- `unknown`: 無法明確判斷時

---

## 2. 實施過程

### 2.1 腳本建立

建立 `scripts/create_diagnostic_datasets.py`，功能包括：

1. **掃描所有 Camera 資料**
   - 掃描 `data/skyfinder_*` 資料夾
   - 跳過 `sample` 資料夾（與 10066 相同）
   - 收集所有可用 camera 的圖片和 mask

2. **產生 Hold-out Test 清單**
   - 僅選取 camera_id = 10870 的所有圖片（94 張）
   - 自動計算所有可推斷的標籤

3. **產生 Diagnostic Scenario Set 清單**
   - 從所有 6 個 camera 抽樣（不區分 train/val/test）
   - 抽樣策略：
     - 每個 camera 基礎配額約 15 張
     - 根據 camera 特性調整（night 比例高的多抽 night 樣本）
     - 10870 固定抽 10 張（取代原本 sample 的 10 張）
     - 確保每個 camera 都有 day/dusk/night 的代表樣本
     - 偏向抽取少見情境（如夜晚、遮擋）

4. **生成縮圖 Preview**
   - 為 Diagnostic Set 生成縮圖（256x256）
   - 按 `camera_id` 分資料夾組織
   - 檔名使用 `image_id`，方便對應回 CSV

### 2.2 可用 Camera 資料

根據 `outputs/camera_inventory.csv`，ready 狀態的 camera：

| Camera | 張數 | Night 比例 | 平均亮度 | 用途 |
|--------|------|------------|----------|------|
| 10066 | 50 | 2% | 81.8 | Train |
| 10870 | 94 | 54% | 39.2 | Test / Diagnostic |
| 1093 | 90 | 18% | 59.7 | Train |
| 9112 | 100 | 48% | 52.8 | Train |
| 9291 | 82 | 64% | 39.5 | Train |
| 9483 | 100 | 22% | 94.7 | Val |

總計：6 個 camera，516 張圖片

---

## 3. 執行結果

### 3.1 Hold-out Test（camera 10870）

**產出檔案**：`data/test_data/holdout_10870.csv`

**統計**：
- 總數：94 張
- Light 分布：
  - day: 23 張 (24.5%)
  - dusk: 13 張 (13.8%)
  - night: 58 張 (61.7%)

**觀察**：
- Camera 10870 以 night 樣本為主（61.7%），符合其 night_ratio=54% 的特性
- 適合用於評估模型在夜間場景的泛化能力

### 3.2 Diagnostic Scenario Set

**產出檔案**：`data/test_data/diagnostic_candidates.csv`

**抽樣配額與分布**：

| Camera | 抽樣數 | Day | Dusk | Night |
|--------|--------|-----|------|-------|
| 10870 | 10 | 2 | 2 | 6 |
| 10066 | 10 | 3 | 6 | 1 |
| 1093 | 19 | 2 | 13 | 4 |
| 9112 | 21 | 10 | 2 | 9 |
| 9291 | 17 | 2 | 8 | 7 |
| 9483 | 21 | 11 | 2 | 8 |
| **總計** | **96** | **30** | **33** | **35** |

**整體統計**：
- 總數：96 張
- Light 分布：
  - day: 30 張 (31.2%)
  - dusk: 33 張 (34.4%)
  - night: 33 張 (34.4%)
- Weather 分布：
  - unknown: 70 張 (72.9%)
  - snow: 13 張 (13.5%)
  - rain: 11 張 (11.5%)
  - clear: 1 張 (1.0%)
  - cloudy: 1 張 (1.0%)

**觀察**：
- Light 分布相對均衡（day/dusk/night 各約 1/3）
- Weather 自動判斷大部分為 unknown，需要人工補標註
- 10870 的 10 張樣本中，night 佔 60%，符合其特性

### 3.3 縮圖 Preview

**產出位置**：`data/test_data/previews/`

**結構**：
```
previews/
├── camera_10066/    (16 張縮圖)
├── camera_10870/    (10 張縮圖)
├── camera_1093/     (33 張縮圖)
├── camera_9112/     (40 張縮圖)
├── camera_9291/     (28 張縮圖)
└── camera_9483/     (36 張縮圖)
```

**用途**：方便人工標註時快速瀏覽圖片，補齊 `scene`、`occlusion`、`sea_sky_confusable` 等欄位

---

## 4. 技術細節

### 4.1 抽樣策略

**目標**：從 6 個 camera 抽樣 96 張，確保情境多樣性

**策略**：
1. 先計算每個 camera 的亮度分布（day/dusk/night）
2. 按比例抽樣，確保每組都有代表
3. 如果某些情境樣本太少（如夜晚、遮擋），偏向多抽一些
4. 10870 固定抽 10 張（取代原本 sample 的 10 張）

**配額計算**：
- 基礎配額：根據總數比例
- 調整：night 比例高的 camera 多抽 20%
- 確保至少每組都有 2-3 張代表

### 4.2 自動標註實作

**Light 判斷**：
- 使用 `utils.image_utils.mean_luma_linear` 計算平均亮度
- 閾值：night < 0.15, dusk 0.15-0.30, day >= 0.30

**Weather 判斷**：
- 計算圖像特徵：mean_luma, std_luma, color_saturation, contrast, edge_density
- 使用簡單規則判斷：clear/cloudy/fog/rain/snow
- 不確定時標記為 "unknown"

**Has_sky 判斷**：
- 讀取 mask 檔案
- 檢查是否有非零像素（> 128）

**Pred_sky_area_ratio 計算**：
- 計算 mask 中天空像素數量 / 總像素數量

---

## 5. 產出檔案

### 5.1 CSV 檔案

- `data/test_data/holdout_10870.csv` - Hold-out Test 清單（94 張）
- `data/test_data/diagnostic_candidates.csv` - Diagnostic Scenario Set 清單（96 張）

### 5.2 縮圖 Preview

- `data/test_data/previews/` - 按 camera_id 分組的縮圖資料夾

### 5.3 腳本

- `scripts/create_diagnostic_datasets.py` - 產生資料集與標註的腳本

---

## 6. 後續工作

### 6.1 人工標註

需要人工補齊的欄位：
- `sea_sky_confusable`: 檢查圖片是否有海天混淆的情況
- `scene`: 判斷場景類型（sea/urban/forest/other）
- `occlusion`: 評估遮擋程度（none/partial/heavy）
- `weather`: 修正自動判斷的結果（目前大部分為 unknown）
- `notes`: 任何額外備註

### 6.2 使用方式

1. 開啟 `diagnostic_candidates.csv`
2. 對照 `previews/` 資料夾中的縮圖
3. 逐筆補齊需要手動標註的欄位
4. 完成後可用於：
   - 比較不同模型（DL vs VLM/SAM）在不同情境下的表現
   - 分析 failure mode
   - 建立選模策略

---

## 7. 今日結論

> **成功建立情境診斷資料集與標註工具，包含 Hold-out Test（94 張）和 Diagnostic Scenario Set（96 張）。自動標註了 light、has_sky、pred_sky_area_ratio 和部分 weather，並生成縮圖 preview 方便人工標註。這為後續的方法理解與選模策略分析奠定了基礎。**

**關鍵成果**：
1. ✅ 產生兩份資料清單（Hold-out Test 和 Diagnostic Scenario Set）
2. ✅ 實現自動標註邏輯（light、has_sky、pred_sky_area_ratio、weather）
3. ✅ 生成縮圖 preview 方便人工標註
4. ✅ 跳過 sample 資料夾，用 10870 的 10 張取代

**下一步**：
- 人工補齊 CSV 中的 `scene`、`occlusion`、`sea_sky_confusable` 等欄位
- 使用完成標註的資料集進行不同模型的比較分析

---

## 8. 後續更新：新增三個新 Camera 補齊情境覆蓋

### 8.1 新增 Camera 的背景

為了補齊 diagnostic set 的情境覆蓋，新增三個新 camera：

| Camera | 張數 | 特殊情境 | 說明 |
|--------|------|----------|------|
| **3888** | ~1626 | `scene=sea`, `sea_sky_confusable=1` | 海天易混淆情境（原本資料中沒有） |
| **4795** | ~725 | `occlusion=heavy` | 嚴重遮擋情境 |
| **21444** | ~305 | `has_sky=FALSE` | 完全沒有天空的情境 |

### 8.2 資料整理過程

**步驟**：
1. 將 `data/3888`、`data/4795`、`data/21444` 整理成標準格式
2. 建立 `images/` 和 `masks/` 資料夾結構
3. 重新命名為 `data/skyfinder_3888/`、`data/skyfinder_4795/`、`data/skyfinder_21444/`
4. 將 images 重新命名為標準格式（001.jpg, 002.jpg, ...）
5. 使用單一 template mask 複製成對應的 mask（001.png, 002.png, ...）

**腳本**：`scripts/prepare_new_cameras_for_diagnostic.py`

### 8.3 加入 Diagnostic Set

**抽樣策略**：
- 每個新 camera 各抽 20 張（按檔名排序取前 20）
- 總計新增 60 張到 `diagnostic_candidates.csv`

**預設標註**（針對新加入的 20 張）：
- **3888**：
  - `scene=sea`
  - `sea_sky_confusable=1`（首次有真正的海天混淆 case）
  - `has_sky=TRUE`
  - `occlusion=none`
- **4795**：
  - `occlusion=heavy`（嚴重遮擋）
  - `scene=urban`（預設）
- **21444**：
  - `has_sky=FALSE`（完全沒有天空）

### 8.4 手動補標註

**3888 的額外 5 張**（手動補標）：
- `297`: `weather=rain, light=day`
- `429`: `weather=fog, light=day`
- `444`: `weather=clear, light=dusk`
- `508`: `weather=clear, light=day`
- `873`: `weather=fog, light=day`

這些補標讓「海＋各種天氣／光線」的組合更完整，特別是 `sea_sky_confusable=1` 配合不同 weather/light 的情境。

### 8.5 人工標註完成狀態

✅ **已完成人工標註的欄位**：
- `scene`：sea / urban / forest / other（四類都有覆蓋）
- `occlusion`：none / partial / heavy（三類都有覆蓋）
- `weather`：clear / cloudy / rain / fog / snow / unknown
- `light`：day / dusk / night（已修正自動判斷的錯誤）

✅ **發現**：
- `sea_sky_confusable` 全為 0（原本資料中沒有真正的海天混淆 case）
- 因此引入 camera 3888 來補齊這個情境

---

## 9. Diagnostic Set 的完整情境覆蓋

### 9.1 目前 Diagnostic Candidates 的 Camera 分布

根據 `data/test_data/diagnostic_candidates.csv`（最新版本）：

| Camera | 樣本數 | 在 DL Model 中的角色 | 說明 |
|--------|--------|---------------------|------|
| 10066 | 13 | Train | 原本訓練資料 |
| 1093 | 19 | Train | 原本訓練資料 |
| 9112 | 21 | Train | 原本訓練資料 |
| 9291 | 17 | Train | 原本訓練資料 |
| 9483 | 23 | Val | 原本驗證資料 |
| 10870 | 10 | Test | 原本測試資料（unseen camera） |
| **3888** | **25** | **Unseen** | 新增：海天混淆情境 |
| **4795** | **20** | **Unseen** | 新增：嚴重遮擋情境 |
| **21444** | **20** | **Unseen** | 新增：無天空情境 |
| **總計** | **168** | | |

### 9.2 Unseen Camera 記錄

**重要**：以下 camera 在 diagnostic set 中，但**不在原本 DL model 的訓練資料中**：

| Camera ID | 樣本數 | 特殊情境 | 用途 |
|-----------|--------|----------|------|
| **3888** | 25 | `scene=sea`, `sea_sky_confusable=1` | 測試海天混淆情境下的模型表現 |
| **4795** | 20 | `occlusion=heavy` | 測試嚴重遮擋情境下的模型表現 |
| **21444** | 20 | `has_sky=FALSE` | 測試無天空情境下的模型表現（FP 分析） |
| **10870** | 10 | Test camera（原本就是 unseen） | 跨 camera 泛化評估 |

**原本 DL Model 見過的 Camera**：
- Train: 10066, 9291, 9112, 1093
- Val: 9483
- Test: 10870（但 diagnostic set 中只抽了 10 張）

**總計 Unseen Camera 樣本**：75 張（3888: 25 + 4795: 20 + 21444: 20 + 10870: 10）

### 9.3 情境維度完整度

目前 diagnostic set 已涵蓋：

**Scene 維度**：
- ✅ `sea`：3888 提供（25 張，含 `sea_sky_confusable=1`）
- ✅ `urban`：多個 camera 提供
- ✅ `forest`：9291, 10066 提供
- ✅ `other`：可能有

**Occlusion 維度**：
- ✅ `none`：多個 camera 提供
- ✅ `partial`：多個 camera 提供
- ✅ `heavy`：4795 提供（20 張）

**Weather 維度**：
- ✅ `clear`：多個 camera 提供
- ✅ `cloudy`：多個 camera 提供
- ✅ `rain`：3888 (297), 1093, 9483 提供
- ✅ `fog`：3888 (429, 873), 10066, 9483 提供
- ✅ `snow`：9483 提供
- ✅ `unknown`：部分樣本待補

**Light 維度**：
- ✅ `day`：多個 camera 提供
- ✅ `dusk`：多個 camera 提供
- ✅ `night`：多個 camera 提供

**Sky Presence / Confusability**：
- ✅ `has_sky=TRUE`：大部分 camera
- ✅ `has_sky=FALSE`：21444 提供（20 張）
- ✅ `sea_sky_confusable=1`：3888 提供（25 張）

---

## 10. 方法理解與選模策略的準備

### 10.1 Diagnostic Set 的用途

這份 diagnostic set 現在可以用來：

1. **比較不同方法在不同情境下的表現**
   - DL model（U-Net / multi-camera trained）
   - VLM（如 GPT-4V, Claude Vision）
   - SAM（Segment Anything Model）

2. **分析 Failure Mode**
   - 哪些情境下 DL model 表現差？
   - 哪些情境下 VLM/SAM 表現好？
   - Unseen camera（3888, 4795, 21444）vs Seen camera 的差異

3. **建立選模策略**
   - 什麼情境適合用 DL model？
   - 什麼情境適合用 VLM/SAM？
   - 什麼情境需要 ensemble 或 fallback？

### 10.2 後續分析方向

**建議的分析軸**：
- Camera（domain shift）：Seen vs Unseen
- Scene：sea vs urban vs forest
- Occlusion：none vs partial vs heavy
- Weather：clear vs fog vs rain vs snow
- Light：day vs dusk vs night
- Sky presence：has_sky=TRUE vs FALSE
- Sea-sky confusion：sea_sky_confusable=0 vs 1

**下一步**：
- 使用這份 diagnostic set 跑不同模型的預測
- 計算各情境組合下的 IoU / FP / FN
- 對照出「哪種方法在哪些情境特別好/特別差」

---

## 11. 今日結論（更新）

> **成功建立完整的情境診斷資料集，包含 168 張樣本，涵蓋 9 個 camera（其中 4 個是 unseen）。已完成人工標註 scene、occlusion、weather、light 四個情境維度，並補齊了原本缺失的 `sea_sky_confusable=1` 和 `has_sky=FALSE` 情境。這為後續的方法理解與選模策略分析提供了完整的基礎。**

**關鍵成果**：
1. ✅ 產生兩份資料清單（Hold-out Test 和 Diagnostic Scenario Set）
2. ✅ 實現自動標註邏輯（light、has_sky、pred_sky_area_ratio、weather）
3. ✅ 生成縮圖 preview 方便人工標註
4. ✅ 新增三個新 camera（3888, 4795, 21444）補齊情境覆蓋
5. ✅ 完成人工標註四個情境維度（scene, occlusion, weather, light）
6. ✅ 記錄 unseen camera（3888, 4795, 21444, 10870）用於 domain shift 分析

**重要發現**：
- 原本資料中沒有真正的 `sea_sky_confusable=1` case，透過引入 3888 補齊
- 透過 4795 補齊 `occlusion=heavy` 情境
- 透過 21444 補齊 `has_sky=FALSE` 情境

**下一步**：
- 使用這份 diagnostic set 進行不同模型（DL vs VLM/SAM）的比較分析
- 建立情境 vs 方法的 success/failure 對照表
- 發展選模策略

---

*紀錄日期：2026-01-28（更新）*
