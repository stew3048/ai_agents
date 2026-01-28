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

*紀錄日期：2026-01-28*
