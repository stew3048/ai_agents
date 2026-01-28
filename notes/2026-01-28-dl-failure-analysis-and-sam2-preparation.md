# 2026-01-28 DL 失敗模式分析與 SAM 2.0 比較準備

> 目標：深入分析 DL 模型的失敗模式，並準備 SAM 2.0 vs DL 的對比分析工具

---

## 0. 前提

### 0.1 資料集狀態

- **Diagnostic Set**：166 個樣本（已標註情境標籤）
- **Failure Cases**：25 個 top-5 worst 樣本（5 個情境各 5 個）
  - `outputs/failure_cases_analysis.csv`
- **DL 評估結果**：`outputs/diagnostic_with_dl_metrics.csv`（含 IoU, FP Rate, FN Rate）
- **Failure Mode 標註**：`outputs/diagnostic_with_failure_modes.csv`（含 failure_mode, failure_reason）

### 0.2 今日工作重點

1. 分析特定 camera 的失敗模式
2. 生成詳細的視覺化報告
3. 準備 SAM 2.0 vs DL 比較分析工具

---

## 1. Camera 失敗模式分析

### 1.1 腳本建立

建立 `scripts/analyze_camera_failure_modes.py`：
- 讀取 `diagnostic_with_failure_modes.csv`
- 分析每個 camera 的失敗模式分布
- 統計各 camera 的 FP/FN/balanced/good 比例
- 計算平均指標（IoU, FP Rate, FN Rate）
- 列出 Top 5 FP/FN 失敗案例

### 1.2 分析結果

**輸出**：`outputs/camera_failure_modes_analysis.md`

**主要發現**：
- **Camera 3888**（Sea scene）：72% FP 主導，主要問題是海天顏色相似誤判
- **Camera 4795**（Heavy occlusion）：100% FP 主導，建築物頂部/玻璃反光被誤判
- **Camera 21444**（No-sky）：100% FP 主導，高亮度區域被誤判為天空
- **Camera 10870**（Urban, unseen）：表現相對較好，但夜間仍有 FP 問題

---

## 2. 五個情境子集的 DL 失敗模式分析

### 2.1 子集合定義與切片方法

使用 `scripts/analyze_diagnostic_failures.py` 從 `diagnostic_with_dl_metrics.csv`（166 個樣本）中，按照以下規則切片並選出每個情境的 top-5 worst 案例：

| 子集名稱 | 篩選條件 | 排序依據 | 直覺問題 |
|---------|---------|---------|---------|
| **no-sky** | `has_sky == FALSE` | `dl_pred_positive_ratio` 由大到小 | 在「沒有天空」的圖裡，哪幾張 DL 預測出最多「天空」？ |
| **sea-sky-confusable** | `sea_sky_confusable == 1` | `dl_fp_rate` 由大到小 | 海天混淆場景中，哪幾張 FP 最嚴重？ |
| **heavy-occlusion** | `occlusion == heavy` | `dl_fn_rate` 由大到小 | 嚴重遮擋下，哪幾張漏檢天空最嚴重？ |
| **night** | `light == night` 且 `has_sky == TRUE` | `dl_iou` 由小到大 | 夜間有天空的圖裡，整體 IoU 最差的前 5 張 |
| **urban** | `scene == urban` 且 `has_sky == TRUE` | `dl_fp_rate` 由大到小 | 都市場景中，哪幾張把非天空（建築、反光）當天空的 FP 最嚴重？ |

**輸出檔案**：
- `outputs/failure_cases_analysis.csv`：25 個 top-5 worst 樣本（5 個情境 × 5 個）
- `outputs/failure_analysis_report.md`：文字版分析報告

### 2.2 各子集的 DL 失敗模式摘要

#### 2.2.1 NO-SKY 子集

- **資料來源**：Camera **21444**（20 張全 no-sky）
- **Top-5**：全部來自 21444（image 11, 13, 16, 17, 20）
- **指標特徵**：
  - `dl_pred_positive_ratio` ≈ 0.41–0.45
  - `dl_fp_rate` 同上（因為沒有 GT sky，所有 positive 都是 FP）
- **一句話失敗模式**：
  > 亮牆/雪地/高亮度區域被當成天空，模型只靠顏色亮度判斷

**觀察**：DL 幾乎是「只要夠亮就很容易被當成天空」，對「沒有天空但很亮」的場景沒有語義制約。

---

#### 2.2.2 SEA–SKY-CONFUSABLE 子集

- **資料來源**：Camera **3888**（scene=sea, sea_sky_confusable=1）
- **Top-5**：image 429, 297, 873, 508, 17
- **指標特徵**：
  - `dl_fp_rate` 約 0.17–0.34
  - `dl_iou` 約 0.46–0.58
- **一句話失敗模式**：
  > 海天顏色相似導致部分誤判（海面反光/波浪被當成天空）

**觀察**：DL 會在海面反光、地平線附近「沾黏」成天空，邊界 jitter 明顯，IoU 雖不極差但 FP 偏高。

---

#### 2.2.3 HEAVY-OCCLUSION 子集

- **資料來源**：Camera **4795**（`occlusion=heavy`）
- **Top-5**：image 9, 4, 2, 8, 13
- **指標特徵**：
  - `dl_fp_rate` 明顯偏高（0.17–0.22）
  - `dl_fn_rate` 較低（0.01–0.04）
  - `dl_iou` 很低（~0.16–0.20）
- **一句話失敗模式**：
  > **建築物邊緣/旁邊的建築物被誤判為天空（FP 主導）**

**觀察**：雖然腳本用 `dl_fn_rate` 排序，但實際檢查 overlay 和數據發現，這些案例的 **FP rate 遠高於 FN rate**，主要問題是把建築物邊緣、旁邊的建築物（特別是亮色/反光部分）誤判為天空。這是 **FP 主導**的失敗模式，而非 FN（漏檢天空）。

**修正說明**：初始分析誤以為是「漏檢天空（FN）」，但實際查看 overlay 和 `diagnostic_with_failure_modes.csv` 中的 `failure_mode` 標註，這些樣本都是 `FP_dominant`，`failure_reason` 為「建築邊緣誤判 | 嚴重遮擋場景 | 整體IoU低」。

---

#### 2.2.4 NIGHT 子集

- **資料來源**：Camera **9112** + **9483**（night & has_sky=TRUE）
- **Top-5**：9112 (59, 47, 26, 91), 9483 (40)
- **指標特徵**：
  - `dl_fn_rate` 高（~0.18–0.29）
  - `dl_iou` 中等（0.64–0.82）
- **一句話失敗模式**：
  > 夜間低對比度，模型無法識別天空邊界，導致漏檢

**觀察**：在夜景中，天空與建築/山體亮度非常接近，DL 難以找出正確的 sky–non-sky 邊界，狹長天空常被吃掉。

---

#### 2.2.5 URBAN 子集

- **資料來源**：Camera **4795**（heavy urban）、部分 seen camera
- **Top-5**（在 failure_cases_analysis 裡）：4795 (11, 12, 17, 18, 16)
- **指標特徵**：
  - `dl_fp_rate` 非常高（約 0.32–0.34）
  - `dl_iou` 很低（~0.12）
- **一句話失敗模式**：
  > 建築物頂部/玻璃反光與天空混淆，模型誤判非天空為天空

**觀察**：DL 很容易把「建築物頂部、玻璃反光、強反光邊緣」塗成天空，FP 主導；特別在 4795 這種 heavy-occlusion + urban 的情境。

---

### 2.3 三份檔案之間的關係

1. **`outputs/diagnostic_with_dl_metrics.csv`**
   - 全部 166 張 diagnostic 的 DL 指標（IoU, FP Rate, FN Rate, pred_positive_ratio）
   - 來源：`scripts/eval_diagnostic_set.py` 對 `diagnostic_candidates.csv` 執行 DL inference

2. **`outputs/diagnostic_with_failure_modes.csv`**
   - 在上面再加上 `failure_mode`（FP_dominant / FN_dominant / good / balanced）與 `failure_reason` 的人工/半自動歸納
   - 來源：`scripts/add_failure_mode_to_csv.py` 自動分類 + 用戶手動修正

3. **`outputs/failure_cases_analysis.csv`**
   - 用 `scripts/analyze_diagnostic_failures.py` 從 166 張裡面，按照上述五個子集的規則，挑出每個子集的 top-5 worst，並把它們匯總成 25 筆「代表性失敗案例」
   - 用於後續 SAM 2.0 評估和對比分析

---

## 3. 詳細視覺化報告

### 2.1 腳本建立

建立 `scripts/generate_detailed_visualization_report.py`：
- 整體統計（Failure Mode 分布）
- 按 Scene/Light/Occlusion 分析
- 各情境的 Top 5 失敗案例
- 失敗原因統計

### 2.2 報告內容

**輸出**：`outputs/detailed_visualization_report.md`

**統計摘要**：
- 總樣本數：166
- FP 主導：78 張（47.0%）
- FN 主導：16 張（9.6%）
- 表現良好：67 張（40.4%）

**各 Scene 分析**：
- **URBAN**：91 個樣本，FP 主導 30 張（33.0%），FN 主導 14 張（15.4%）
- **FOREST**：25 個樣本，FP 主導 9 張（36.0%），FN 主導 1 張（4.0%）
- **SEA**：25 個樣本，FP 主導 18 張（72.0%），FN 主導 0 張（0.0%）

---

## 3. CSV 格式修正

### 3.1 問題發現

用戶反映 `diagnostic_with_failure_modes.csv` 格式跑掉：
- 使用 tab 分隔符（而非逗號）
- 編碼為 big5（Excel 無法正確打開）

### 3.2 修正方案

建立多個修正腳本：
- `scripts/fix_csv_format.py`：初步轉換
- `scripts/fix_csv_format_v2.py`：處理編碼問題
- `scripts/properly_convert_csv.py`：最終正確轉換

**修正結果**：
- 分隔符：tab → 逗號
- 編碼：big5 → UTF-8-sig（Excel 可正確打開）
- 所有欄位正確對齊

---

## 4. SAM 2.0 vs DL 比較分析準備

### 4.1 樣本選擇策略調整

**原始計劃**：從 5 個情境各選 10 個樣本（共 50 個）

**實際執行**：直接使用 `failure_cases_analysis.csv` 中的 25 個 top-5 worst 樣本
- 每個情境的 top 5 worst 案例
- 這些是 DL 模型表現最差的案例，適合測試 SAM 2.0 是否能在這些困難案例上表現更好

### 4.2 SAM 2.0 Inference 腳本

建立 `scripts/eval_sam2_failure_cases.py`：

**功能**：
- 讀取 `failure_cases_analysis.csv`（25 個樣本）
- 載入 SAM 2.0 模型（支援多種模型大小）
- 實作多種提示策略：
  - **no-sky**：不使用提示（讓 SAM 自動分割）
  - **sea-sky-confusable**：Box prompt（框選上方 30-50% 區域）
  - **heavy-occlusion**：Point prompt（多點提示）
  - **night**：Point prompt（中心點）
  - **urban**：Box prompt（框選上方 40% 區域）
- 計算指標（IoU, FP Rate, FN Rate）
- 生成 overlay 視覺化（與 DL 相同格式）

**輸出**：
- `outputs/sam2_failure_cases_metrics.csv`：SAM 2.0 指標結果
- `outputs/sam2_overlays/`：SAM 2.0 overlay 圖片

### 4.3 對比報告腳本

建立 `scripts/generate_dl_vs_sam_comparison.py`：

**功能**：
- 合併 DL 和 SAM 2.0 結果
- 計算差異和改善率
- 生成對比 CSV 和報告

**輸出**：
- `outputs/dl_vs_sam_comparison.csv`：對比表格
- `outputs/dl_vs_sam_comparison_report.md`：詳細對比報告
  - 整體統計
  - 各情境分析（平均指標、改善情況）
  - Top 改善/惡化案例
  - 總結

### 4.4 使用說明文件

建立 `scripts/README_SAM2_COMPARISON.md`：
- 安裝說明
- 使用步驟
- 參數說明
- 故障排除

---

## 5. 依賴套件更新

更新 `requirements.txt`：
- 新增 `sam2>=1.1.0`（SAM 2.0 套件）

---

## 6. 建立的腳本清單

### 6.1 分析腳本
- `scripts/analyze_camera_failure_modes.py`：Camera 失敗模式分析
- `scripts/generate_detailed_visualization_report.py`：詳細視覺化報告
- `scripts/analyze_updated_failure_modes.py`：分析用戶修改後的 failure modes
- `scripts/generate_detailed_scene_analysis.py`：詳細 scene 分析

### 6.2 CSV 處理腳本
- `scripts/add_failure_mode_to_csv.py`：為 CSV 添加 failure_mode 欄位
- `scripts/fix_csv_format.py`：修正 CSV 格式
- `scripts/fix_csv_format_v2.py`：修正 CSV 格式（v2）
- `scripts/properly_convert_csv.py`：正確轉換 CSV
- `scripts/fix_csv_encoding_issues.py`：修正編碼問題

### 6.3 SAM 2.0 相關腳本
- `scripts/eval_sam2_failure_cases.py`：SAM 2.0 inference 主腳本
- `scripts/generate_dl_vs_sam_comparison.py`：生成對比報告
- `scripts/prepare_vlm_sam_comparison.py`：準備 VLM/SAM 比較（已建立但未使用）

### 6.4 其他腳本
- `scripts/README_SAM2_COMPARISON.md`：使用說明

---

## 7. 輸出檔案

### 7.1 分析報告
- `outputs/camera_failure_modes_analysis.md`：Camera 失敗模式分析
- `outputs/detailed_visualization_report.md`：詳細視覺化報告
- `outputs/updated_failure_analysis_summary.md`：更新後的失敗分析摘要
- `outputs/detailed_scene_failure_analysis.md`：詳細 scene 失敗分析

### 7.2 CSV 檔案
- `outputs/diagnostic_with_failure_modes.csv`：含 failure_mode 的診斷資料（已修正格式）
- `outputs/vlm_sam_comparison_samples.csv`：VLM/SAM 比較樣本（50 個，未使用）

---

## 8. 下一步工作

### 8.1 立即執行（下次回來時）

1. **安裝 SAM 2.0**：
   ```bash
   pip install sam2
   ```

2. **執行 SAM 2.0 Inference**：
   ```bash
   python scripts/eval_sam2_failure_cases.py
   ```
   - 這會對 25 個 failure cases 執行 SAM 2.0 inference
   - 生成 `outputs/sam2_failure_cases_metrics.csv` 和 `outputs/sam2_overlays/`

3. **生成對比報告**：
   ```bash
   python scripts/generate_dl_vs_sam_comparison.py
   ```
   - 這會生成 DL vs SAM 2.0 的對比分析
   - 輸出 `outputs/dl_vs_sam_comparison.csv` 和 `outputs/dl_vs_sam_comparison_report.md`

### 8.2 後續分析方向

1. **驗證 SAM 2.0 預期行為假設**：
   - 檢查 SAM 2.0 是否在特定情境下表現更好
   - 分析 SAM 2.0 的失敗模式是否與 DL 不同

2. **深入分析改善/惡化案例**：
   - 查看 overlay 圖片，理解為什麼 SAM 2.0 在某些案例上表現更好/更差
   - 分析提示策略的效果

3. **建立選模策略**：
   - 基於對比結果，建立「什麼情境適合用什麼方法」的策略
   - 可能需要嘗試不同的提示策略

4. **擴展到其他 VLM**：
   - 如果 SAM 2.0 表現良好，可以考慮測試其他 VLM（如 CLIPSeg, Grounding DINO + SAM）

---

## 9. 今日結論

1. **完成 DL 失敗模式深度分析**：建立了多個分析腳本，深入理解各 camera 和各情境的失敗模式

2. **準備 SAM 2.0 比較工具**：建立了完整的 SAM 2.0 inference 和對比分析工具，可以直接使用

3. **修正資料格式問題**：解決了 CSV 格式和編碼問題，確保資料可以正確使用

4. **明確下一步方向**：使用 25 個 top-5 worst 樣本進行 SAM 2.0 評估，驗證 SAM 2.0 是否能在 DL 的困難案例上表現更好

**下次回來時**：直接執行 SAM 2.0 inference 和對比分析，開始驗證不同方法在不同情境下的表現差異。

---

## 10. 輸出檔與指令

### 10.1 新增的腳本
- `scripts/analyze_camera_failure_modes.py`
- `scripts/generate_detailed_visualization_report.py`
- `scripts/eval_sam2_failure_cases.py`
- `scripts/generate_dl_vs_sam_comparison.py`
- `scripts/add_failure_mode_to_csv.py`
- `scripts/fix_csv_format*.py`（多個版本）
- `scripts/README_SAM2_COMPARISON.md`

### 10.2 生成的報告
- `outputs/camera_failure_modes_analysis.md`
- `outputs/detailed_visualization_report.md`
- `outputs/updated_failure_analysis_summary.md`
- `outputs/detailed_scene_failure_analysis.md`

### 10.3 下次執行的指令

```bash
# 1. 安裝 SAM 2.0
pip install sam2

# 2. 執行 SAM 2.0 inference（25 個 failure cases）
python scripts/eval_sam2_failure_cases.py

# 3. 生成對比報告
python scripts/generate_dl_vs_sam_comparison.py
```

---

*建立日期：2026-01-28*
