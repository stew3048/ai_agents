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

## 2. 詳細視覺化報告

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
