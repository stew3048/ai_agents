# SAM 2.0 vs DL 比較分析使用說明

## 概述

本專案提供了對 DL 模型和 SAM 2.0 在失敗案例上的對比分析工具。使用 `outputs/failure_cases_analysis.csv` 中的 25 個 top-5 worst 樣本進行評估。

## 前置需求

1. 安裝 SAM 2.0：
```bash
pip install sam2
```

2. 下載 SAM 2.0 模型權重（首次使用時會自動下載，或手動下載）：
- 模型會自動下載到 `~/.cache/sam2/` 目錄

## 使用步驟

### 步驟 1: 執行 SAM 2.0 Inference

```bash
python scripts/eval_sam2_failure_cases.py
```

**參數說明**：
- `--input_csv`: 輸入 CSV（預設：`outputs/failure_cases_analysis.csv`）
- `--output_csv`: 輸出 CSV（預設：`outputs/sam2_failure_cases_metrics.csv`）
- `--overlay_dir`: Overlay 輸出目錄（預設：`outputs/sam2_overlays/`）
- `--model_type`: SAM 2.0 模型類型（預設：`sam2_hiera_large`）
  - 選項：`sam2_hiera_tiny`, `sam2_hiera_small`, `sam2_hiera_base`, `sam2_hiera_large`
- `--device`: 計算設備（預設：`cuda`，如果不可用會自動切換到 CPU）

**輸出**：
- `outputs/sam2_failure_cases_metrics.csv`: SAM 2.0 的指標結果
- `outputs/sam2_overlays/`: SAM 2.0 的 overlay 視覺化圖片

### 步驟 2: 生成對比報告

```bash
python scripts/generate_dl_vs_sam_comparison.py
```

**參數說明**：
- `--dl_csv`: DL 結果 CSV（預設：`outputs/failure_cases_analysis.csv`）
- `--sam_csv`: SAM 2.0 結果 CSV（預設：`outputs/sam2_failure_cases_metrics.csv`）
- `--comparison_csv`: 對比 CSV 輸出路徑（預設：`outputs/dl_vs_sam_comparison.csv`）
- `--comparison_report`: 對比報告輸出路徑（預設：`outputs/dl_vs_sam_comparison_report.md`）

**輸出**：
- `outputs/dl_vs_sam_comparison.csv`: DL vs SAM 2.0 對比表格
- `outputs/dl_vs_sam_comparison_report.md`: 詳細對比報告

## 提示策略

腳本會根據不同情境自動選擇提示策略：

- **no-sky**: 不使用提示（讓 SAM 自動分割）
- **sea-sky-confusable**: Box prompt（框選上方 30-50% 區域）
- **heavy-occlusion**: Point prompt（多點提示，在圖片上方選多個點）
- **night**: Point prompt（在圖片上方中心選點）
- **urban**: Box prompt（框選上方 40% 區域）

## 輸出檔案說明

### sam2_failure_cases_metrics.csv
包含每個樣本的 SAM 2.0 指標：
- `sam2_iou`: IoU 分數
- `sam2_fp_rate`: False Positive Rate
- `sam2_fn_rate`: False Negative Rate
- `sam2_pred_positive_ratio`: 預測為天空的像素比例
- `sam2_overlay_path`: Overlay 圖片路徑

### dl_vs_sam_comparison.csv
包含 DL 和 SAM 2.0 的對比：
- DL 指標（`dl_iou`, `dl_fp_rate`, `dl_fn_rate`）
- SAM 2.0 指標（`sam2_iou`, `sam2_fp_rate`, `sam2_fn_rate`）
- 差異（`iou_diff`, `fp_rate_diff`, `fn_rate_diff`）
- 改善率（`iou_improvement`, `fp_rate_improvement`, `fn_rate_improvement`）

### dl_vs_sam_comparison_report.md
詳細的對比報告，包含：
- 整體統計
- 各情境的分析（平均指標、改善情況）
- Top 改善/惡化案例
- 總結

## 注意事項

1. **首次運行**：SAM 2.0 模型會自動下載，可能需要一些時間
2. **GPU 記憶體**：如果使用 `sam2_hiera_large` 模型，建議至少有 8GB GPU 記憶體
3. **圖片路徑**：腳本會自動從 `diagnostic_with_failure_modes.csv` 讀取圖片路徑
4. **GT Mask**：腳本會自動載入對應的 GT mask 進行評估

## 故障排除

1. **SAM 2.0 導入錯誤**：
   - 確認已安裝：`pip install sam2`
   - 確認 Python 版本 >= 3.10

2. **找不到圖片**：
   - 確認 `data/skyfinder_*/images/` 目錄存在
   - 檢查 `diagnostic_with_failure_modes.csv` 中的 path 欄位

3. **CUDA 記憶體不足**：
   - 使用較小的模型：`--model_type sam2_hiera_small`
   - 或使用 CPU：`--device cpu`
