# 2026-01-28 SAM 2.0 Inference 與 DL 對比流程

> 依計畫執行：對 25 個 failure cases 跑 SAM 2.0 inference，並生成 DL vs SAM 2.0 對比報告。

---

## 1. 執行指令與參數

### 環境

- 使用專案 **.venv**（Python 3.13），已安裝 `sam2>=1.1.0`。
- 腳本已適配 sam2 1.1+ API：改為 `SAM2ImagePredictor.from_pretrained(model_id)`（HuggingFace model_id）。

### SAM 2.0 Inference（實際執行）

```bash
.\.venv\Scripts\Activate.ps1
python scripts/eval_sam2_failure_cases.py --model_type sam2_hiera_small --device cpu
```

- **模型**：`sam2_hiera_small`（facebook/sam2-hiera-small）
- **設備**：CPU（CUDA 不可用時自動切換）
- 輸入：`outputs/failure_cases_analysis.csv`（25 筆）、`outputs/diagnostic_with_failure_modes.csv`
- 輸出：`outputs/sam2_failure_cases_metrics.csv`、`outputs/sam2_overlays/`（25 張 overlay）

### 生成 DL vs SAM 2.0 對比報告

```bash
python scripts/generate_dl_vs_sam_comparison.py
```

- 輸出：`outputs/dl_vs_sam_comparison.csv`、`outputs/dl_vs_sam_comparison_report.md`

---

## 2. 各情境 DL vs SAM 2.0 差異（實際結果）

- **整體**：DL 平均 IoU 0.387，SAM 2.0 平均 IoU 0.450，平均 IoU 改善 +0.063（約 +16.3%）。改善 12 筆、惡化 8 筆、相近 1 筆。

| 情境 | 平均 IoU 改善 | 改善/惡化筆數 | 摘要 |
|------|----------------|----------------|------|
| **no-sky** | （無 IoU） | - | SAM 2.0 FP Rate 大降（0.42→0.02，約 -96%），亮牆/雪地誤判大幅減少 |
| **sea-sky-confusable** | -0.014（略降） | 2 改善 / 3 惡化 | 海天混淆情境 SAM 與 DL 接近，部分樣本 SAM 略差 |
| **heavy-occlusion** | -0.141（惡化） | 0 改善 / 5 惡化 | 嚴重遮擋下 SAM 2.0 明顯較差（點提示易偏離天空區域） |
| **night** | +0.201（約 +28%） | 5 改善 / 0 惡化 | 夜間 SAM 2.0 明顯較穩，IoU 與 FN 皆改善 |
| **urban** | +0.206（約 +170%） | 5 改善 / 0 惡化 | 都市建築/反光情境 SAM 2.0 明顯較佳，FP 大降 |

- **小結**：SAM 2.0 在 **no-sky、night、urban** 上明顯優於 DL；在 **heavy-occlusion** 上較差，可考慮調整提示策略（例如改 box 或更保守的點位）。

---

## 3. 輸出檔與指令摘要

| 產物 | 路徑 |
|------|------|
| SAM 2.0 指標 | `outputs/sam2_failure_cases_metrics.csv` |
| SAM 2.0 overlay | `outputs/sam2_overlays/` |
| 對比逐筆 | `outputs/dl_vs_sam_comparison.csv` |
| 對比報告 | `outputs/dl_vs_sam_comparison_report.md` |

**日後重跑（.venv 已備好）**：

1. `.\.venv\Scripts\Activate.ps1`
2. `python scripts/eval_sam2_failure_cases.py`（可加 `--model_type sam2_hiera_large --device cuda` 若有 GPU）
3. `python scripts/generate_dl_vs_sam_comparison.py`

---

## 4. 下一步

- **heavy-occlusion**：嘗試不同提示（例如上方 box、或多點更集中於天空區），或改用 sam2_hiera_large 比較。
- **sea-sky-confusable**：可微調 box/點位或比較 base-plus / large 模型。
- 可選：擴充到其他 VLM 或更多 diagnostic 樣本。

---

## 5. 今日結論

- 以 **.venv（Python 3.13）+ sam2** 完成 25 筆 failure cases 的 SAM 2.0 推論與 DL 對比。
- 腳本已適配 sam2 1.1+ 的 `from_pretrained` API；對比報告已修正除零錯誤（avg_dl_fn=0 時顯示 N/A）。
- SAM 2.0 在 no-sky / night / urban 表現較佳，在 heavy-occlusion 較差，可依情境再調提示或模型。

---

## 6. 今日工作紀要

- **計畫執行**：依 SAM 2.0 inference 與 DL 對比計畫，完成步驟 2（推論）與步驟 3（對比報告）；步驟 1 安裝改以專案 .venv 處理。
- **環境**：以 `py -3.13 -m venv .venv` 建立 Python 3.13 虛擬環境，`pip install -r requirements.txt` 安裝依賴（含 sam2）；`requirements.txt` 移除中文註解以避免 Windows cp950 讀取錯誤；`.gitignore` 新增 `.venv/`。
- **腳本修正**：  
  - `eval_sam2_failure_cases.py`：適配 sam2 1.1+ API，改為 `SAM2ImagePredictor.from_pretrained(model_id)`，新增 `MODEL_TYPE_TO_HF_ID` 對照；修正迴圈內 `key` 使用前未定義；SAM 未安裝時改為自動 stub 模式產出空白 CSV。  
  - `generate_dl_vs_sam_comparison.py`：平均 FN/FP/IoU 改善率計算時若分母為 0 改為顯示 N/A，避免 ZeroDivisionError。
- **執行**：於 .venv 內執行 `eval_sam2_failure_cases.py --model_type sam2_hiera_small --device cpu`（25 筆）、`generate_dl_vs_sam_comparison.py`，產出實際 SAM 2.0 指標與對比報告。
- **說明文件**：  
  - Overlay 顏色：紅色 = FP（預測為天空但 GT 非天空），藍色 = FN（GT 為天空但預測非天空）。  
  - 新增 `notes/2026-01-28-vlm-scenario-why-better-worse.md`，整理五情境中 VLM 變好／變爛原因，以及 sea-sky-confusable、heavy-occlusion 表現不佳之說明（提示策略與情境特性）。
- **產物**：`outputs/sam2_failure_cases_metrics.csv`、`outputs/sam2_overlays/`、`outputs/dl_vs_sam_comparison.csv`、`outputs/dl_vs_sam_comparison_report.md`，以及本則日誌與 VLM 情境說明筆記。
