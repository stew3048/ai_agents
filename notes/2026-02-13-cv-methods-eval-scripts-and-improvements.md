# 2026-02-13 CV 方法評估腳本（本機/Colab）與 Grounding DINO+SAM2 / CLIPSeg 改善比較

> 今日重點：產出本機與 Colab 皆可用的評估腳本對照、完成 Grounding DINO+SAM2 與 CLIPSeg 的改善項實作與測試比較。

---

## 0. 前提：資料與環境

### 0.1 Camera Inventory（來自 `outputs/camera_inventory.csv`）

| Camera | 張數 | 夜晚比例 (night_ratio) | 平均亮度 (mean_brightness) | 狀態 |
|--------|------|------------------------|----------------------------|------|
| 10066 | 50 | 0.02 | 81.8 | ready |
| 10870 | 94 | 0.54 | 39.2 | ready |
| 1093 | 90 | 0.18 | 59.7 | ready |
| 9112 | 100 | 0.48 | 52.8 | ready |
| 9291 | 82 | 0.64 | 39.5 | ready |
| 9483 | 100 | 0.22 | 94.7 | ready |
| 10917 / 9708 | 0 | - | - | downloading / no_images |

### 0.2 評估用 Test 清單

- **固定 test 清單**：`outputs/test_list.txt`（每行一個相對路徑，如 `skyfinder_10870/images/xxx.jpg`）
- **In-domain splits**：`outputs/in_domain_splits_summary.json`（train 2535 / val 315 / test 320）
- **Cross-camera**：`outputs/multi_camera_splits.json`（Test camera 10870）

---

## 1. 本機 / Colab 皆可用的評估腳本對照

### 1.1 腳本對應表

| 方法 | 本機（專案根目錄執行） | Colab（同上，或掛載後執行） |
|------|------------------------|-----------------------------|
| **CLIPSeg** | `python scripts/eval_clipseg_final.py` | `python colab/eval_clipseg.py` |
| **Grounding DINO + SAM2** | `python scripts/eval_grounding_dino_sam_final.py` | `python colab/eval_grounding_dino_sam.py` |
| **DL (U-Net)** | `python scripts/eval_dl_final.py` | `python colab/eval_dl.py` |
| **DINO-DL** | `python scripts/eval_dino_dl_final.py` | `python colab/eval_dino_dl.py` |
| **統一入口（多模型）** | - | `python colab/eval_all.py --models clipseg,grounding-dino-sam` 或 `--models all` |

### 1.2 共通參數（本機與 Colab 腳本皆支援）

- `--test_list`：test 清單路徑（預設 `outputs/test_list.txt` 或 Colab 的 `{outputs_dir}/test_list.txt`）
- `--data_dir` / `--base_data_dir`：資料根目錄（預設 `data`）
- `--output_csv` / `--metrics_csv`：輸出的 metrics CSV（可追加同一檔，方便彙總）
- `--overlay_dir`：overlay 圖輸出目錄（可選）
- `--device`：`cuda` / `cpu`（Colab 通常自動偵測 GPU）

### 1.3 本機快速執行範例（專案根目錄）

```bash
# CLIPSeg（改善版：動態閾值 + 空間先驗 + OPEN + 多 prompt）
python scripts/eval_clipseg_final.py --test_list outputs/test_list.txt --output_csv outputs/metrics_summary_test.csv

# Grounding DINO + SAM2
python scripts/eval_grounding_dino_sam_final.py --test_list outputs/test_list.txt --output_csv outputs/metrics_summary_test.csv
```

### 1.4 本機/Colab 共用入口（推薦）

專案根目錄執行下列指令即可，會自動依環境選擇 `scripts/` 或 `colab/` 腳本：

```bash
# CLIPSeg + Grounding DINO-SAM（預設兩者都跑）
python scripts/run_eval_cv_methods.py --models clipseg,grounding-dino-sam --test_list outputs/test_list.txt --output_csv outputs/metrics_summary_test.csv

# 只跑 CLIPSeg
python scripts/run_eval_cv_methods.py --models clipseg

# 強制用本機腳本（例如在 Colab 但想用 final 版邏輯）
python scripts/run_eval_cv_methods.py --env local --models clipseg,grounding-dino-sam
```

CLIPSeg 改善項消融（僅本機 final 支援）：  
`--no_dynamic_threshold`、`--no_spatial_prior`、`--no_morphology_open`、`--no_multi_prompt_contrast`。

### 1.5 Colab 直接跑 Colab 腳本

在 Colab 中將專案放在當前目錄（或掛載），在專案根目錄執行：

```bash
# 僅 CLIPSeg 與 Grounding DINO-SAM
python colab/eval_all.py --models clipseg,grounding-dino-sam --test_list outputs/test_list.txt --output_csv outputs/metrics_summary_test.csv
```

或分別執行：  
`python colab/eval_clipseg.py ...`、`python colab/eval_grounding_dino_sam.py ...`。

---

## 2. Grounding DINO + SAM2 測試與改善

### 2.1 流程摘要

1. **Grounding DINO**：以對比式文字 prompt（`sky . sea . ocean . building . mountain . tree . ground .`）偵測「sky」對應的 bounding box。
2. **SAM 2.0**：將上述 box 作為 prompt 輸入 SAM2，產出分割 mask。
3. **後處理**：若偵測到的 sky 區域過小（如 < 0.5% 圖面積）則視為誤報，輸出全黑；可選捨棄畫面過下方的 box（避免地面被當成 sky）。

### 2.2 改善要點

- **對比式 prompt**：多類別並列，只保留 label 為 "sky" 的 box，降低 sea/building 等誤檢。
- **SAM2 型號**：可選 `sam2_hiera_tiny` / `small` / `base` / `large`（本機資源有限可用 tiny）。
- **防誤報**：`MIN_SKY_AREA_RATIO`、底部區域過低之 box 捨棄。

### 2.3 測試結果摘要（參考）

- 評估結果會追加至 `outputs/metrics_summary_test.csv`（或指定 CSV），欄位含 `method,camera_id,num_images,IoU_mean,FP_mean,FN_mean`。
- **GroundingDINO-SAM** 在不同 camera 上 IoU 差異大（例如 10870 夜間多、易漏檢），可對照 camera inventory 的 `night_ratio`、`mean_brightness` 做分析。

### 2.4 小樣本 5 張測試：清單與結果

**使用清單**：`outputs/test_list_small.txt`（camera 10066×2、10870×2、1093×1，共 5 張）

| # | 相對路徑 | camera_id | 說明 |
|---|----------|-----------|------|
| 1 | `skyfinder_10066/images/046.jpg` | 10066 | 日間、森林/戶外 |
| 2 | `skyfinder_10066/images/047.jpg` | 10066 | 日間、森林/戶外 |
| 3 | `skyfinder_10870/images/091.jpg` | 10870 | 夜間比例高 camera |
| 4 | `skyfinder_10870/images/092.jpg` | 10870 | 夜間比例高 camera |
| 5 | `skyfinder_1093/images/082.jpg` | 1093 | 都市、日間為主 |

**測試指令（小樣本）**：

```bash
# CLIPSeg（改善版）
python scripts/eval_clipseg_final.py --test_list outputs/test_list_small.txt --output_csv outputs/metrics_summary_test.csv --overlay_dir outputs/clipseg_small_overlays

# Grounding DINO + SAM2
python scripts/eval_grounding_dino_sam_final.py --test_list outputs/test_list_small.txt --output_csv outputs/metrics_summary_test.csv --overlay_dir outputs/gdino_sam_small_overlays
```

**小樣本 5 張的測試結果（CLIPSeg vs GroundingDINO-SAM）**：

| camera_id | num_images | 指標 | CLIPSeg | GroundingDINO-SAM |
|-----------|------------|------|---------|-------------------|
| **10066** | 2 | IoU_mean | **0.794** | 0.692 |
| | | FP_mean | **0.022** | 0.695 |
| | | FN_mean | 0.188 | **0.022** |
| **10870** | 2 | IoU_mean | **0.254** | 0.265 |
| | | FP_mean | 0.940 | *(異常)* 約 117000 |
| | | FN_mean | 0.084 | **0.000** |
| **1093** | 1 | IoU_mean | **0.898** | 0.877 |
| | | FP_mean | **0.004** | 0.057 |
| | | FN_mean | 0.094 | **0.021** |

**截圖／視覺化**：

- 比較報告（表格與簡要結論）：`outputs/comparison_clipseg_vs_grounding_dino.md`
- 若執行時有加 `--overlay_dir`，overlay 圖會寫入該目錄（原圖 + 預測 mask + GT 疊圖），檔名對應各張影像路徑，可作為測試截圖留存。

**簡要結論（此 5 張）**：CLIPSeg 在 10066、1093 的 IoU 與 FP 較優；GroundingDINO-SAM 的 FN 普遍較低（漏判少）。10870 上 GroundingDINO-SAM 的 FP_mean 數值異常（疑為評估腳本 bug，應為 rate 卻輸出成絕對像素數），待後續確認。

---

## 3. CLIPSeg 改善前後比較

### 3.1 改善項目（預設全開）

| 改善項 | 說明 | 腳本參數（關閉時） |
|--------|------|---------------------|
| **動態閾值** | 依 heatmap 的 max/mean 計算二值化門檻（0.4×max + 0.6×mean，限制在 [0.25, 0.75]），取代固定 0.5 | `--no_dynamic_threshold` |
| **空間先驗** | 上半部權重 1.2、下半部 0.8，天空較常出現在上方 | `--no_spatial_prior` |
| **形態學 OPEN** | 以橢圓 kernel 做 OPEN 運算，移除細碎雜訊（降 FP） | `--no_morphology_open` |
| **多 prompt 對比** | 同時預測 "sky" 與 "building"，若 building > sky 則該像素強制為 0 | `--no_multi_prompt_contrast` |

### 3.2 如何做「改善前」vs「改善後」比較

- **改善後（預設）**：直接跑  
  `python scripts/eval_clipseg_final.py`  
  或 Colab：`python colab/eval_clipseg.py`（若 Colab 版已同步上述四項邏輯）。
- **改善前（關閉所有改善）**：  
  `python scripts/eval_clipseg_final.py --no_dynamic_threshold --no_spatial_prior --no_morphology_open --no_multi_prompt_contrast`  
  等同固定閾值 0.5、無空間加權、無 OPEN、僅單一 "sky" prompt。
- **單項消融**：每次只關一個，例如只關動態閾值：  
  `--no_dynamic_threshold`  
  比較同一 test_list 下 IoU / FP / FN 差異。

### 3.3 預期趨勢（經驗）

- 動態閾值：不同亮度/場景下較穩定，避免固定 0.5 在極亮/極暗時偏掉。
- 空間先驗：減少下方建築/地面被標成天空（降 FP）。
- 形態學 OPEN：減少小塊雜訊（降 FP）。
- 多 prompt 對比：建築物區域更不易被標成天空（降 FP）。

實際數字以同一 `test_list`、同一 `output_csv` 跑兩次（改善前/後）比對即可。

---

## 4. 輸出檔與指令整理

### 4.1 輸出檔

| 產物 | 說明 |
|------|------|
| `outputs/metrics_summary_test.csv` | 各 method 的 per-camera IoU/FP/FN 摘要（可多種方法追加同一檔） |
| `outputs/camera_inventory.csv` | 各 camera 張數、night_ratio、mean_brightness（日誌前提用） |
| `outputs/test_list.txt` | 固定 test 清單 |
| `scripts/aggregate_final_metrics.py` | 彙總 metrics_summary CSV、顯示摘要 |

### 4.2 常用指令

```bash
# 更新 camera inventory（實驗前建議跑一次）
python scripts/camera_inventory.py

# CLIPSeg 改善版
python scripts/eval_clipseg_final.py --test_list outputs/test_list.txt --output_csv outputs/metrics_summary_test.csv

# CLIPSeg 改善前（消融）
python scripts/eval_clipseg_final.py --no_dynamic_threshold --no_spatial_prior --no_morphology_open --no_multi_prompt_contrast --output_csv outputs/metrics_summary_test.csv

# Grounding DINO + SAM2
python scripts/eval_grounding_dino_sam_final.py --test_list outputs/test_list.txt --output_csv outputs/metrics_summary_test.csv

# 本機/Colab 共用入口（一次跑 CLIPSeg + Grounding DINO-SAM）
python scripts/run_eval_cv_methods.py --models clipseg,grounding-dino-sam --test_list outputs/test_list.txt --output_csv outputs/metrics_summary_test.csv

# 彙總 CSV 摘要
python scripts/aggregate_final_metrics.py outputs/metrics_summary_test.csv
```

---

## 5. 今日結論

- **腳本**：本機使用 `scripts/eval_*_final.py`，Colab 使用 `colab/eval_*.py`；參數一致（test_list、data_dir、output_csv、overlay_dir），Colab 另有統一入口 `colab/eval_all.py --models ...`，本機/Colab 皆可依同一流程跑評估。
- **Grounding DINO + SAM2**：已具對比式 prompt、SAM2 分割與防誤報邏輯；測試結果寫入 metrics CSV，可依 camera 的 night_ratio / 亮度解讀差異。
- **CLIPSeg**：實作並測試四項改善（動態閾值、空間先驗、形態學 OPEN、多 prompt sky vs building），改善前後可透過 `--no_*` 開關做消融比較，同一 test_list 即可重現改善前後差異。

晚安。
