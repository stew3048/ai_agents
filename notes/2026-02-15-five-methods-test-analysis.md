# 每日日誌 2026-02-15：五方法完整 test 跑完與三種分析寫入日記

---

## 0. 前提

- **測試集**：`outputs/test_list.txt`（完整 test list），五種方法皆跑同一測試集。
- **五種方法**：Grounding DINO + SAM、CLIPSeg、CLIPSeg 空間改善、DINO + Linear、DINO + dino_segmentation。
- **評估腳本**：`scripts/eval_five_methods_in_domain_test.py`；指標與昨日各方法腳本對齊（GT、IoU/FP/FN 公式一致）。
- **資料來源**：`outputs/eval_five_methods/per_image_metrics.csv`（僅統計有 GT 的圖）；分析腳本 `scripts/analyze_five_methods_metrics.py` 產出三種分析。

---

## 1. 設定摘要

- **Test 名單**：`load_list_file(outputs/test_list.txt)`，與 train_in_domain / U-Net eval 同一套。
- **Overlay**：僅對 overall best_10 / worst_10 輸出，目錄 `outputs/eval_five_methods/overlays/`，各方法分子目錄。
- **輸出**：每方法 `*_test_best_worst_metrics.json`、合併 `per_image_metrics.csv`；分析報告見 §2。

---

## 2. 測試（Test）結果：三種分析

### 2.1 分析一：每方法整體平均（IoU / FP rate / FN rate）

| 方法 | n 張圖 | mean IoU | mean FP rate | mean FN rate |
|------|--------|----------|--------------|--------------|
| **DINO-dino_segmentation** | 116 | **0.9204** | **0.0105** | **0.0506** |
| DINO-Linear | 116 | 0.8363 | 0.0242 | 0.1061 |
| CLIPSeg_spatial_improved | 116 | 0.7566 | 0.0474 | 0.1021 |
| CLIPSeg | 116 | 0.7514 | 0.0835 | 0.1108 |
| GroundingDINO-SAM | 106 | 0.7282 | 0.1696 | 0.0124 |

- **IoU 最高**：DINO-dino_segmentation（0.92）；**FP 最低**亦為 DINO-dino_segmentation（0.01）。
- **FN 最低**：GroundingDINO-SAM（0.01）；CLIPSeg 兩支 FN 較高（約 0.10–0.11）。
- Grounding 僅 106 張（部分 camera 無 sky / 無 GT），整體 FP 最高（0.17）。

---

### 2.2 分析二：每方法 × 每 camera（各方法在不同 camera 上的表現）

**CLIPSeg**

| camera | n | iou | fp_rate | fn_rate |
|--------|---|-----|---------|---------|
| 1093 | 9 | 0.8675 | 0.0047 | 0.1238 |
| 3888 | 21 | 0.7524 | 0.0919 | 0.0577 |
| 4795 | 32 | 0.7337 | 0.0089 | 0.1515 |
| 9112 | 10 | 0.7799 | 0.0153 | 0.2035 |
| 9291 | 9 | 0.6314 | 0.1652 | 0.0638 |
| 9483 | 10 | 0.8734 | 0.0078 | 0.1173 |
| 10066 | 5 | 0.8324 | 0.0551 | 0.1199 |
| 10870 | 10 | 0.4695 | 0.5438 | 0.1294 |
| 21444 | 10 | 0.9000 | 0.0000 | 0.0000 |

**CLIPSeg_spatial_improved**

| camera | n | iou | fp_rate | fn_rate |
|--------|---|-----|---------|---------|
| 1093 | 9 | 0.8720 | 0.0049 | 0.1190 |
| 3888 | 21 | 0.7648 | 0.0815 | 0.0551 |
| 4795 | 32 | 0.7649 | 0.0134 | 0.0949 |
| 9112 | 10 | 0.7637 | 0.0085 | 0.2276 |
| 9291 | 9 | 0.6349 | 0.1623 | 0.0619 |
| 9483 | 10 | 0.8708 | 0.0060 | 0.1220 |
| 10066 | 5 | 0.8590 | 0.0002 | 0.1408 |
| 10870 | 10 | 0.6026 | 0.1619 | 0.1821 |
| 21444 | 10 | 0.7000 | 0.0088 | 0.0000 |

**DINO-Linear**

| camera | n | iou | fp_rate | fn_rate |
|--------|---|-----|---------|---------|
| 1093 | 9 | 0.8095 | 0.0033 | 0.1840 |
| 3888 | 21 | 0.8756 | 0.0278 | 0.0417 |
| 4795 | 32 | 0.7911 | 0.0025 | 0.1634 |
| 9112 | 10 | 0.8515 | 0.1065 | 0.0319 |
| 9291 | 9 | 0.7600 | 0.0470 | 0.1116 |
| 9483 | 10 | 0.9347 | 0.0177 | 0.0449 |
| 10066 | 5 | 0.6736 | 0.0062 | 0.3210 |
| 10870 | 10 | 0.7954 | 0.0415 | 0.1167 |
| 21444 | 10 | 1.0000 | 0.0000 | 0.0000 |

**DINO-dino_segmentation**

| camera | n | iou | fp_rate | fn_rate |
|--------|---|-----|---------|---------|
| 1093 | 9 | 0.8984 | 0.0040 | 0.0936 |
| 3888 | 21 | 0.9231 | 0.0162 | 0.0259 |
| 4795 | 32 | 0.9333 | 0.0010 | 0.0469 |
| 9112 | 10 | 0.9205 | 0.0371 | 0.0347 |
| 9291 | 9 | 0.8594 | 0.0286 | 0.0506 |
| 9483 | 10 | 0.9736 | 0.0082 | 0.0163 |
| 10066 | 5 | 0.7277 | 0.0020 | 0.2709 |
| 10870 | 10 | 0.9117 | 0.0086 | 0.0667 |
| 21444 | 10 | 1.0000 | 0.0000 | 0.0000 |

**GroundingDINO-SAM**（無 21444，因該 camera 多無 sky）

| camera | n | iou | fp_rate | fn_rate |
|--------|---|-----|---------|---------|
| 1093 | 9 | 0.8303 | 0.0205 | 0.0255 |
| 3888 | 21 | 0.6413 | 0.2763 | 0.0024 |
| 4795 | 32 | 0.7524 | 0.0514 | 0.0101 |
| 9112 | 10 | 0.9609 | 0.0083 | 0.0297 |
| 9291 | 9 | 0.6105 | 0.2106 | 0.0072 |
| 9483 | 10 | 0.9688 | 0.0088 | 0.0211 |
| 10066 | 5 | 0.7719 | 0.4554 | 0.0177 |
| 10870 | 10 | 0.3523 | 0.6009 | 0.0046 |

- **CLIPSeg 在 10870 明顯較差**：iou=0.47, fp=0.54（原始版）；改善版 iou=0.60, fp=0.16。
- **Grounding 在 10870、10066 的 FP 很高**：10870 fp=0.60、10066 fp=0.46。
- **DINO-dino_segmentation 在各 camera 普遍 IoU > 0.85**，僅 10066 略低（0.73）、FN=0.27。
- **21444**：CLIPSeg 為 iou=0.9/0.7、DINO 兩支為 1.0（無天空或極易）。

---

### 2.3 分析三：每 camera 跨方法平均（該 camera 預測難度：IoU 低 / FP 或 FN 高 = 較難）

| 難度排序 | camera | mean IoU | mean FP rate | mean FN rate | 說明 |
|----------|--------|----------|--------------|--------------|------|
| 1 最難 | **10870** | 0.6263 | 0.2713 | 0.0999 | IoU 最低、FP 最高 |
| 2 | 9291 | 0.6992 | 0.1228 | 0.0590 | |
| 3 | 10066 | 0.7729 | 0.1038 | **0.1741** | FN 最高（易漏判天空） |
| 4 | 3888 | 0.7914 | 0.0987 | 0.0366 | |
| 5 | 4795 | 0.7951 | 0.0154 | 0.0933 | |
| 6 | 9112 | 0.8553 | 0.0351 | 0.1055 | |
| 7 | 1093 | 0.8556 | 0.0075 | 0.1092 | |
| 8 | 21444 | 0.9000 | 0.0022 | 0.0000 | 多為無天空，易預測 |
| 9 最易 | 9483 | 0.9243 | 0.0097 | 0.0643 | |

- **預測難度最高**：**camera 10870**（平均 IoU 最低、FP 最高）。
- **10066**：平均 **FN 最高**（多種方法易漏判天空）。
- **21444、9483**：整體最容易（IoU 高、FP/FN 低）。

---

## 3. 觀察與分析

- **整體排序（mean IoU）**：DINO-dino_segmentation > DINO-Linear > CLIPSeg 空間改善 > CLIPSeg > GroundingDINO-SAM；兩支 DINO in-domain 明顯優於 zero-shot。
- **Zero-shot 弱點**：Grounding 在 10870、10066 FP 高；CLIPSeg 在 10870 的 IoU 低、FP 高，空間改善版有提升但仍不如 DINO。
- **Camera 難度**：10870 為各方法共同難點（可對照日後 camera_inventory 的 night_ratio／亮度）；10066 以 FN 高為主，適合作為漏判改善的目標 camera。

---

## 4. 下一步改善方向

- 針對 **10870**：可檢視該 camera 場景（夜間／低對比）是否需資料增強或專案 fine-tune。
- 針對 **10066**：聚焦降低 FN（漏判天空），可從 DINO 解碼器或 loss 權重著手。
- 分析腳本可重複執行；若更新 `per_image_metrics.csv` 後再跑 `analyze_five_methods_metrics.py` 即可更新三種分析。

---

## 5. 輸出檔與指令

| 類型 | 路徑 |
|------|------|
| 五方法 JSON | `outputs/eval_five_methods/*_test_best_worst_metrics.json` |
| Per-image CSV | `outputs/eval_five_methods/per_image_metrics.csv` |
| 分析報告（文字） | `outputs/eval_five_methods/five_methods_analysis_report.txt` |
| 分析 JSON | `outputs/eval_five_methods/five_methods_analysis.json` |
| 分析摘要 | `outputs/eval_five_methods/five_methods_analysis_summary.md` |
| Overlays | `outputs/eval_five_methods/overlays/{GroundingDINO-SAM,CLIPSeg,...}/` |

**重新跑五方法測試**：
```bash
.\.venv\Scripts\python.exe scripts/eval_five_methods_in_domain_test.py --overlay_dir outputs/eval_five_methods/overlays
```

**只跑分析（不重跑推論）**：
```bash
.\.venv\Scripts\python.exe scripts/analyze_five_methods_metrics.py --csv outputs/eval_five_methods/per_image_metrics.csv --out_dir outputs/eval_five_methods
```

---

## 6. 今日結論

五種方法已在完整 test list 上跑完，並完成三種分析寫入本日誌：**(1) 每方法整體平均 IoU/FP/FN**、(2) **每方法 × 每 camera 的 IoU/FP/FN**、(3) **每 camera 跨方法平均（預測難度排序）**。整體最佳為 DINO-dino_segmentation（mean IoU 0.92）；最難 camera 為 10870（IoU 低、FP 高），10066 則 FN 最高、適合作為漏判改善目標。
