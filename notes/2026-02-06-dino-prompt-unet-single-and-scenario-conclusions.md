# 2026-02-06 DINO prompt、U-Net 單張測試與情境結論

> 今日重點：DINO 當 SAM prompt 實驗、U-Net 單張 IoU/FP/FN、天海一線／fog 與 night 情境的結論整理。

---

## 0. 前提

- **延續**：先前已有 DL vs SAM（Legacy prompt）依五大情境與依天氣對比；互動式 SAM、依天氣分情境報告均已完成。
- **今日範圍**：DINO 視覺相似性當 SAM prompt、單張 U-Net 評估、情境難度與架構差異的討論。

---

## 1. 實驗與產物

### 1.1 DINO 當 SAM prompt（25 failure cases）

- **實作**：`scripts/dino_sky_prompt.py` 用 DINOv2 取 patch 特徵，以圖上方 25% 當 sky prototype，找最像天空的 patch 中心當 SAM 的單點 prompt；`eval_sam2_failure_cases.py --prompt dino` 產出 `sam2_failure_cases_metrics_dino.csv`。
- **比較**：`scripts/compare_sam_legacy_vs_dino.py` 對比 Legacy vs DINO。
- **結果**：DINO 整體**較 Legacy 差**（整體 IoU 0.31 vs 0.45，DINO 的 FN 很高 0.61），各情境除 night 外 DINO 皆較差或持平。目前「圖上方 25% 當 prototype、單點」策略在 failure cases 上不如既有 heuristics。

### 1.2 U-Net 單張評估（與 SAM 比較用同一模型）

- **腳本**：`scripts/eval_unet_single_image.py`，讀取 `train_multi_camera_*/checkpoints/best.pth`，對單張圖輸出 IoU / FP rate / FN rate。
- **今日跑過**：
  - **3888 / 95**：IoU 0.5256，FP 0.2652，FN 0.0025
  - **9291 / 8**：IoU 0.5651，FP 0.2076，FN 0
  - **9291 / 6**：IoU 0.5692，FP 0.2041，FN 0

### 1.3 情境觀察（9291/6、9291/8）

- 使用者觀察：這兩張在 **DL（U-Net）與 VLM（SAM）表現接近**，且在天海一線與 fog 情境下**兩者都差**。
- 對照：依天氣報告中 fog 情境 SAM 平均 IoU 低於 DL；sea-sky 情境兩者接近或 SAM 略差。與「天海一線、fog 都難」一致。

---

## 2. 今日結論（情境難度 vs 架構）

### 2.1 天海一線與 fog：情境本身難度

- **不是單一架構的缺陷**：DL 與 VLM 在這兩種情境下表現接近且都不理想。
- **原因**：天空與海／霧在**顏色、紋理上相似**，邊界本身曖昧，不論是逐像素分類（DL）或由點擴張（SAM）都難以穩定畫出界線。
- **小結**：可視為**情境本身的視覺歧義**導致的高難度，而非「換成 VLM 就會好」。

### 2.2 為何 night 時 VLM 比 DL 好？

- **Night 的難點**：天空**外觀**暗、對比低，DL 靠顏色/亮度容易漏檢（FN 高）、邊界不穩。
- **VLM 的優勢**：我們給的是**空間先驗**（點在上方 ≈ 天空）；SAM 做的是「從這點擴出去的連通區域」，不必從外觀「認出天空」。因此 night 時**結構先驗**彌補了**外觀難認**。
- **對比**：天海一線／fog 的難點是**外觀歧義**（天空與非天空長得像），點的先驗無法化解「哪裡是邊界」，所以 DL 與 VLM 都受情境限制。

### 2.3 一句話

- **天海一線、fog**：情境本身難度高，DL 與 VLM 表現接近且都不佳。
- **Night**：難在外觀難認，VLM 的空間先驗（點在上方）彌補了這點，故 VLM 較 DL 好。

---

## 3. 輸出檔與指令

| 產物 | 說明 |
|------|------|
| `scripts/dino_sky_prompt.py` | DINOv2 找最像天空的點，回傳 SAM point prompt |
| `scripts/eval_unet_single_image.py` | U-Net 單張評估，輸出 IoU / FP / FN |
| `scripts/compare_sam_legacy_vs_dino.py` | Legacy vs DINO prompt 比較報告 |
| `outputs/sam2_failure_cases_metrics_dino.csv` | DINO prompt 的 25 筆 SAM 結果 |
| `outputs/sam2_overlays_dino/` | DINO prompt 的 overlay |

**單張 U-Net 指令**（與 SAM 比較用同一 checkpoint）：
```batch
cd c:\Users\yiching\ai_projects
.venv\Scripts\activate
python scripts/eval_unet_single_image.py --camera_id 9291 --image_id 6
```

---

## 4. 今日結論（摘要）

- DINO 當 SAM prompt 在 25 failure cases 上整體不如 Legacy heuristics，FN 偏高。
- 以 U-Net 單張評估 3888/95、9291/8、9291/6，確認 9291 兩張在 DL 與 VLM 表現接近。
- **天海一線與 fog**：屬情境本身難度，DL 與 VLM 皆受限；**night**：VLM 因空間先驗（點在上方）優於 DL。以上結論已整理於本則日誌。
