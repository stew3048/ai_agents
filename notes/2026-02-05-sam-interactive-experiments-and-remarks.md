# 2026-02-05 互動式 SAM 實驗與實驗備註（experiments.csv remark）

> 使用 `scripts/sam2_interactive_one_image.py` 在單張圖（camera 4795, image 9）上手動點選不同區域，觀察 SAM 2.0 預測結果並記錄於 `outputs/sam2_interactive/experiments.csv`（含 Remark 欄）。

---

## 0. 前提

- **今日焦點**：互動式 SAM——手動點「這裡是天空」取代自動 box/點，釐清遮擋／天海一線情境較差是**提示策略**還是**模型限制**。
- **測試圖**：camera_id=4795, image_id=9（`009.jpg`），為 failure case 之一（重遮擋／建築物邊界）。
- **實驗記錄**：每次按 P 預測並儲存 overlay 後，腳本會自動將該次實驗名稱、IoU、FP_rate、FN_rate 等寫入 `outputs/sam2_interactive/experiments.csv`；你另在 CSV 中加了 **Remark** 欄作為備註參考。

---

## 1. 設定摘要

| 項目 | 內容 |
|------|------|
| **腳本** | `scripts/sam2_interactive_one_image.py` |
| **模型** | sam2_hiera_small（預設） |
| **裝置** | CPU（預設） |
| **圖片** | camera_id=4795, image_id=9 → `data/skyfinder_4795/images/009.jpg` |
| **操作** | 左鍵點擊加入「天空」點 → **P** 預測並存 overlay → **C** 清除重點 → **Q** 結束 |

---

## 2. 實驗結果（來自 experiments.csv + Remark）

以下五筆為同一張圖（009.jpg）上不同點選策略的結果；**Remark** 依你填寫的備註整理（CSV 若為 Big5 存檔可能顯示亂碼，此處以語意還原）。

| time | experiment | IoU | FP_rate | FN_rate | Remark（參考） |
|------|------------|-----|---------|---------|----------------|
| 2026/2/5 23:16 | pointToSky | **0.9558** | 0.0006 | 0.0319 | 點天空就真的點到天空 |
| 2026/2/5 23:18 | pointToGround | 0 | 0.2346 | 1 | 點地可以點出旁邊的地 |
| 2026/2/5 23:20 | pointToWindow | 0 | 0.0368 | 1 | 點窗戶可以點出旁邊的窗戶 |
| 2026/2/5 23:22 | pointToLight | 0 | 0.1221 | 1 | 點路燈點的區域比較發散 |
| 2026/2/5 23:26 | pointBuilding | 0.0004 | 0.0331 | 0.9993 | 點建築物可以把旁邊的輪廓點出來，雖然有點偏還是會點成天空 |

- **pointToSky**：點在明確天空處 → IoU 高、FP/FN 低，與 GT 幾乎一致。
- **pointToGround / pointToWindow / pointToLight**：點在非天空物體 → 預測為該物體或鄰近區域，天空漏檢（FN=1 或極高），FP 視區域而定。
- **pointBuilding**：點建築物 → 能分出建築輪廓，但易偏成天空（FN 仍高），與 Remark「有點偏還是會點成天空」一致。

---

## 3. 觀察與分析

- **提示決定結果**：同一張圖上，**點哪裡就分割哪裡**；點天空則天空分割正確，點地/窗/路燈/建築則 SAM 跟隨提示，不會自動「整張圖的天空」。
- **與自動提示的對照**：先前自動給 box/點在遮擋情境常落在建築或邊界，導致 SAM 表現差；手動點明確天空可達 IoU≈0.96，支持「**遮擋情境較差主要與提示策略有關**」。
- **Remark 的價值**：在 experiments.csv 加 Remark 有助之後回顧每筆實驗「當時在點什麼、觀察到什麼」，方便寫報告或改自動提示策略。

---

## 4. 下一步改善方向

- 若要以 SAM 改善此類 failure case：可考慮**自動提示**改為「優先落在明確天空區」或縮小 box 避開建築邊界，再重跑整批 25 張比對。
- 將「CV / U-Net / SAM 各適用情境」整理成表格，放在 `2026-02-04-vlm-scenario-why-better-worse.md` 或報告用 note。

---

## 5. 輸出檔與指令

### 輸出檔

- **實驗記錄**：`outputs/sam2_interactive/experiments.csv`（含 time, camera_id, image_id, experiment, overlay_file, image, IoU, FP_rate, FN_rate, **Remark**）
- **Overlay 圖**：`outputs/sam2_interactive/009_interactive_overlay_pointToSky.png` 等（依 experiment 或 `--suffix` 命名）

### 使用 Anaconda Prompt 執行的指令

在 **Anaconda Prompt** 中，先進入專案目錄並啟動環境，再執行互動式 SAM（若本機 PowerShell 遇 torch DLL 錯誤，可改用 Anaconda 環境）：

```batch
cd c:\Users\yiching\ai_projects
conda activate .venv
python scripts/sam2_interactive_one_image.py --camera_id 4795 --image_id 9
```

- 若要自訂 overlay 檔名後綴（例如存成 `009_interactive_overlay_Test1.png`）：

```batch
python scripts/sam2_interactive_one_image.py --camera_id 4795 --image_id 9 --suffix Test1
```

- 只顯示圖片、不載入 SAM（測試視窗與路徑是否正常）：

```batch
python scripts/sam2_interactive_one_image.py --camera_id 4795 --image_id 9 --no-sam
```

- 換第二張圖（例如 camera 3888, image 429）：

```batch
python scripts/sam2_interactive_one_image.py --camera_id 3888 --image_id 429
```

---

## 6. 今日結論

- 在 camera 4795 / image 9 上做了五種手動點選實驗並記錄於 **experiments.csv**，**Remark 欄**已作為各筆實驗的備註參考。
- **點天空**時 SAM 表現佳（IoU≈0.96）；點地/窗/路燈/建築則跟隨提示分割該區域，天空漏檢高，印證「**提示策略**」對結果影響大。
- 日後可依此表與 overlay 比對自動提示，或整理成方法適用情境表供報告使用。
