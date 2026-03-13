# 2026-02-05 今晚計畫（約 2 小時）

> 目標：釐清「遮擋／天海一線情境 SAM 較差」是**提示策略問題**還是**SAM 本身限制**，並試跑互動式 SAM。

---

## 背景

- **已做**：CV、U-Net、SAM 用同一批 test / failure cases 跑過；SAM 在 **heavy-occlusion**、**sea-sky-confusable** 明顯較差。
- **不確定**：是 SAM 在這兩種情境本來就弱，還是我們**自動給的 box/點**不適合，所以還不想下結論。
- **已備好**：互動式 SAM 腳本 `scripts/sam2_interactive_one_image.py`（你手動點「這裡是天空」再按 P 預測）。

---

## 今晚 2 小時要做的事

### 1. 跑互動式 SAM（約 30–50 分鐘）

- **目的**：用「你點的提示」取代自動 box/點，看同一張圖上手動點是否明顯比自動好。
- **做法**：
  - **方式 A**：在專案根目錄雙擊或執行 `scripts\run_interactive_sam_tonight.bat`（會依序開兩張圖）。
  - **方式 B**：手動在 PowerShell 執行（先 `cd 專案根目錄`）：
    ```powershell
    .\.venv\Scripts\Activate.ps1
    python scripts/sam2_interactive_one_image.py --camera_id 4795 --image_id 9
    ```
    第一張結束（按 Q）後再跑：
    ```powershell
    python scripts/sam2_interactive_one_image.py --camera_id 3888 --image_id 429
    ```
  - 第一張（4795/9）：在圖上**只點明顯是天空**的幾點（避開建築頂），按 **P** 看預測 overlay，必要時按 **C** 重點再按 **P**，最後按 **Q** 關閉。
  - 第二張（3888/429）：在**天空區**點幾點（盡量不點到海），按 **P** 看結果，按 **Q** 關閉。
- **產物**：`outputs/sam2_interactive/` 會存 overlay；可和 `outputs/sam2_overlays/` 裡同張圖的**自動提示**結果比對。

### 2. 比對「手動 vs 自動」並寫結論（約 20–30 分鐘）

- 看兩張圖的互動 overlay 與原本 SAM overlay（紅/藍/綠分布）。
- 若**手動點**明顯較準（紅藍變少、更貼 GT）→ 偏向**提示策略問題**，之後可調自動 box/點或做「半自動」。
- 若手動點也差不多差 → 偏向**該情境下 SAM 本身限制**，結論可寫「在此任務/情境下 SAM 不適合」或需別種用法（例如多張圖、別種模型）。

### 3. 寫進筆記（約 10 分鐘）

- 在本則或同一份 note 底下補：
  - 試了哪兩張圖、手動怎麼點（簡述）。
  - 手動 vs 自動的觀感或簡單比較（有算 IoU 就寫數字）。
  - **暫時結論**：遮擋／天海一線是「提示問題」還是「SAM 限制」，以及下一步（例如：改自動提示再跑、或接受結論並整理方法適用情境表）。

### 4. 若還有時間（可選）

- 把「CV / U-Net / SAM 各適用情境」整理成一小段或表格，放在同一份 note 或 `2026-02-04-vlm-scenario-why-better-worse.md` 的「後續」裡，方便之後寫報告或投影片。

---

## 不排進今晚的（之後再做）

- 改自動提示（例如 sea-sky 縮小 box、heavy-occlusion 改 box）並重跑整批 25 張。
- 用 sam2_hiera_large 或其它模型再比一輪。
- 擴充到更多張互動測試。

---

## 一句話

**今晚重點**：用互動 SAM 試「遮擋一張 + 天海一線一張」，比對手動 vs 自動結果，寫出「是提示問題還是 SAM 限制」的暫時結論，並記在筆記裡。

---

## 實際執行結果

- **只顯示圖片（--no-sam）**：✅ 成功，視窗與圖片路徑正常（Matplotlib 後端 Qt5Agg）。
- **含 SAM 的步驟**：❌ 在本機載入 torch 時出現 `[WinError 1114]`（c10.dll 初始化失敗），無法跑互動 SAM。
- **結論**：互動式「手動點 vs 自動提示」比對尚未能在本機完成。可改在**他機**或**修好 torch 環境**後再跑同一支腳本。目前仍可依既有自動提示的對比報告（`outputs/dl_vs_sam_comparison_report.md`）與 `2026-02-04-vlm-scenario-why-better-worse.md` 下**暫時結論**：遮擋／天海一線較差主要與**提示策略**有關（點或框落在非天空或含海）；若日後能跑互動 SAM，再用手動點驗證。
