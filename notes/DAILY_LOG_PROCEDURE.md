# 每日日誌流程（Daily Log Procedure）

> 當你為一整天做結尾、要寫當天日誌時，依本流程產出 `notes/{YYYY-MM-DD}-{描述}.md`，之後就不必重複說明。

---

## 1. 觸發時機與檔名

| 觸發 | 使用者可能說：「寫今日日誌」「為一整天做結尾」「生成今天日誌」等 |
|------|------------------------------------------------------------------|
| **檔名** | `notes/{YYYY-MM-DD}-{描述}.md`，日期以使用者提供的為準，`-` 後接描述（由 AI 依當日工作命名，kebab-case） |
| **範例** | `notes/2026-01-24-multi-camera-cross-camera-generalization.md` |

---

## 2. 實驗前：Camera Inventory 先行（務必記錄在「前提」）

在 **split 與訓練之前**，先跑：

```bash
python scripts/camera_inventory.py
```

- 產出：`outputs/camera_inventory.csv`（各 camera 的張數、night_ratio、mean_brightness、pairing_status 等）
- **日誌「前提」** 要從此表整理出：
  - 各 camera 的張數、**夜晚比例 (night_ratio)**、**平均亮度 (mean_brightness)**
  - Train / Val / Test 各來自哪些 camera、張數
- 這樣後面看訓練／測試結果時，才能回推：例如 Test camera 夜晚多、亮度低 → 易漏檢、domain shift 等。

---

## 3. 日誌結構（與 2026-01-24 範本一致）

依序寫入以下段落，並在過程裡點出**觀察到的問題**。

| 段落 | 內容 |
|------|------|
| **0. 前提** | 資料規模（Train/Val/Test 張數、來源 camera）、**各 camera 的白天／夜晚比例**（來自 `camera_inventory.csv`）、splits 設定 |
| **1. 設定摘要** | Splits、train/val/test cameras、訓練參數、checkpoint 路徑 |
| **2. 驗證（Val）結果** | 各 Epoch 的 Train/Sanity/Val 指標（含 **Val FP、FN**）、最佳 Val、**Val overlay 路徑**、**val_best_worst_metrics.json**、最好／最差 5 的 IoU／FP／FN |
| **3. 測試（Test）結果** | 整體 Test 指標（IoU、Dice、Pixel Acc、**FP、FN**）、Val vs Test、最好／最差 5（含 FP/FN）、**test_overlays/**、**test_best_worst_metrics.json** |
| **4. 觀察與分析** | Val／Test「訓練不好的情境」、domain shift、極端失敗案例；可對照前提中的 night_ratio、亮度 |
| **5. 下一步改善方向** | 依觀察提出：如增補夜晚／低光資料、augment、finetune 等 |
| **6. 輸出檔與指令** | 訓練日誌、best.pth、val_overlays、test_overlays、`val_best_worst_metrics.json`、`test_best_worst_metrics.json`、camera_inventory.csv、重新跑 Test 的指令 |
| **7. 今日結論** | 一至三句話總結當日結論（可附註來源如 ChatGPT 結論） |

---

## 4. Overlay 與 metrics 對照

- **Overlay 檔名**：使用 **split 編碼**（`val_0042`、`test_0073`），表示該 split 清單的第 42、73 筆；對照 `multi_camera_splits.json` 或 `orig_path` 即可找到原圖。
- **Val**：除 overlay 外，`val_overlays/val_best_worst_metrics.json` 記錄 best_5／worst_5 的 `rank`、`source_id`、`iou`、`fp_rate`、`fn_rate`、`orig_path`。
- **Test**：`test_overlays/test_best_worst_metrics.json` 同結構，供日誌與後續分析使用。

---

## 5. 參考範例

- **完整範例**：`notes/2026-01-24_multi-camera_cross-camera_generalization.md`  
  （若新檔名改為 `-` 連結，則為 `notes/2026-01-24-multi-camera-cross-camera-generalization.md`）

---

## 6. 與訓練流程的關係

| 階段 | 動作 |
|------|------|
| **實驗前** | 跑 `scripts/camera_inventory.py`，把前提（含各 camera 日夜比）留到日誌 |
| **訓練** | `train_multi_camera.py`：Val FP/FN、val overlay、`val_best_worst_metrics.json` |
| **Test** | `eval_multi_camera_test.py`：Test overlay、`test_best_worst_metrics.json` |
| **日誌** | 依 §3 產出 `notes/{YYYY-MM-DD}-{描述}.md`，含前提～訓練～評估～測試～觀察～分析～下一步～今日結論 |

---

*建立日期：2026-01-24*
