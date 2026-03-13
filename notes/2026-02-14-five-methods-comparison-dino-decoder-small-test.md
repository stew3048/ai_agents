# 每日日誌 2026-02-14：五種方法差異、DINO Decoder 訓練與架構、五張小樣本測試

---

## 0. 前提

- **小樣本測試集**：`outputs/test_list_small.txt`，共 **5 張**（10066×2、10870×2、1093×1）。
- **比較對象**：五種天空分割方法——三種 zero-shot（Grounding DINO + SAM、CLIPSeg 原始版、CLIPSeg 空間改善後）與兩種 **in-domain 訓練**（DINO + Linear、DINO + dino_segmentation）。
- **訓練資料**：兩支 DINO 模型皆使用同一 in-domain 設定——`outputs/train_list.txt`（2535 張）、`outputs/val_list.txt`（315 張）；Backbone 為 DINOv2 ViT-S/14（凍結），僅訓練 Decoder。

---

## 1. 五種方法差異說明

| 方法 | 類型 | 說明 |
|------|------|------|
| **Grounding DINO + SAM** | Zero-shot | 文字 prompt 偵測「sky」→ SAM 分割；不需訓練，漏檢常較低但 10066 上 FP 高、10870 有異常。 |
| **CLIPSeg（原始版）** | Zero-shot | 固定閾值 0.5、單一 prompt "sky"、無空間先驗／OPEN／多 prompt；10870 上 FP 極高（0.94）。 |
| **CLIPSeg 空間改善後** | Zero-shot | 動態閾值 + 空間先驗 + OPEN + sky vs building 多 prompt；同一腳本預設參數，10870 明顯改善。 |
| **DINO + Linear** | 訓練式 | DINOv2 backbone 凍結，**兩層 1×1 解碼頭**（Fusion 384→256、Prediction 256→1）；輕量、訓練快。 |
| **DINO + dino_segmentation** | 訓練式 | DINOv2 backbone 凍結，**最後四層特徵 concat + 4-block 卷積解碼器**（含 3×3）；多尺度、空間融合強。 |

- 僅 **後兩者** 有訓練；訓練資料相同，差異在 Decoder 架構與特徵來源（見 §4）。

---

## 2. 訓練結果：Train Loss / Val Loss（兩支 DINO）

兩者皆為 **10 epochs**、`batch_size=2`、`lr=1e-4`、`image_size=(160,160)`、Loss = BCEWithLogitsLoss + Dice（CombinedLoss），僅訓練 Decoder、Backbone 凍結。

### 2.1 DINO + Linear（SkySegModel）

- **Checkpoint 目錄**：`outputs/train_dino_linear_20260214_022830/`
- **training_log.csv** 摘要：

| Epoch | Train Loss | Train IoU | Val Loss | Val IoU | Val FP_rate | Val FN_rate |
|-------|------------|-----------|----------|---------|-------------|-------------|
| 1 | 0.4615 | 0.728 | 0.3075 | 0.843 | 0.0248 | 0.0688 |
| 5 | 0.1841 | 0.852 | 0.2392 | 0.874 | 0.0173 | 0.0638 |
| 10 | **0.1503** | **0.874** | **0.2125** | **0.894** | 0.0155 | 0.0486 |

- Val 最佳：Epoch 10，Val IoU **0.894**、Val Loss **0.212**。

### 2.2 DINO + dino_segmentation（DINOv2SegmentationModel）

- **Checkpoint 目錄**：`outputs/train_dino_segmentation_20260214_101339/`
- **training_log.csv** 摘要：

| Epoch | Train Loss | Train IoU | Val Loss | Val IoU | Val FP_rate | Val FN_rate |
|-------|------------|-----------|----------|---------|-------------|-------------|
| 1 | 0.2245 | 0.855 | 0.1926 | 0.916 | 0.0151 | 0.0296 |
| 5 | 0.1000 | 0.918 | 0.1645 | 0.935 | 0.0099 | 0.0299 |
| 10 | **0.0851** | **0.928** | **0.1592** | **0.938** | 0.0092 | 0.0295 |

- Val 最佳：Epoch 10，Val IoU **0.938**、Val Loss **0.159**；整體 Val 優於 DINO + Linear（Val IoU 0.94 vs 0.89）。

---

## 3. 五張小樣本測試表現

- **測試清單**：10066（046、047）、10870（091、092）、1093（082），共 5 張。
- **指標**：每格為 IoU_mean / FP_rate / FN_rate（camera 內平均）；整體 IoU = 5 張平均。

| 方法 | 10066 (n=2) | 10870 (n=2) | 1093 (n=1) | 整體(5張) IoU |
|------|-------------|-------------|------------|----------------|
| Grounding DINO + SAM | 0.692 / 0.695 / 0.022 | 0.265 / (異常) / 0.000 | 0.877 / 0.057 / 0.020 | ≈0.61 |
| CLIPSeg（原始版） | 0.794 / 0.022 / 0.188 | 0.254 / 0.940 / 0.084 | 0.898 / 0.004 / 0.094 | ≈0.60 |
| CLIPSeg 空間改善後 | 0.824 / 0.000 / 0.176 | 0.432 / 0.245 / 0.269 | 0.895 / 0.004 / 0.094 | ≈0.68 |
| DINO + Linear | 0.627 / 0.000 / 0.373 | 0.792 / 0.055 / 0.091 | 0.843 / 0.000 / 0.157 | 0.736 |
| **DINO + dino_segmentation** | 0.623 / 0.003 / 0.375 | **0.900** / **0.006** / 0.084 | **0.983** / **0.001** / **0.015** | **0.806** |

- **簡要排序（整體 IoU）**：DINO + dino_segmentation（0.806）> DINO + Linear（0.736）> CLIPSeg 空間改善後（≈0.68）> Grounding DINO + SAM（≈0.61）> CLIPSeg 原始版（≈0.60）。
- **觀察**：在 10870（夜間比例高）、1093 上，DINO + dino_segmentation 的 IoU／FP／FN 皆最佳；10066 上 CLIPSeg 空間改善後 IoU 最高（0.824），DINO 兩支在此 camera 的 FN 偏高（約 0.37）。Overlay 與指標一致（綠=TP、紅=FP、藍=FN）；整體較低者主因 10870 拉低平均。

---

## 4. 兩種 DINO Decoder 的訓練方式與架構

- **共同點**：Backbone 皆為 DINOv2 ViT-S/14（`torch.hub`），**全部凍結**；僅 **Decoder 可訓練**。訓練資料、optimizer（lr=1e-4）、epochs=10、image_size=(160,160)、CombinedLoss 相同。
- **差異**：特徵來源與 Decoder 結構不同。

### 4.1 DINO + Linear（SkySegModel）

- **特徵**：Backbone **最後一層** `x_norm_patchtokens` → (B, N, 384)，reshape 成 (B, 384, h, w)。
- **Decoder**（可訓練）：
  - **Fusion**：Conv2d(384→256, 1×1) + BatchNorm2d + ReLU
  - **Prediction**：Conv2d(256→1, 1×1) → Logit
- **輸出**：Bilinear upsample (h,w)→(H,W) → (B, 1, H, W) Logit。
- **功能**：單層特徵、僅 1×1 卷積，輕量、訓練快；缺點為無多尺度、無 3×3 空間上下文。

### 4.2 DINO + dino_segmentation（DINOv2SegmentationModel）

- **特徵**：Backbone `get_intermediate_layers(x, n=4, norm=True)` → 4 個 (B, N, 384)，**channel concat** → (B, N, 1536)，reshape 成 (B, 1536, h, w)。
- **Decoder**（可訓練，4 blocks）：
  - Block1：Conv2d(1536→512, 1×1) + BN + ReLU
  - Block2：Conv2d(512→512, 3×3, padding=1) + BN + ReLU
  - Block3：Conv2d(512→256, 3×3, padding=1) + BN + ReLU
  - Block4：Conv2d(256→1, 1×1) → Logit
- **輸出**：Bilinear upsample → (B, 1, H, W) Logit。
- **功能**：多層特徵 + 3×3 空間融合，表達力與邊緣／夜間表現較佳；參數量與計算較大。

架構圖與對照表見 **`outputs/dino_decoder_architecture.md`**。

---

## 5. 觀察與分析

- **Val**：DINO + dino_segmentation 的 Val IoU（0.938）與 Val Loss（0.159）均優於 DINO + Linear（0.894 / 0.212），且 Val FP/FN 較低。
- **五張小樣本**：DINO + dino_segmentation 在 10870、1093 上明顯較優，整體 IoU 0.806 最高；10066 上 zero-shot CLIPSeg 空間改善後 IoU 最高，兩支 DINO 在 10066 的 FN 偏高（漏標天空較多）。
- **Overlay 與指標**：表內為 camera 平均；單張視覺（如 10066 047）與對應 camera 指標一致（改善版少紅藍、DINO+Linear 較多藍），無計算或繪圖錯誤。

---

## 6. 下一步改善方向

- 若需加強 **10066**：可考慮該 camera 的資料量／augmentation 或 loss 權重。
- 若需 **零訓練、日間為主**：CLIPSeg 空間改善後；若需 **漏檢最少**：Grounding DINO + SAM（需修 10870 異常）。
- 可將架構圖（`dino_decoder_architecture.md` 或產出之 PNG）納入報告或 HTML 比較頁。

---

## 7. 輸出檔與指令

| 項目 | 路徑／指令 |
|------|------------|
| 五方法比較表 | `outputs/comparison_three_methods_small.md`、`outputs/comparison_three_methods_small.html`（含 overlay 比較） |
| DINO Decoder 架構說明 | `outputs/dino_decoder_architecture.md` |
| DINO + Linear 訓練 log | `outputs/train_dino_linear_20260214_022830/training_log.csv` |
| DINO + dino_segmentation 訓練 log | `outputs/train_dino_segmentation_20260214_101339/training_log.csv` |
| 小樣本 overlay | 各方法對應 overlay 目錄（見 comparison 內 Overlay 章節） |
| 重跑小樣本評估 | `python scripts/eval_clipseg_final.py --test_list outputs/test_list_small.txt`（CLIPSeg）；DINO 用對應 eval 腳本 + `--test_list outputs/test_list_small.txt` |

---

## 8. 今日結論

- **五種方法**：三種 zero-shot（Grounding DINO+SAM、CLIPSeg 原始／空間改善後）與兩種 in-domain 訓練（DINO+Linear、DINO+dino_segmentation）；僅後兩者有訓練，且訓練資料相同（train_list.txt / val_list.txt）。
- **訓練結果**：DINO + dino_segmentation 的 Val IoU（0.938）與 Val Loss（0.159）優於 DINO + Linear（0.894 / 0.212）；兩者 Decoder 架構差異為單層 1×1  vs  四層特徵 concat + 4-block 卷積（含 3×3）。
- **五張小樣本**：DINO + dino_segmentation 整體 IoU 最高（0.806），尤其在 10870、1093；10066 上 CLIPSeg 空間改善後 IoU 最佳，兩支 DINO 在 10066 的 FN 偏高。
