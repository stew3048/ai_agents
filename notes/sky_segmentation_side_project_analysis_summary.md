# Sky Segmentation Side Project：專案分析總結

> 結論先行：**是。目前在本專案中，DINO + 深層卷積解碼器（DINO-dino_segmentation）效果最好。**  
> 以下為專案目標、方法比較、實測結果與建議。

---

## 1. 專案目標與範圍

- **任務**：天空分割（Sky Segmentation）——輸入一張影像，輸出天空區域的二元 mask（0/1）。
- **資料**：SkyFinder 多相機 in-domain 資料，含 train/val/test split；test 涵蓋多個 camera（如 1093、3888、4795、9112、9291、9483、10066、10870、21444 等）。
- **評估指標**：IoU、Dice、FP rate（誤判為天空）、FN rate（漏判天空）；與既有 U-Net / in-domain 評估 pipeline 一致。

---

## 2. 方法演進：U-Net → ViT 骨架 → DINOv2 + DL Head

專案一開始曾用 **U-Net** 在相同 in-domain 資料上訓練，但發現**不論怎麼調參、改 loss 或 augmentation，表現就是比後來的 ViT 骨架方案差**。因此轉向以 **Vision Transformer（ViT）相關骨架**做分割，最後採用 **DINOv2（自監督預訓練）當 backbone、接上可訓練的 DL Head（解碼頭）**，得到目前最佳結果。簡述原因與結構好處如下。

### 2.1 為何 U-Net 在相同資料量下表現較差？（概念與理論）

- **從頭訓練 CNN 的資料需求**：U-Net 的 Encoder 是 CNN（如 ResNet / VGG），若**從頭訓練**，參數量不小，需要較多標註資料才能學到穩定、泛化好的特徵。在**相同、有限的 in-domain 資料**下，CNN 容易過擬合或學到的表徵偏狹，對未見過的 camera／光照／場景變化較敏感，導致 Val/Test IoU 或 FP/FN 調不上去。
- **歸納偏置的局限**：CNN 的歸納偏置是**局部性**（卷積核只看局部）與**平移等變性**，對紋理、邊緣有利，但**高層語義**（例如「整片天空」「雲與建築的界線」）往往需要更大感受野與全局關係。U-Net 雖有 skip connection 融合多尺度，但 Encoder 若未經大規模預訓練，高層特徵的語義強度有限，在天空這種「大區塊、語義單一但外觀多變」的任務上，容易在邊界或少見場景失準。
- **對比：預訓練帶來的先驗**：若改用在大規模影像上**自監督預訓練**過的骨架（如 DINOv2），模型已經見過**大量多樣影像**，學到的是更通用的視覺表徵（紋理、形狀、場景結構），而不是只擬合當前資料集。在相同 in-domain 資料量下，**凍結預訓練 backbone、只訓練輕量 Head**，等於把「學過很多影像不同樣子」的先驗搬進來，資料效率與泛化通常優於從頭訓練的 U-Net。

因此：**不是 U-Net 架構本身一定差，而是在「資料量有限、且希望好泛化」的前提下，從頭訓練的 U-Net Encoder 不如「大規模自監督預訓練的 ViT 骨架」來當特徵提取器。**

### 2.2 為何改採 Vision Transformer 相關骨架？

- **ViT** 以 **Self-Attention** 建模整張圖的 patch 關係，**全局感受野**從淺層就存在，有利於「天空 vs 非天空」這種需要整體布局的語義。
- 實務上 **ViT 系模型**（如 DINO、DINOv2）常搭配**大量無標註影像的自監督預訓練**（對比學習、masked image modeling 等），學到的是**與類別無關的通用視覺表徵**，遷移到分割任務時，只需學「從這些特徵畫出 mask」的輕量 Head，資料需求較低、泛化較好。
- 因此專案從「純 U-Net（CNN Encoder 從頭訓練）」轉為「**以 ViT 相關骨架為 Encoder**」做分割，再在此基礎上選定 DINOv2 + DL Head。

### 2.3 為何最後是 DINOv2 + DL Head 最優？

- **DINOv2** 在**大量影像上做自監督訓練**（無需 pixel-level 標註），學過**各式各樣的場景、光照、視角**，其 backbone 輸出的 patch 特徵已具備強語義與良好泛化。拿來當**凍結的 Encoder**，等於把「看過很多影像不同樣子」的知識固定住，只讓後面的 **DL Head（解碼頭）** 用 in-domain 標註資料學習「特徵 → 天空 mask」的對應。
- **DL Head 結構的好處**（以本專案最佳配置 **DINO + 深層卷積解碼器**為例）：
  - **多層特徵融合**：使用 Backbone **最後四層**的 patch 特徵做 channel concat（1536 維），兼顧淺層細節與深層語義，比只用最後一層（如 Linear Head）更能保留邊界與多尺度資訊。
  - **卷積解碼器的空間整合**：Head 以 **1×1 → 3×3 → 3×3 → 1×1** 的卷積 block 組成，**3×3 卷積**在特徵圖上做局部空間聚合，把 ViT 的 patch 級輸出轉成平滑、連續的 mask，並強化邊緣與小結構，避免僅用 1×1（點對點）導致的鋸齒或破碎。
  - **輕量、只訓練 Head**：Backbone 凍結，僅訓練解碼器，參數量與過擬合風險小，在**相同資料量**下比從頭訓練 U-Net 更容易得到高 IoU、低 FP/FN，且跨 camera 更穩。

整體脈絡：**U-Net（同資料量調不上去）→ 改用 ViT 相關骨架（全局語義、預訓練先驗）→ 以 DINOv2 自監督 backbone + 可訓練 DL Head（多層特徵 + 卷積解碼）最優。**

---

## 3. 當前比較的五種方法

| 方法 | 類型 | 說明 |
|------|------|------|
| **Grounding DINO + SAM** | Zero-shot | 文字 prompt 偵測「sky」→ SAM 分割；不需訓練，漏檢常較低，但部分 camera FP 高。 |
| **CLIPSeg** | Zero-shot | 單一 prompt、固定閾值；10870 上 FP 極高。 |
| **CLIPSeg 空間改善** | Zero-shot | 動態閾值 + 空間先驗 + 形態學 + 多 prompt 對比；同一腳本改善版。 |
| **DINO + Linear** | In-domain 訓練 | DINOv2 ViT-S/14 **凍結**，僅 **兩層 1×1 解碼頭**（384→256→1）；輕量、訓練快。 |
| **DINO + dino_segmentation** | In-domain 訓練 | DINOv2 ViT-S/14 **凍結**，**最後四層特徵 concat（1536 維）+ 4-block 卷積解碼器**（1×1→3×3→3×3→1×1）；多尺度、空間融合強。 |

後兩者使用相同 in-domain 資料與訓練設定（僅 Decoder 可訓練），差異在 **Decoder 架構**：Linear 為極輕量 1×1；dino_segmentation 為 **深層卷積**（含 3×3 空間上下文）。

---

## 4. 整體結論：DINO + 深層卷積最佳

在完整 test list 上的 **整體平均（僅有 GT 的圖）** 如下。

| 方法 | n 張 | mean IoU | mean FP rate | mean FN rate |
|------|------|----------|--------------|--------------|
| **DINO-dino_segmentation** | 116 | **0.9204** | **0.0105** | **0.0506** |
| DINO-Linear | 116 | 0.8363 | 0.0242 | 0.1061 |
| CLIPSeg_spatial_improved | 116 | 0.7566 | 0.0474 | 0.1021 |
| CLIPSeg | 116 | 0.7514 | 0.0835 | 0.1108 |
| GroundingDINO-SAM | 106 | 0.7282 | 0.1696 | 0.0124 |

- **IoU 最高、FP 最低、FN 次低**：皆為 **DINO-dino_segmentation**（DINO + 深層卷積解碼器）。
- **DINO + Linear**（淺層 1×1）明顯次之（IoU 0.84），但仍優於三種 zero-shot。
- **Zero-shot**：Grounding FN 最低但 FP 高；CLIPSeg 兩支 IoU 約 0.75、FN 較高。

因此：**目前 sky segmentation 這個 side project 中，用 DINO 加上「深層卷積解碼器」（dino_segmentation）的效果最好。**

---

## 5. 為何 DINO + 深層卷積較好？

- **特徵**：dino_segmentation 使用 Backbone **最後四層** patch tokens 做 channel concat（1536 維），相較 Linear 只用**最後一層**（384 維），多尺度與高層語義更足。
- **解碼器**：4-block 結構含 **3×3 卷積**，具備空間上下文與邊緣細化能力；Linear 僅 1×1，無局部空間建模。
- **訓練**：兩者皆只訓練 Decoder、Backbone 凍結；同一資料與 loss 下，深層卷積的 Val IoU（約 0.938）與 Test IoU（0.92）均高於 Linear（Val 約 0.894、Test 0.84），且 FP/FN 更均衡。
- **實務**：在難 camera（如 10870、10066）上，dino_segmentation 的 IoU/FP/FN 普遍優於 Linear，顯示深層卷積對場景變化與邊界更穩健。

---

## 6. 各方法在不同 camera 上的表現（節錄）

- **10870**（整體最難）：DINO-dino_segmentation 仍能維持 IoU 0.91、FP 0.009；CLIPSeg 與 Grounding 在此 camera 的 IoU 低、FP 高。
- **10066**（FN 最高）：各方法普遍 FN 偏高；dino_segmentation 在此 IoU 0.73、FN 0.27，仍為五方法中較佳。
- **21444 / 9483**：多為無天空或極易場景，DINO 兩支可達 IoU 1.0 或 0.97，zero-shot 亦不差。
- **結論**：DINO-dino_segmentation 在「難 camera」上拉開與其他方法的差距，在「易 camera」上維持高 IoU、低 FP/FN。

---

## 7. 各 camera 預測難度（跨方法平均）

依 mean IoU 由低到高（越前面越難）：

| 難度 | camera | mean IoU | mean FP | mean FN | 備註 |
|------|--------|----------|---------|---------|------|
| 1 最難 | 10870 | 0.626 | 0.271 | 0.100 | IoU 最低、FP 最高 |
| 2 | 9291 | 0.699 | 0.123 | 0.059 | |
| 3 | 10066 | 0.773 | 0.104 | **0.174** | FN 最高（易漏判天空） |
| 4 | 3888 | 0.791 | 0.099 | 0.037 | |
| 5 | 4795 | 0.795 | 0.015 | 0.093 | |
| 6 | 9112 | 0.855 | 0.035 | 0.106 | |
| 7 | 1093 | 0.856 | 0.008 | 0.109 | |
| 8 | 21444 | 0.900 | 0.002 | 0.000 | 多為無天空 |
| 9 最易 | 9483 | 0.924 | 0.010 | 0.064 | |

可作為後續「針對難 camera 加強資料或 loss」的依據。

---

## 8. 方法選擇建議

| 需求 | 建議 |
|------|------|
| **效果最佳、可接受訓練** | **DINO + dino_segmentation**（深層卷積解碼器） |
| **輕量、仍要 in-domain** | DINO + Linear（略遜 IoU，但參數少、訓練快） |
| **零訓練、日間為主** | CLIPSeg 空間改善版 |
| **漏檢最少、可接受高 FP** | Grounding DINO + SAM（需注意 10870/10066 的 FP） |

---

## 9. 輸出與重現

- **五方法評估**：`scripts/eval_five_methods_in_domain_test.py`，輸出 `outputs/eval_five_methods/*_test_best_worst_metrics.json`、`per_image_metrics.csv`。
- **三種分析**：`scripts/analyze_five_methods_metrics.py` → `five_methods_analysis_report.txt` / `.json` / `_summary.md`。
- **日誌與細表**：`notes/2026-02-15-five-methods-test-analysis.md`。
- **架構說明**：DINO + Linear vs DINO + dino_segmentation 見 `notes/2026-02-14-five-methods-comparison-dino-decoder-small-test.md`、`models/dinov2_segmentation_decoder.py`。

---

## 10. 一句總結

**專案歷程**：先用 U-Net 在相同資料上訓練，發現不論怎麼調表現仍較差（CNN 從頭訓練、資料有限時泛化不足）→ 改採 Vision Transformer 相關骨架以取得全局語義與預訓練先驗 → 最後以 **DINOv2（自監督、看過大量多樣影像）為凍結 backbone，接上可訓練的 DL Head（多層特徵 + 深層卷積解碼器）** 最優。**目前 sky segmentation 專案中，DINO-dino_segmentation（DINOv2 + 深層卷積 Head）效果最好（mean IoU 0.92）；DINO + Linear 次之；三種 zero-shot 再次之。建議以 DINO-dino_segmentation 為主力。**
