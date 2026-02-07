# Sky Segmentation Side Project — 收尾總結

> 依 notes 日誌整理：目標、做了什麼、Findings、Conclusion。  
> 日期：2026-02-06

---

## 一、目標（Goal）

1. **問題定義**：天空分割（Sky Segmentation）為 **Semantic Segmentation**，產出「哪裡是天空」的 mask，用於後續視覺／曝光等應用。
2. **核心問題**：不是選「最強模型」，而是**判斷「問題壞在哪一層」、再選對工具**（心智圖與老闆對齊）。
3. **本 side project 具體目標**：
   - 建立 **DL（U-Net）** 與 **VLM（SAM 2.0）** 在「同一批資料、多情境」下的可比較流程。
   - 理解**不同情境**（no-sky、sea-sky、heavy-occlusion、night、urban 等）下，誰適合用 DL、誰適合用 VLM，形成**選模策略**。
   - 產出 diagnostic 資料集、評估腳本、情境報告，供後續方法論與產品決策使用。

---

## 二、做了什麼（What We Did）

### 2.1 資料與評估管線

- **CV baseline**：HSV + 顏色閾值驗證 pipeline（toy dataset IoU ~90%），確認指標與對齊正確。
- **Multi-camera 訓練**：`train_multi_camera.py` 在 10066 / 9291 / 9112 訓練，Val=9483、Test=10870（unseen），採用條件式 augmentation（針對 night 微調），產出 U-Net checkpoint。
- **Diagnostic 資料集**：166 筆情境標註（has_sky、sea_sky_confusable、scene、occlusion、light、weather 等），用於**情境理解與 failure 分析**，非公平競賽用 test。
- **Failure cases**：依五情境各取 top-5 worst（共 25 筆），用於 DL 失敗模式分析與 SAM 對比。

### 2.2 DL 分析

- 各 camera / 各情境的 DL 失敗模式分析（FP 主導 vs FN 主導）。
- Val / Test 的 night vs day 切分評估（luma 閾值），確認夜間 FN 高、跨 camera 泛化落差。
- 單張 U-Net 評估腳本（`eval_unet_single_image.py`），與 SAM 使用同一 checkpoint 可比對。

### 2.3 VLM（SAM 2.0）對比

- **提示策略**：依情境給不同 prompt（no-sky→中心點、sea-sky→上方 box、heavy-occlusion→9 點、night→上方單點、urban/default→上方 box）。
- **25 failure cases**：跑 SAM 2.0（sam2_hiera_small），產出 overlay 與 IoU/FP/FN，與 DL 逐筆對比。
- **全 diagnostic 166 張**：跑 SAM，產出 `diagnostic_sam_metrics.csv`、`sam2_diagnostic_overlays/`，與 DL 依情境彙總比較。
- **互動式實驗**：單張圖手動點選（pointToSky / pointToGround 等），驗證「點哪裡割哪裡」、遮擋情境差主要來自提示策略。
- **DINO 當 SAM prompt**：用 DINOv2 找「最像天空」的點當 prompt；結果在 25 failure cases 上整體不如既有 heuristics（FN 高）。

### 2.4 情境彙總與解讀

- **依五大情境 + default** 彙總 DL vs SAM（平均 IoU/FP/FN、每張圖誰贏）。
- **依天氣**（clear / cloudy / fog / night 等）彙總對比。
- 撰寫「為什麼 DL 在多數情境較好」「各情境該看什麼指標」「SAM 語意空間與整塊偏掉」等解讀，收錄於 `2026-02-06-dl-vs-sam-by-scenario-report.md`。

---

## 三、Findings（發現）

### 3.1 情境 vs 方法

- **No-sky**：SAM 的 FP 明顯低於 DL（提示約束強，不會整圖當天空）；無真天空時只看 FP。
- **Sea-sky-confusable**：DL 平均 IoU/FP 較佳；SAM 的 box 含海天交界，語意上天空與海相似，易整塊偏掉或邊界不穩。**情境本身難度高**，DL 與 VLM 都不理想。
- **Heavy-occlusion**：DL 明顯優於 SAM。SAM 多點常落在建築/樹上，分割的是「點所在物體」而非「所有天空」，導致 FN/FP 都高；手動點在明確天空則 IoU 可達 ~0.96，說明**差在提示策略**。
- **Night**：在 **25 failure cases** 子集上 SAM 優於 DL（空間先驗：點在上方 ≈ 天空，不依賴夜間外觀）；在 **全 diagnostic 166 張** 上平均仍是 DL 略優，但 SAM 在「每張圖 IoU 誰贏」上夜間贏 22 張、DL 贏 16 張，視樣本與提示而定。**Night 難在外觀難認，VLM 的空間先驗可彌補**。
- **Urban / default**：DL 平均 IoU/FN 較優；SAM 的 FP 較低但 FN 較高（割得較保守，用漏標換少誤報）。

### 3.2 為何 DL 在多數情境平均較好？

- **任務專用 vs 通用**：U-Net 在天空分割資料上訓練，直接學「什麼是天空」；SAM 是通用分割，依 prompt 在**整張圖語意空間**找「結構合理的連通區」，**沒有為天空類別訓練**。提示不足或給錯時，SAM 易在語意模糊處選到「對模型合理、但非天空」的區域，**整塊偏掉**（整片海、霧、或建築）。
- **困難情境**（天海一線、多霧、夜間）下，語意本就模糊，prompt 若又偏掉，整塊偏更明顯。

### 3.3 天海一線／Fog vs Night（情境難度 vs 架構）

- **天海一線、fog**：屬**情境本身視覺歧義**（天空與海/霧外觀相似），DL 與 VLM 表現接近且都不佳；非單一架構缺陷。
- **Night**：難在**外觀難認**（暗、低對比）；VLM 的**空間先驗**（點在上方）彌補了「認不出天空」的問題，故在部分設定下 VLM 較 DL 好。

### 3.4 指標取捨（FP 降、FN 升）

- 若某情境 SAM 的 **FP 大幅下降、但 IoU/FN 變差**：代表 SAM **預測較保守**（少報天空），錯報少、漏報多，是**用漏標換少誤報**。是否可接受依應用而定（例如下游不能把建築當天空 → FP 重要；要天空覆蓋率 → FN 重要）。

---

## 四、Conclusion（結論）

1. **選模策略**：  
   - **No-sky**：若只關心「不要誤報天空」，SAM（或類似的提示約束）可顯著降 FP。  
   - **Sea-sky / Fog**：情境難度高，DL 與 VLM 皆受限；可視為共同瓶頸，而非換模型即解。  
   - **Heavy-occlusion**：DL 較穩；若要用 SAM，需**提示落在明確天空區**（或改 box/少點），避免點在遮擋物上。  
   - **Night**：VLM 的空間先驗有優勢，可考慮 SAM（或類似方法）補強 DL 的夜間漏檢。  
   - **Urban / 一般**：DL 平均較穩；若應用特別在意 FP，可看 SAM 的保守預測是否可接受（並承受較高 FN）。

2. **方法論**：  
   - 建立「情境標註 → failure 分析 → DL vs VLM 同圖比較 → 依情境/天氣彙總」的流程，可複用於其他資料或模型。  
   - 各情境應**優先看的指標**不同（no-sky→FP、sea-sky→FP+IoU、heavy-occlusion→FN+IoU、night→IoU/FP/FN 都看、urban→IoU+FP）。

3. **Side project 收尾**：  
   - 目標已達：同一批資料下 DL vs SAM 可比較、情境化結論與解讀已整理、選模邏輯與限制已記錄於 notes。  
   - 後續若擴充：可調 SAM 提示策略（如 sea-sky 縮小 box、heavy-occlusion 只點天空區）、或擴充 diagnostic 與更多 VLM/backbone 比較。

---

*以上依 notes 日誌（2026-01-20 ～ 2026-02-06）整理。*
