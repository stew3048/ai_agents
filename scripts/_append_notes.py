import os, sys
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
notes_path = os.path.join(_root, "notes", "2026-03-13-EVAL_SUMMARY.md")

appendix = """

---

## 十、CLIPSeg 融合實驗與結論

### 背景

在完成三方法評估後，嘗試將 CLIPSeg 的語義理解能力融合進 DINOv2+CNN，希望改善濃霧等低對比場景的表現。

---

### 10-1. 架構融合（DINOv2+CLIPSeg）

**做法：** 建立新模型 `DINOv2CLIPSegModel`（`models/dinov2_clipseg_decoder.py`）

- DINOv2 最後 4 層特徵（1536 ch）為主幹
- CLIPSeg heatmap（1 ch）在推論時產生，min-max normalize 後 resize 至 feature map 解析度
- 兩者沿 channel 方向 concatenate → decoder 第一層輸入從 1536 改為 1537
- 舊 checkpoint 權重遷移：新增的第 1537 個 channel 以零初始化

**訓練（`scripts/train_dino_clipseg.py`）：**
- Phase 1：預算全部 2850 張的 CLIPSeg heatmap 快取至 `data/clipseg_heatmap_cache/`
- Phase 2：以舊 DINOv2+CNN best.pth 初始化，fine-tune 10 epochs
- 最佳 checkpoint：`outputs/train_dino_clipseg_20260313_184550/checkpoints/best.pth`（Val IoU = 0.9388）

**5 張測試結果對比（vs. custom_data_metrics.csv）：**

| 場景 | DINOv2+CNN | DINOv2+CLIPSeg | 差異 |
|---|---|---|---|
| 濃霧 10066_046 | 0.6041 | 0.5608 | ▼ -0.043 |
| 夜間建築物 10870_040 | 0.9387 | 0.8283 | ▼ -0.110 |
| 天海一線 3888_1480 | 0.9235 | 0.8779 | ▼ -0.046 |
| 夜間強遮擋 4795_017 | 0.9312 | 0.4778 | ▼ -0.453 |
| 日間建築物 9483_017 | 0.9754 | 0.9463 | ▼ -0.029 |
| **平均** | **0.8746** | 0.7382 | ▼ **-0.136** |

**結論：全面退步。** 最嚴重的是夜間強遮擋（-0.453），CLIPSeg heatmap 在夜間場景引導模型過度預測錯誤區域（Recall=1.0 但 Precision=0.478）。

---

### 10-2. 機率圖分析（`scripts/visualize_prob_map.py`）

針對濃霧場景（10066_046）產生三個模型的 JET colormap 機率熱力圖，輸出至 `output/prob_maps/10066_046_prob_comparison.png`。

**信心統計（mean entropy）：**

| 模型 | mean entropy | 說明 |
|---|---|---|
| CLIPSeg | **0.486** | 接近最大值 ln(2)≈0.693，大量 pixel 停在 0.5 附近，幾乎是均勻雜訊 |
| DINOv2+CNN | 0.040 | 非常確定，保守但精準 |
| DINOv2+CLIPSeg | 0.050 | 確定，但仍漏判大量霧中天空 |

**關鍵發現：** CLIPSeg 在濃霧場景的 entropy=0.486，代表其 heatmap 不含有效語義訊號。把雜訊圖注入 decoder 的 channel concatenation，造成負面干擾。

---

### 10-3. Post-processing Ensemble 實驗

不重新訓練，測試多種 post-processing 策略於濃霧場景（`scripts/test_fog_ensemble.py`）：

| 策略 | IoU |
|---|---|
| A. DINOv2 baseline (thr=0.5) | 0.5016 |
| B. max(DINO, CLIPSeg×0.8) | 0.5679 |
| C. 0.7×DINO + 0.3×CLIPSeg | 0.5020 |
| D. DINOv2 lower threshold (0.30) | 0.5135 |
| **E. Mask Expansion (dil=15px, CLIPSeg>0.35)** | **0.6323** |

**方法 E（Mask Expansion）說明：**
1. 取 DINOv2 二值 mask 為基底
2. 對 mask 邊界向外膨脹 15px（形態學 dilation）→ 製造不確定帶
3. 不確定帶內，CLIPSeg > 0.35 的 pixel 補入最終 mask

**全 5 張測試（`scripts/test_mask_expansion_all5.py`）：**

| 場景 | Baseline IoU | Expansion IoU | 差異 |
|---|---|---|---|
| 濃霧 | 0.5016 | 0.6323 | ▲ **+0.131** |
| 夜間建築物 | 0.7385 | 0.7722 | ▲ +0.034 |
| 天海一線 | 0.8400 | 0.8264 | ▼ -0.014 |
| 夜間強遮擋 | 0.4428 | 0.3785 | ▼ **-0.064** |
| 日間建築物 | 0.9284 | 0.9418 | ▲ +0.013 |

夜間強遮擋退步原因：天空被遮擋物切割成碎片，dilation 膨脹後直接覆蓋相鄰遮擋物，CLIPSeg 對夜間暗色區域同樣輸出 > 0.35，造成大量 FP。

---

### 10-4. 最終結論：為何無法有效融合 VLM

| 根本原因 | 說明 |
|---|---|
| **CLIPSeg 在困難場景失效** | 濃霧（entropy=0.486）、夜間，CLIPSeg 輸出幾乎是均勻雜訊，無可信語義訊號 |
| **Channel concatenation 無法選擇性忽略雜訊** | Decoder 在每個場景都被迫接收 CLIPSeg channel，即使該 channel 是雜訊 |
| **Post-processing 有得有失** | Mask Expansion 改善濃霧 +0.131，但夜間強遮擋退步 -0.064，找不到跨場景通用參數 |
| **CLIPSeg 弱點正好是 DINOv2+CNN 的強項** | DINOv2 在夜間、遮擋場景 IoU > 0.90，這些恰好是 CLIPSeg entropy 最高的場景 |

**最佳模型仍為 DINOv2+CNN**
`outputs/train_dino_segmentation_20260214_101339/checkpoints/best.pth`

若要真正改善濃霧場景，更合理的方向是訓練資料增強（gaussian blur + brightness reduction + fog overlay），讓 DINOv2+CNN 自身學會「灰白紋理也可以是天空」，而非依賴 entropy 偏高的 VLM 訊號。
"""

with open(notes_path, 'a', encoding='utf-8') as f:
    f.write(appendix)
print("已成功附加至 EVAL_SUMMARY.md")
