# ViT 與 SAM 架構說明

> 今日日誌：整理 ViT（Vision Transformer）是什麼、我們做天空分割的三方法誰有用到 ViT，以及 SAM 的架構簡述。  
> 日期：2026-02-08

---

## 一、ViT 是什麼？

**ViT = Vision Transformer**，把 Transformer 從 NLP 搬到影像的做法：

1. **把圖切成 patch**  
   將影像切成小塊（例如 14×14 或 16×16），每塊展平後做線性投影成 **patch embedding**。

2. **加位置編碼**  
   每個 patch 有對應的位置向量，讓模型知道空間關係。

3. **用 Transformer encoder**  
   多層 self-attention + FFN，讓每個 patch 與整張圖其他 patch 互動，捕捉**全局關係**（不只鄰近像素）。

4. **輸出**  
   每個 patch 一個特徵向量，可再接分類頭、分割頭等。

**與 CNN 的差別**：CNN 用局部卷積、逐層擴大感受野，偏重局部紋理；ViT 一開始就對整張圖的 patch 做 attention，偏重全局與長距離關係，通常需要較多資料或 pretrain 才穩。

---

## 二、我們做天空分割的方法有沒有用 ViT？

| 方法 | 有沒有用 ViT？ | 說明 |
|------|----------------|------|
| **DL（U-Net）** | **沒有** | 專案內的 U-Net 是純 CNN（Conv2d、BatchNorm、ReLU、skip connection），沒有 Transformer。 |
| **SAM 2** | **有** | Image encoder 為 Transformer / ViT 類架構（如 sam2_hiera_small 的 Hiera）。 |
| **CLIPSeg（VLM）** | **有** | CLIP 的 vision 端是 ViT，CLIPSeg 沿用。 |
| **DINO**（做 SAM prompt 用） | **有** | 使用 DINOv2 ViT-B/14（dino_sky_prompt.py）。 |

---

## 三、SAM 架構簡述（三塊）

SAM 可分成三階段：**Image Encoder → Prompt Encoder → Mask Decoder**。

```
[ 影像 ]  →  Image Encoder (ViT)  →  整圖的 patch 特徵
                                            ↓
[ 提示：點/框 ]  →  Prompt Encoder  →  提示特徵  →  Mask Decoder  →  [ 預測 mask ]
```

### 1. Image Encoder（ViT）

- **輸入**：一張圖。
- **做法**：用 ViT 類 backbone 把圖切成 patch，過 Transformer，得到每個 patch 的特徵。
- **輸出**：整張圖的 **spatial 特徵圖**（每個位置一個特徵向量），供後續 decoder 使用。

### 2. Prompt Encoder

- **輸入**：使用者的提示（點、框、或粗 mask）。
- **做法**：將點/框/mask 編碼成與 image 特徵維度對齊的 **prompt 特徵向量**。
- **輸出**：代表「使用者指定的是哪裡」的 embedding。

### 3. Mask Decoder

- **輸入**：Image encoder 的整圖 patch 特徵 + Prompt encoder 的提示特徵。
- **做法**：輕量 Transformer decoder（cross-attention）：用提示去問「整圖特徵裡與提示最相關的區域」，再經預測頭輸出像素級前景/背景；常輸出多個 mask 候選再依分數選。
- **輸出**：預測的 mask。

**一句話**：ViT 負責「看懂整張圖」→ Prompt 負責「指定目標」→ Mask Decoder 負責「根據兩者畫出 mask」。

---

## 四、小結

- **ViT**：以 patch + Transformer 做影像特徵，偏全局關係；我們專案中 DL（U-Net）未使用，SAM、CLIPSeg、DINO 有使用。
- **SAM**：先以 ViT 產出整圖 patch 特徵，再與編碼後的提示一起送入 Mask Decoder 產出分割結果；因此「先用 ViT 輸出每個 patch 的特徵，再接分割相關架構」的理解是正確的。

---

## 五、MLP vs CNN 筆記（來源：`C:\Users\yiching\handwriting-recognition-project\node`）

### 總結

因為 MLP 是全連結神經網路，它不像 CNN 內建局部性與平移等 inductive bias，所以本身沒有明確的空間結構假設；
相對地，CNN 透過局部卷積與權重共享，自然保留了影像的空間關係。
MLP 的表達能力其實非常彈性，但正因為缺乏這些結構性假設，它通常需要更多資料才能學到穩定、可泛化的表示，否則容易 overfitting；
在這個 MNIST 的例子中，MLP 不容易快速學到「某個視覺 pattern 在不同位置出現仍屬於同一概念」，因此在效率與穩定性上不如 CNN。

### 自問自答 QA

- **Q：那為什麼不用 MLP 就好？**
- **A：** 在影像這種高度結構化的資料上，適當的 inductive bias 其實比模型彈性更重要，因為它能大幅降低 sample complexity。
