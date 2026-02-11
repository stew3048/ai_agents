# DINO 接 DL 架構討論與方法理解

> 今日重點：深入討論 DINO、ViT、U-Net、Encoder/Decoder 架構，以及「DINO 接 DL」的實作方案分析。  
> 日期：2026-02-11

---

## 一、前提

- **延續**：先前已完成 DL vs SAM vs CLIPSeg 的三方法比較，理解各方法在不同情境下的表現。
- **今日範圍**：討論 DINO 是什麼、DINO 接 DL 的可能實作方案（方案 A/B/C/D），以及相關的架構概念（ViT、U-Net、Encoder/Decoder、Skip Connection 等）。
- **目標**：為後續「DINO 接 DL」的實作做準備，釐清架構設計與技術細節。

---

## 二、方法理解：核心概念

### 2.1 Vision Transformer (ViT) 詳解

#### ViT 的基本架構

**ViT = Vision Transformer**，把 Transformer 從 NLP 搬到影像的做法：

1. **Patch Embedding（圖像 → Patch Vectors）**
   - 將影像切成小塊（例如 14×14 或 16×16）
   - 每塊展平後做線性投影成 **patch embedding**
   - 例如：224×224 影像 → 14×14 = 196 個 patch，每個 patch 變成 768 維向量

2. **Position Embedding（加入位置資訊）**
   - 每個 patch 有對應的位置向量，讓模型知道空間關係

3. **Transformer Encoder（Self-Attention 層）**
   - 多層 self-attention + FFN
   - 讓每個 patch 與整張圖其他 patch 互動，捕捉**全局關係**（不只鄰近像素）

4. **任務頭（Task Head）**
   - 分類：Linear(768 → num_classes)
   - 分割：Decoder

**與 CNN 的差別**：
- CNN 用局部卷積、逐層擴大感受野，偏重局部紋理
- ViT 一開始就對整張圖的 patch 做 attention，偏重全局與長距離關係
- ViT 通常需要較多資料或 pretrain 才穩

#### ViT 在分類 vs 分割任務的差異

**分類任務：ViT 直接輸出 logits**
```
[ 影像 224×224 ]
    ↓
ViT (完整架構)
    ↓
[ Patch embeddings: 14×14×768 ] ← 196個patch，每個768維
    ↓
[ CLS token: 1×768 ] ← 全局特徵
    ↓
[ Linear/FFN: 768 → 1000 ] ← 分類頭
    ↓
[ Logits: 1000個類別 ]
```

**為什麼分類沒有 decoder？**
- 分類只需要一個全局特徵（CLS token），不需要恢復像素級輸出
- 所以：ViT 本身 = encoder（提取特徵），最後接分類頭（FFN）

**分割任務：ViT encoder + Decoder**
```
[ 影像 256×256 ]
    ↓
ViT Encoder (只取 encoder 部分)
    ↓
[ Patch embeddings: 16×16×768 ] ← 256個patch，每個768維
    ↓
[ Decoder: 上採樣回原圖尺寸 ]
    ↓
[ Mask: 256×256×1 ]
```

**為什麼分割有 decoder？**
- 分割需要每個像素的預測，但 ViT encoder 輸出是 patch-level（例如 16×16）
- 需要上採樣回原圖尺寸（256×256）
- 所以：ViT encoder（提取特徵）+ Decoder（恢復解析度）

**為什麼叫「ViT encoder」？**
- 在分割任務中，我們只使用 ViT 的 encoder 部分（提取特徵），後面接 decoder（恢復解析度）
- 在分類任務中，ViT 是完整的（encoder + 分類頭），所以直接叫「ViT」

#### CLS Token 的運作方式

**CLS Token 不是把 196 個 patch 合併成 1 個，而是一個額外加入的特殊 token**

**完整流程**：
```
[ 原始影像 224×224 ]
    ↓ 切成 patch
[ 196 個 patch (14×14) ]
    ↓ 每個 patch 變成 embedding
[ 196 個 patch embeddings: 196×768 ]
    ↓ 加入 CLS token
[ CLS token: 1×768 ]  ← 這是額外加入的！
    ↓ 拼接在一起
[ 總共 197 個 tokens: (1 + 196) × 768 ]
    ↓ 經過 Transformer encoder
[ 197 個 tokens: (1 + 196) × 768 ]
    ↓ 只取 CLS token
[ CLS token: 1×768 ]  ← 代表全局特徵
```

**重點**：
- CLS token 透過 self-attention 與所有 patch tokens 互動
- 經過多層後，CLS token 的向量會包含整張圖的語義資訊
- 196 個 patch embeddings 仍然存在，但分類任務只需要一個全局特徵，所以只用 CLS token

**非分類任務（例如分割）是否不用 CLS Token？**
- **通常不用**，因為分割需要每個 patch 的輸出（每個 patch 對應圖像的一個區域）
- CLS token 是全局特徵，分割不需要單一全局特徵
- 但有些設計會同時使用（例如用 CLS token 提供全局上下文）

#### Patch Embedding 是 ViT 的一部分嗎？

**答案：是的，Patch Embedding 是 ViT 的第一個組件**

**ViT 的完整架構**：
```
ViT (Vision Transformer) 包含：

1. Patch Embedding（圖像 → Patch Vectors）
   ↓
2. Position Embedding（加入位置資訊）
   ↓
3. Transformer Encoder（Self-Attention 層）
   ↓
4. 任務頭（Task Head）
   - 分類：Linear(768 → num_classes)
   - 分割：Decoder
```

**為什麼會覺得「Patch Embedding 在 Transformer 之前」？**
- 因為架構可以這樣理解：圖像處理階段 → Transformer 處理階段 → 任務階段
- 但實際上：Patch Embedding 是 ViT 的第一個組件，整個流程（Patch Embedding → Transformer → Head）都是 ViT 的一部分

#### 不同任務下 ViT 的 embedding 是否相同？

**答案：不同任務下，ViT encoder 的 embedding 會不同**

**詳細邏輯說明**：

1. **任務與 Label 不同 → Loss 函數不同**
   - **分類任務**：Label 是類別 ID（例如 "cat" = 3），Loss 是 Cross-Entropy
     ```
     Loss = -log(P(class=3 | image))
     ```
   - **分割任務**：Label 是 pixel-level mask（每個像素是 0 或 1），Loss 是 BCE 或 Dice
     ```
     Loss = BCE(pred_mask, gt_mask) 或 Dice(pred_mask, gt_mask)
     ```
   - **不同的 Loss 函數**：對同一張圖，分類任務關心「整張圖是什麼類別」，分割任務關心「每個像素是不是天空」

2. **Loss 不同 → Gradient 不同**
   - **分類任務的 Gradient**：
     ```
     ∇Loss_classification = ∂Loss_class / ∂ViT_encoder_weights
     ```
     - Gradient 的方向是「讓 ViT encoder 輸出更適合分類的特徵」（例如：強調類別區分的特徵）
   
   - **分割任務的 Gradient**：
     ```
     ∇Loss_segmentation = ∂Loss_seg / ∂ViT_encoder_weights
     ```
     - Gradient 的方向是「讓 ViT encoder 輸出更適合分割的特徵」（例如：強調像素級邊界的特徵）
   
   - **關鍵**：即使同一張圖，分類任務的 gradient 和分割任務的 gradient **方向不同、大小不同**

3. **Gradient 不同 → 權重更新不同**
   - **Backpropagation 更新權重**：
     ```
     ViT_encoder_weights = ViT_encoder_weights - learning_rate × gradient
     ```
   - **分類任務**：權重被更新成「適合分類」的方向
   - **分割任務**：權重被更新成「適合分割」的方向
   - **結果**：經過多次訓練後，兩個任務的 ViT encoder 權重**完全不同**

4. **權重不同 → Embedding 不同**
   - ViT encoder 的輸出 embedding = `ViT_encoder(image, weights)`
   - 因為權重不同，所以即使輸入同一張圖，輸出的 embedding 也不同
   - **分類任務的 embedding**：偏向「類別區分」的特徵
   - **分割任務的 embedding**：偏向「像素級邊界」的特徵

**完整流程圖**：
```
任務 A（分類）：
Image + Label_class → Loss_class → Gradient_class → 更新 ViT 權重 → Embedding_A

任務 B（分割）：
Image + Label_mask → Loss_seg → Gradient_seg → 更新 ViT 權重 → Embedding_B

結果：Embedding_A ≠ Embedding_B
```

**例子**：
- **任務 A：ImageNet 分類**
  - Label：`"cat"` (類別 ID = 3)
  - Loss：Cross-Entropy（預測類別 vs 真實類別）
  - Gradient：讓 ViT encoder 學到「貓 vs 狗 vs 鳥」的區分特徵
  - 結果：ViT encoder 的 embedding 偏向「類別區分」
  
- **任務 B：天空分割**
  - Label：`mask` (256×256，每個像素 0 或 1)
  - Loss：BCE（預測 mask vs 真實 mask）
  - Gradient：讓 ViT encoder 學到「天空 vs 非天空」的像素級邊界特徵
  - 結果：ViT encoder 的 embedding 偏向「像素級邊界」

**如果 ViT encoder 凍結（frozen）呢？**
- 如果 ViT encoder 凍結（不更新權重），那麼：
  - **不同任務下，ViT encoder 的輸出 embedding 會相同**（因為權重相同）
  - 但任務頭（分類頭或 decoder）會不同，所以最終輸出不同
  - **例子**：用同一個 frozen ViT encoder，接分類頭 → 分類結果，接分割 decoder → 分割結果

---

### 2.2 DINO vs ViT 的差別

#### DINO（Self-Supervised）

- **訓練方式**：不需要 label，只用大量無標註影像
- **目標**：學「視覺結構與一致性」（例如：同一張圖的不同 crop 應該相似、不同圖應該不同）
- **輸出**：patch-level 特徵（與 ViT 類似，但學到的表示更通用）

#### ViT（標準版，Supervised）

- **訓練方式**：需要 image + label（例如 ImageNet：圖 + 類別）
- **目標**：學「這張圖是什麼類別」
- **輸出**：可做分類、detection、分割（需接任務頭）

#### 差別總結

| 項目 | ViT (Supervised) | DINO (Self-Supervised) |
|------|------------------|------------------------|
| **訓練資料** | 需要標註 | 不需要標註 |
| **學習目標** | 學「類別」 | 學「視覺結構」 |
| **應用** | 通常需 fine-tune 到特定任務 | 特徵可直接用於多種下游任務 |

**重點**：
- DINO 是 self-supervised，ViT 通常是 supervised
- DINO 的特徵更通用，可直接用於分割、detection 等任務
- DINO 也是一種 ViT（使用 ViT 架構，但用 self-supervised 方式訓練）

---

### 2.3 Encoder / Decoder 架構

#### 基本概念

- **Encoder**：把輸入「編碼」成特徵（例如圖 → 特徵向量）
- **Decoder**：把特徵「解碼」回輸出（例如特徵 → mask）

#### 在視覺 AI 中

**Encoder**：通常是「下採樣」（圖變小、通道變多）
- 例如：256×256×3 → 128×128×64 → 64×64×128 → 32×32×256
- 目的：提取高層語義特徵

**Decoder**：通常是「上採樣」（特徵變大、通道變少）
- 例如：32×32×256 → 64×64×128 → 128×128×64 → 256×256×1
- 目的：把特徵變回原圖尺寸的預測

**不是所有視覺 AI 都有 encoder/decoder**：
- **分類**：通常只有 encoder（圖 → 特徵 → 類別）
- **分割**：通常有 encoder + decoder（圖 → 特徵 → mask）
- **Detection**：可能有 encoder + decoder（例如 YOLO 的 head）

---

### 2.4 U-Net 架構詳解

#### U-Net 的結構（Encoder-Decoder + Skip Connections）

```
輸入 (256×256×3)
    ↓
[Encoder 路徑：下採樣]
    ↓
Conv → 128×128×64
    ↓
Conv → 64×64×128
    ↓
Conv → 32×32×256
    ↓
Conv → 16×16×512  ← 最底層（bottleneck）
    ↓
[Decoder 路徑：上採樣]
    ↓
Upsample + Concat(512層) → 32×32×256
    ↓
Upsample + Concat(256層) → 64×64×128
    ↓
Upsample + Concat(128層) → 128×128×64
    ↓
Upsample + Concat(64層) → 256×256×1
    ↓
輸出 mask (256×256×1)
```

#### 重點

1. **Encoder**：下採樣，提取語義特徵
2. **Decoder**：上採樣，恢復解析度
3. **Skip Connections**：把 encoder 的細節資訊直接傳給 decoder（避免細節丟失）

**為什麼叫 U-Net**：架構像 U 字（下採樣再上採樣）

#### Receptive Field 的解釋

**Receptive Field**：某一層的某個輸出像素，對應到原始輸入圖像的區域大小

**多層堆疊後 Receptive Field 變大**：
- **kernel_size 仍然是 3×3**（沒有變大）
- 但對應原始圖像的區域變大了（因為深層是濃縮版）
- 深層的一個像素「看到」原始圖像更大的區域

**例子**：
```
原始圖像: 256×256
    ↓ Conv (3×3, stride=2)
128×128  ← 每個像素對應原始圖像的 3×3 區域
    ↓ Conv (3×3, stride=2)
64×64   ← 每個像素對應原始圖像的 5×5 區域
    ↓ Conv (3×3, stride=2)
32×32   ← 每個像素對應原始圖像的 9×9 區域
```

**所以**：Filter size 不變（仍然是 3×3），但對應原始圖像的區域擴大（因為深層是濃縮版）

#### Upsample（上採樣）與 Concat（拼接）

**Upsample（上採樣）**：
- **目的**：把特徵圖變大（例如 32×32 → 64×64）
- **方法**：最近鄰插值、雙線性插值、Transposed Convolution
- **為什麼 H×W 變大，但通道數不變？**
  - Upsample 只改變空間尺寸（H×W），不改變通道數（C）

**Concat（拼接）**：
- **目的**：把兩個特徵圖在通道維度上拼接
- **例子**：
  ```
  Feature A: 64×64×128
  Feature B: 64×64×128
      ↓ Concat (dim=2, 通道維度)
  輸出: 64×64×256  ← 128 + 128 = 256
  ```
- **為什麼第三個數字（通道數）會相加？**
  - Concat 是在通道維度拼接，所以通道數 = A的通道數 + B的通道數

**與 Transformer Decoder 的差異**：
- Transformer Decoder（self-attention + cross-attention）：用於序列生成
- U-Net 的 decoder：CNN-based，用於分割任務
- 兩者不同：U-Net decoder 用 upsample + concat，Transformer decoder 用 attention

#### Skip Connection 的架構

**Skip Connection 的設計**：
```
Encoder 路徑:
256×256×3 → Conv → 128×128×64  ← 這層的輸出
    ↓
128×128×64 → Conv → 64×64×128
    ↓
64×64×128 → Conv → 32×32×256

Decoder 路徑:
32×32×256 → Upsample → 64×64×256
    ↓
Concat(64×64×256, 64×64×128) ← Skip connection: 把 encoder 的 64×64×128 接過來
    ↓
64×64×384 → Conv → 64×64×128
    ↓
Upsample → 128×128×128
    ↓
Concat(128×128×128, 128×128×64) ← Skip connection: 把 encoder 的 128×128×64 接過來
    ↓
128×128×192 → Conv → 128×128×64
```

**重點**：
- Skip connection = 把 encoder 對應層的特徵直接傳給 decoder
- 實現方式：Concat（在通道維度拼接）
- 目的：保留 encoder 的細節資訊（邊界、紋理）

**U-Net 的「像素級細節」vs CNN 的「像素級細節」**：
- **U-Net**：透過 skip connection 保留細節（例如邊界、紋理）
- **CNN Decoder（無 skip connection）**：只有上採樣，細節在 encoder 下採樣時丟失，decoder 無法恢復

#### CNN Decoder 如何改變通道數但保持空間尺寸？

**關鍵：使用 Padding=1 的 Conv，Stride=1**

```python
# 輸入: 256×256×C (C 可能是 768 或其他)
Conv2d(in_channels=C, out_channels=256, kernel_size=3, stride=1, padding=1)
# 輸出: 256×256×256

Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1)
# 輸出: 256×256×128

Conv2d(in_channels=128, out_channels=1, kernel_size=3, stride=1, padding=1)
# 輸出: 256×256×1
```

**為什麼 H×W 不變？**
- `stride=1`：每次移動 1 個像素
- `padding=1`：在邊緣補 0，保持尺寸
- 公式：`output_size = (input_size + 2*padding - kernel_size) / stride + 1`
  - `(256 + 2*1 - 3) / 1 + 1 = 256`

**為什麼通道數會變？**
- `out_channels` 決定輸出通道數
- 例如：`in_channels=768, out_channels=256` → 通道數從 768 變 256

**Filter 個數 = out_channels**
- `Conv2d(..., out_channels=256)` → 有 256 個 filter，每個 filter 產生一個通道

---

## 三、DINO 接 DL 的實作方案分析

### 3.1 方案 A：DINO (frozen) + 簡單 CNN Decoder

```
[ 影像 256×256 ] 
    ↓
DINOv2 (frozen) → Patch features (37×37×768)
    ↓
[ Reshape/Upsample 到 256×256×C ]
    ↓
[ 簡單 CNN Decoder：幾層 Conv + Upsample ]
    ↓
[ Mask 256×256×1 ]
```

**特點**：
- **DINO frozen**：不更新 DINO 的權重，只當 feature extractor
- **簡單 CNN Decoder**：例如 3-4 層 transposed conv，沒有 skip connection
- **訓練快**：只訓練 decoder（參數少）
- **表現**：可能不如方案 B（沒有 CNN encoder 的細節）

**如何改變通道數但保持空間尺寸？**
- 使用 `stride=1, padding=1` 的 Conv
- 例如：`Conv2d(768, 256, kernel_size=3, stride=1, padding=1)` → 256×256×256
- 最後一層：`Conv2d(128, 1, kernel_size=3, stride=1, padding=1)` → 256×256×1

---

### 3.2 方案 B：DINO + U-Net Fusion（詳細說明）

#### 完整流程

```
[ 影像 256×256 ]
    ↓
    ├─→ DINOv2 → DINO features (37×37×768)
    │
    └─→ U-Net Encoder → CNN features
         ↓
    256×256×3 → Conv → 128×128×64
         ↓
    128×128×64 → Conv → 64×64×128
         ↓
    64×64×128 → Conv → 32×32×256  ← CNN features (32×32×256)
         ↓
    [ DINO features 上採樣到 32×32×256 ]
         ↓
    [ Fusion：Concat 或 Add ]
         ↓
    [ 融合後 features (32×32×512 或 32×32×256) ]
         ↓
    [ U-Net Decoder：上採樣 + skip connection ]
         ↓
    [ Mask 256×256×1 ]
```

#### 如何融合

**方法 1：Concatenate**
```python
# DINO features: 32×32×256
# CNN features: 32×32×256
fused = torch.cat([dino_features, cnn_features], dim=1)
# 輸出: 32×32×512  ← 256 + 256 = 512
# 再接一層 Conv 降維回 256
conv = Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
# 輸出: 32×32×256
```

**方法 2：Add**
```python
# DINO features: 32×32×256
# CNN features: 32×32×256
fused = dino_features + cnn_features
# 輸出: 32×32×256（直接相加，需確保維度一致）
```

#### DINO features 如何上採樣到 32×32×256？

```python
# DINO 輸出: 37×37×768
dino_features = dino(image)  # Shape: (batch, 768, 37, 37)

# 1. 上採樣空間尺寸: 37×37 → 32×32
dino_upsampled = F.interpolate(
    dino_features, 
    size=(32, 32), 
    mode='bilinear'
)  # Shape: (batch, 768, 32, 32)

# 2. 降維通道數: 768 → 256
dino_proj = Conv2d(768, 256, kernel_size=1)(
    dino_upsampled
)  # Shape: (batch, 256, 32, 32)
```

**目的**：把 DINO features 的維度（37×37×768）對齊到 CNN features（32×32×256），這樣才能融合（concat 或 add）

#### U-Net Decoder 架構

**上採樣 + Concat + Conv**：
```
32×32×256 → Upsample → 64×64×256
    ↓
Concat(64×64×256, encoder 的 64×64×128) → 64×64×384
    ↓
Conv → 64×64×128
    ↓
Upsample → 128×128×128
    ↓
Concat(128×128×128, encoder 的 128×128×64) → 128×128×192
    ↓
Conv → 128×128×64
    ↓
... 繼續上採樣到 256×256×1
```

**為什麼 Upsample 會讓 H×W 變大？**
- 在每個像素之間插入新像素（用插值）
- `scale_factor=2`：每個維度放大 2 倍（32 → 64）
- 通道數不變（256）

**為什麼 Concat 會讓通道數相加？**
- `dim=1` 表示在通道維度拼接
- 把兩個特徵圖的通道「疊在一起」
- 通道數 = A的通道數 + B的通道數

**為什麼 Conv 時 H×W 不變，但通道數會變？**
- `stride=1, padding=1`：保持空間尺寸
- `out_channels` 決定輸出通道數
- 例如：`Conv2d(384, 128, kernel_size=3, stride=1, padding=1)` → 64×64×128

**第三個數字（通道數）怎麼來的？**
- 由 `Conv2d(..., out_channels=128)` 的 `out_channels` 參數決定
- 這是設計選擇（可以設成 64、256、512 等）

---

### 3.3 方案 C：DINO (可凍結或微調) + U-Net Decoder

```
[ 影像 256×256 ]
    ↓
DINOv2 → Patch features (37×37×768)
    ↓
[ Reshape/Upsample 到 32×32×256 ] ← 這裡要 reshape 成 U-Net decoder 需要的尺寸
    ↓
[ U-Net Decoder：有 skip connection 的結構 ]
    ↓
[ Mask 256×256×1 ]
```

**與方案 A 的差別**：

1. **Reshape**：
   - 方案 A：直接上採樣到原圖尺寸（256×256）
   - 方案 C：先 reshape 到 U-Net decoder 的輸入尺寸（例如 32×32），再用 U-Net decoder 的 skip connection 結構上採樣

2. **Decoder 複雜度**：
   - 方案 A：簡單 decoder（無 skip connection）
   - 方案 C：U-Net decoder（有 skip connection，但這裡 DINO 沒有多層特徵，所以 skip 可能用不到，或需要額外設計）

3. **DINO 是否凍結**：
   - 方案 A：DINO frozen（不更新權重）
   - 方案 C：DINO 可凍結或微調（fine-tune）

**為何方案 A 不用 reshape？**
- 因為直接上採樣到最終尺寸，不需要中間尺寸

**為什麼多層特徵才需要 Skip Connection？**
- **有 Skip Connection**：Encoder 有多層（例如 256→128→64→32），每一層保留不同層級的細節（淺層：邊界、紋理；深層：語義）
- **沒有多層特徵（例如只有 DINO 輸出）**：DINO 只輸出一個層級的特徵（例如 `37×37×768`），沒有「淺層細節」可以 skip
- **如何知道 DINO 沒有多層特徵？**：DINO 通常只輸出最後一層的 patch features。如果要多層，需要從 DINO 的中間層提取（例如 `dino.blocks[6].output`、`dino.blocks[12].output`）

---

### 3.4 方案 D：Custom Encoder（Init from DINO weights）

#### Custom Encoder 的意思

- **現有套件**：DINOv2、ViT（pretrained）
- **Custom Encoder**：你自己設計的 encoder（例如基於 ViT 但改架構），或用 DINO 權重初始化

#### Init from DINO weights

- **DINO 的權重**：DINOv2 預訓練好的參數（例如 `dinov2_vitb14`）
- **從哪裡來**：Facebook Research 公開的 pretrained model（例如 `torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')`）
- **做法**：
  ```python
  # 載入 DINO pretrained weights
  dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
  
  # 你的 custom encoder（例如 ViT 架構）
  my_encoder = MyViTEncoder()
  
  # 用 DINO 的權重初始化（只初始化相容的部分）
  my_encoder.load_state_dict(dino.state_dict(), strict=False)
  
  # 然後在分割任務上 fine-tune
  ```

---

### 3.5 凍結 vs 不凍結的差別

#### DINO Frozen（凍結）

- **意思**：DINO 的權重**不更新**，只當 feature extractor
- **優點**：
  - 訓練快（只訓練 decoder）
  - 記憶體省（DINO 不用存 gradient）
  - DINO 的強特徵不會被破壞
- **缺點**：DINO 無法適應你的任務

#### DINO 不凍結（Fine-tune）

- **意思**：DINO 的權重**會更新**，在分割任務上微調
- **優點**：
  - DINO 可以適應天空分割任務
  - 可能表現更好
- **缺點**：
  - 訓練慢、記憶體大
  - 需要小心調 learning rate（DINO 用較小的 LR）

---

## 四、待討論的問題與 Plan

### 4.1 已記錄的 Plan

**目前狀態**：已討論方案 A/B/C/D 的架構細節，但尚未決定最終實作方案。

**待決定的問題**：
1. **選擇哪個方案？**（方案 A/C 較簡單，方案 B 較複雜但可能表現更好）
2. **DINO 是否凍結？**（影響訓練速度與表現）
3. **Decoder 的設計細節**：
   - 方案 A：簡單 CNN decoder 的層數與通道數
   - 方案 B：Fusion 方式（Concat vs Add）、U-Net decoder 的 skip connection 設計
   - 方案 C：U-Net decoder 的輸入尺寸與 skip connection 設計
4. **訓練參數**：Learning rate、batch size、loss function 等
5. **評估方式**：如何與現有的 DL baseline 比較？

### 4.2 下一步

**明天討論**：
- 根據今天的架構理解，決定最終實作方案
- 討論訓練與評估的細節
- 開始實作

### 4.3 相關 Plan 記錄

**Plan 文件位置**：`C:\Users\yiching\.cursor\plans\actionable-analysis-next-step_272786bb.plan.md`

**Plan 內容摘要**：
- **目標**：以「提升整體 accuracy」為主目標，先做「取樣重平衡」A/B 實驗
- **主要步驟**：
  1. 建立可行動的基線報表（overall acc、關鍵 pair 錯誤率）
  2. Sampler Rebalance A/B 實驗（Baseline vs Class-balanced vs Mild rebalance）
  3. 把分析結果轉成「下一步動作」（Pass/Fail + 建議）
  4. 輸出可決策報告（Overall metrics、Key pairs、決策門檻、建議下一步）
- **涉及檔案**：
  - `handwriting-recognition-project/scripts/train_emnist_digits_letters.py`
  - `handwriting-recognition-project/scripts/datasets.py`
  - `handwriting-recognition-project/scripts/evaluate_emnist_digits_letters.py`
  - `handwriting-recognition-project/notes/2026-02-11_emnist36-phase1-clean-representation-analysis.md`

**注意**：此 plan 為手寫辨識專案相關，記錄於此作為參考。

---

## 五、今日結論

1. **架構理解**：
   - ViT 是 Vision Transformer，包含 Patch Embedding、Transformer Encoder、任務頭
   - DINO 是 self-supervised 的 ViT，特徵更通用
   - U-Net 是 Encoder-Decoder + Skip Connection 架構，適合分割任務
   - Encoder 下採樣提取特徵，Decoder 上採樣恢復解析度

2. **DINO 接 DL 的方案**：
   - 方案 A：DINO frozen + 簡單 CNN Decoder（簡單、快）
   - 方案 B：DINO + U-Net Fusion（複雜、可能表現更好）
   - 方案 C：DINO + U-Net Decoder（中等複雜度）
   - 方案 D：Custom Encoder（用 DINO 權重初始化）

3. **待決定**：明天根據架構理解，決定最終實作方案與細節。

---

*記錄日期：2026-02-11*
