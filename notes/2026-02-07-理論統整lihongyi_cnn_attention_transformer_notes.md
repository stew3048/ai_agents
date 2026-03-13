# 李宏毅課程筆記整理：CNN / Self-Attention / Transformer

> 本筆記整理自今日課程吸收內容，並加入補充說明。
> 其中以 `*` 開頭的段落為補充或修正說明。

---

## 1. CNN（Convolutional Neural Network）架構

CNN 非常適合影像辨識任務。相較於 Fully Connected Network，CNN 有三個關鍵的簡化與歸納偏好（inductive bias）：

### 1.1 Receptive Field（感受野）

重要的特徵往往只佔影像的一小部分，例如鳥嘴或翅膀，因此每個 neuron 不需要與整張圖片的所有 input 相連，而只需專注在自己的「守備範圍」。

- 每個 receptive field 對應影像中的一小塊區域（例如 `3×3×channel`）
- 同一個 receptive field 中可以有多個 neuron
- 每一組參數稱為一個 **filter / kernel**
- 若有 64 個 filters，代表有 64 種不同的特徵偵測方式

### 1.2 Parameter Sharing（參數共享）

相同的 pattern（如鳥嘴）可能出現在影像中的不同位置。  
因此：
- 不需要為每個位置重新學一組參數
- 同一個 filter 可以「掃過」整張圖

這樣做的結果是：
- 每一個 filter 會產生一張 **feature map**
- 若有 64 個 filters → 產生 `H×W×64` 的新特徵圖，作為下一層的 input

### 1.3 Pooling

當算力不足或希望增加平移不變性時，可透過 pooling 縮小特徵圖大小。

- 常見為 **Max Pooling**
- 可降低 spatial resolution、減少計算量

> *補充：現代架構（如 ResNet、ConvNeXt）中，Pooling 的使用已減少，常以 stride convolution 或其他下採樣方式取代。*

### 1.4 層數與語意抽象

隨著 CNN 層數加深：
- 淺層：顏色、邊緣、紋理
- 中層：形狀、輪廓
- 深層：語意概念（如「鳥」、「車」）

> *補充：感受野的「理論大小」與「有效大小」不同，實際上深層對遠距離像素的影響仍有限，這也是 CNN 在全域建模上的限制之一。*

---

## 2. Self-Attention

以句子 **“I saw a saw”** 為例：  
若每個詞獨立丟入 NN，無法正確判斷第二個 saw 的詞性，因為缺乏上下文資訊。

Self-Attention 的核心目的就是：**讓模型在計算輸出時考慮整個上下文**。

### 2.1 基本機制

- 每個 input vector 會線性轉換成：
  - Query（Q）
  - Key（K）
  - Value（V）
- 透過 `Q · K` 計算關聯性（attention score）
- 對 score 做 softmax（範圍 0~1）
- 再加權加總 Value，得到輸出

若 input 為 `a1 ~ a4`：
- output 為 `b1 ~ b4`
- `b1` 是由 `q1` 對所有 `ki` 計算 attention，再加權加總 `vi` 得到

### 2.2 Attention Matrix

- 會形成一個 `L × L` 的 attention matrix
- 每一格代表「第 i 個 token 對第 j 個 token 的關注程度」

### 2.3 CNN vs Self-Attention

- CNN：
  - 只看 local receptive field
  - 有強 inductive bias
  - 資料量小時表現好
- Self-Attention / ViT：
  - 可看整張圖（global interaction）
  - 高彈性（flexibility）
  - 需要大量資料，否則容易 overfitting

> *補充：ViT 通常需要搭配 pretraining（如 ImageNet-21K、JFT）或強 data augmentation 才能超越 CNN。*

> *補充：Self-Attention 的計算複雜度為 O(L²)，這也是長序列任務的主要瓶頸。*

---

## 3. Transformer

Transformer 是一種 **Sequence-to-Sequence** 模型，特別適合處理：
- Input 長度不固定的問題
- 語音辨識、翻譯、物件偵測等任務

### 3.1 Encoder

- Input：一串 vectors
- Output：數量相同的一串 vectors
- 核心模組：
  - Multi-Head Self-Attention
  - Feed Forward Network
  - Layer Normalization + Residual Connection

> *補充：這些設計能穩定訓練深層模型，並改善 gradient flow。*

### 3.2 Decoder

Decoder 分為兩種：

#### Auto-Regressive (AR)

- 使用 **masked self-attention**
- 每次只能看到過去的 token
- 有 Start token
- 一個一個預測直到 End token

#### Non-Auto-Regressive (NAR)

- 可一次平行預測多個 token
- 預測速度快
- 但整體 performance 通常較差

> *補充：NAR 常用於即時翻譯、低延遲場景，但需搭配額外技巧（如 iterative refinement）改善品質。*

### 3.3 Cross Attention

- Decoder 會對 Encoder 的輸出做 attention
- 用來對齊 input 與 output sequence

### 3.4 Training vs Inference Gap

- 訓練時：Decoder input = Start token + 正確答案（Teacher Forcing）
- 測試時：只能用 model 自己預測的結果接龍

因此：
- 容易產生 error accumulation

改善方式：
- 在訓練時加入噪聲或 bias
- 讓模型學會處理「不完美輸入」

> *補充：這類方法包含 Scheduled Sampling、Label Smoothing 等技巧。*

---

## 總結心法

- CNN = 強先驗、資料效率高、局部關係
- Attention = 高彈性、全域關係、資料需求高
- Transformer = Attention 的完整工程化實現，適用於序列與跨模態任務


---

## 老闆會點頭的理解稿（工程＋人話版）

> 這一段不是教科書解釋，而是「你真的懂在幹嘛」的說法。

### 為什麼 CNN 在小資料時特別強？

CNN 其實是在模型裡 **先幫你假設好世界長什麼樣子**：

- 重要的東西通常是「局部的」（邊緣、角落、器官）
- 同樣的東西會出現在不同位置（鳥嘴不會只長在正中央）
- 位移一點點，不應該就完全認不出來

所以 CNN 等於在一開始就說：  
>「我只看小塊、我會重複用同一套規則、我不亂猜全局關係。」

這讓它：
- 參數變少
- 搜尋空間變小
- **在資料不多時更不容易亂學（overfit）**

👉 老闆通常會認同這句話：  
**「CNN 不是比較聰明，是比較不容易學壞。」**

---

### 為什麼 Attention / Transformer 那麼吃資料？

Attention 很誠實，它幾乎不幫你做任何假設：

- 任兩個 token 都可能有關係
- 左上角的 pixel 也可能影響右下角
- 關係是資料自己學出來的

這代表：
- 表達能力超強
- 彈性超高
- **但如果資料不夠，它會亂連關係**

👉 可以這樣講給老闆聽：  
**「Attention 就像開放式考題，資料少時很容易寫歪；資料夠多時，天花板就很高。」**

---

### 為什麼資料多時 ViT 會超過 CNN？

當資料夠多時：
- CNN 的限制會變成天花板
- Attention 能學到更長距離、更抽象的關係

例如：
- 「這一整片顏色＋下面的結構＋遠方的線條」一起代表天空
- 而不是只靠局部邊緣或顏色

👉 一句話總結：  
**「CNN 用的是人類給的規則；Transformer 用的是資料自己歸納的規則。」**

---

### Transformer Encoder / Decoder 用人話怎麼講？

- Encoder：  
  >「我先把整個 input 看懂，每個位置都帶著『全局理解後』的版本。」

- Decoder（AR）：  
  >「我一次只講一個字，但我知道我前面講過什麼，也知道你 input 在幹嘛。」

- Cross-Attention：  
  >「我現在要講這個字時，應該回頭看 input 的哪一段？」

---

### 為什麼訓練跟測試會有落差？

因為：
- 訓練時老師會偷看答案（Teacher Forcing）
- 測試時要自己接龍

就像：
>「考駕照時旁邊有人幫你踩煞車，上路後就沒有了。」

所以實務上會：
- 故意在訓練時製造一點錯誤
- 讓模型學會怎麼從錯誤中活下來

👉 老闆通常會喜歡這個觀點：  
**「好的模型不是沒犯錯，而是犯錯後不會整個崩掉。」**

---

### 最後一個面試級總結

- CNN：穩、快、省資料，適合工程落地
- Attention：彈性高、可擴充、吃資料
- Transformer：把 Attention 變成可堆疊、可擴模的工程解法

👉 如果只能講一句話：  
**「選模型不是誰比較潮，而是你手上有多少資料、算力、以及錯一次能不能接受。」**



---

## 面試 5 分鐘講稿版本（CNN → Attention → Transformer）

> 情境：面試官問「你怎麼看 CNN、Attention、Transformer 的差異與取捨？」  
> 目標：5 分鐘內讓對方覺得你 **觀念清楚、做過選型、有工程感**

---

### 開場（約 30 秒）

如果我用一句話總結：  
**CNN、Attention、Transformer 不是誰取代誰，而是對「世界長怎樣」的假設不同。**  
選哪一個，取決於資料量、算力，以及任務能不能容錯。

---

### CNN（約 1.5 分鐘）

CNN 的核心精神其實很簡單：  
>「影像裡重要的資訊通常是局部的，而且會重複出現。」

所以 CNN 一開始就幫你把規則寫進模型裡：
- 只看小區域（receptive field）
- 同一套規則到處用（parameter sharing）
- 對小位移不敏感（pooling / stride）

這帶來的好處是：
- 參數少、搜尋空間小
- **在資料不多時很穩，不容易 overfit**

所以在工程上，只要資料量有限、任務偏視覺、需要穩定落地，  
CNN 通常還是第一個 baseline。

---

### Attention / ViT（約 1.5 分鐘）

Attention 則是完全相反的哲學：  
>「我不先假設什麼重要，讓資料自己告訴我。」

任何兩個位置都可以建立關係，  
這讓模型：
- 彈性非常高
- 可以學到長距離、全域的關聯

但代價是：
- 參數多
- **非常吃資料**

所以在資料少的時候，Attention 很容易亂連關係；  
但當資料量夠大時，它的上限會超過 CNN。  
這也是為什麼 ViT 在大規模 pretraining 下能贏過傳統 CNN。

---

### Transformer（約 1 分鐘）

Transformer 可以理解成：  
>「把 Attention 變成一個可堆疊、可工程化的架構。」

Encoder 負責把 input 變成「帶有全域理解」的表示，  
Decoder 則負責一步一步生成 output，  
中間用 cross-attention 對齊 input 跟 output。

這讓 Transformer 特別適合：
- 長度不固定的 input
- 語言、語音、甚至跨模態任務

---

### 收尾（約 30 秒）

所以對我來說，選模型時我會先問三個問題：
1. 我手上有多少資料？
2. 需要多快、能不能平行？
3. 錯一次，系統能不能接受？

👉 如果資料少、要穩定 → CNN  
👉 如果資料多、要彈性 → Attention / Transformer  

**模型不是越新越好，而是越適合當下條件越好。**

---

>（如果面試官點頭，通常就代表：OK，你可以進下一題了）

