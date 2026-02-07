# Sky Segmentation Side Project — Method Analysis & Wrap-up

## 一句話目標（Side Project 定義）

本 side project 的目標，是理解並比較不同 vision 方法（CNN / ViT / VLM / SAM）在 sky segmentation 任務中的角色分工，特別是在 no-sky、夜晚、遮擋等困難情境下，分析哪些方法適合「先做語意判斷」，哪些方法適合「穩定大規模切割」，而非單純追求單一模型的最佳分數。

---

## 專案背景與核心問題

Sky segmentation 的挑戰不只是邊界是否切得精準，而是：

- 圖中是否真的存在天空（no-sky 問題）
- 哪一塊才是語意上正確的天空
- 在夜晚、反光、遮擋等情境下，模型是否會 hallucinate

若未先釐清「在辨識什麼」，直接做 segmentation，結果往往看似合理但語意錯誤。

---

## 方法心智模型對照

| 方法 | 在解什麼問題 | 世界觀 |
|---|---|---|
| CNN / DL Segmentation | 這塊像不像天空 | 外觀 / pattern |
| Vision Transformer (ViT) | Patch 與 patch 的全局關係 | Context / 關係 |
| Vision-Language Model (VLM) | 哪裡在語意上是 sky | 語意對齊 |
| Segment Anything Model (SAM) | 把指定的東西切乾淨 | 結構 / 連通 |

---

## 使用 VLM / SAM 產生 Pseudo-label 的完整流程（含具體例子）

### 為什麼需要 pseudo-label？
人工標註 sky mask 成本高、速度慢，且在 no-sky、夜晚、反光等情境下，僅用 DL 模型容易放大誤判偏差。因此改採「**語意保守但邊界乾淨**」的 VLM / SAM 作為 pseudo-label 的產生來源。

---

### 具體範例說明（從一張圖開始）

假設有一張**未標註的夜晚街景影像**：
- 上半部有微弱夜空
- 下半部是建築物與路燈反光

---

### Step 1：VLM 進行語意存在性判斷（是不是 sky？）
操作：
- 輸入影像 + 文字提示："sky"
- 取得 VLM attention map

結果：
- Attention 在影像上半部微弱但連續
- 判定：**此影像存在 sky（低光）**

對照情境：
- 若 attention 幾乎全黑（例如室內或隧道），則直接標為 **no-sky**

---

### Step 2：由 VLM attention 產生 SAM prompt
操作：
- 將 attention 高回應區轉成一個 box（上半部）
- 或在 attention 峰值處取 1–2 個 point
- 作為 SAM 的 prompt

---

### Step 3：SAM 產生 pseudo sky mask
結果：
- SAM 在 prompt 引導下，切出連續的夜空區域
- 遮蔽區（如建築邊緣）仍能保持連通
- 不會誤切路燈反光或牆面

對 no-sky 圖像：
- 不送入 SAM
- 直接產生全 0 mask（pseudo-label = no-sky）

---

### Step 4：Pseudo-label 品質控管（實務規則）
僅保留可信 pseudo-label，例如：
- sky mask 面積介於 5%–60%
- mask 主要位於影像上半部
- attention 與 mask 有高度重疊

不合格樣本直接丟棄，不進入訓練集。

---

### Step 5：使用 pseudo-label 訓練 DL segmentation
訓練資料變為：
- Input：image
- Target：pseudo sky mask

DL segmentation（如 UNet / Swin-UNet）學到的是：
- **經語意確認後的天空外觀**
- 而非反光、雪地、亮牆等錯誤 pattern

---

### Step 6：部署階段（只留下 DL）
實際部署時：
- 不再使用 VLM
- 不再使用 SAM
- 僅使用 DL segmentation 做快速推論

但：
- DL 的 sky 定義，來自 VLM / SAM 的語意保守標註

---

## 系統層級結論

- **VLM / SAM**：負責「對不對」
  - 語意存在性
  - 遮擋處理
  - pseudo-label 產生
- **DL Segmentation**：負責「穩不穩」
  - 快速推論
  - 結果一致
  - 大規模部署

---

## Bottom Line

> **VLM 解決「是什麼」  
> SAM 解決「切哪個」  
> DL 解決「怎麼穩定地一直切」**

---

## 收尾說明

本 side project 已完成方法理解、實證分析與系統層級整合，並以 pseudo-labeling 為收斂策略，於此階段正式結束。
