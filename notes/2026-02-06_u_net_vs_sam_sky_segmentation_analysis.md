# Sky Segmentation：U-Net vs SAM 的決策層級分析

> 本文件整理一系列實驗觀察，從 **實際結果反推模型原理**，說明為什麼 U-Net 與 SAM 在不同情境下呈現不同 FP / FN / IoU 行為，以及如何正確解讀這些差異。

---

## 1. 問題背景

在 sky segmentation 任務中，我們比較了 **U-Net（CNN-based semantic segmentation）** 與 **SAM（Segment Anything Model）** 在相同 test data 上的表現。

初始觀察：
- 幾乎所有情境中，U-Net 的 **FP 都偏高**
- 改用 SAM 後，在某些情境（night / no-sky / 建築反光）FP 明顯下降
- 但 **IoU 與 FN 並未全面提升，甚至在部分情境惡化**

這導致一個核心問題：
> **FP 降低是否代表 SAM 比較好？**

---

## 2. 實驗結果摘要（依情境）

| 情境 | 主要觀察結果 | 整體結論 |
|----|----|----|
| no-sky | SAM FP 明顯較低 | SAM 明顯較好（FP 是唯一指標） |
| sea-sky-confusable | DL 在 IoU、FP 較優；SAM FN 略低 | DL 較好（IoU 為主指標） |
| heavy-occlusion | DL IoU、FP、FN 均較好 | DL 明顯較好（FN 為主指標） |
| night | 平均 DL 較好；逐張 IoU 有部分 SAM 勝出 | DL 穩定；SAM 有機會但不可靠 |
| urban | DL IoU、FN 較好；SAM 僅 FP 較低 | DL 較好 |
| default | DL IoU、FN 較好；SAM 僅 FP 較低 | DL 較好 |

---

## 3. 為什麼會「一直盯著 FP」？

- FP 在視覺上最顯眼（整片亂塗）
- 即使 FN 在指標上更重要，人眼仍會先注意到 FP
- 因此在 early analysis 階段，FP 容易成為直覺上的主指標

這是**工程上合理但需要被校正的直覺**。

---

## 4. 為什麼 heavy-occlusion 會出現「FN 低但 FP 高」？

### heavy-occlusion 的本質
- 天空確實存在
- 但被樹、建築、電線等部分遮擋

### U-Net 的隱含決策偏好

U-Net 在 pixel-level supervision 下，學到的是：
> 「只要局部 pixel 像天空，就盡量往外擴，避免漏切」

結果是：
- **FN ↓（敢切）**
- **FP ↑（遮擋物被誤判）**

這其實是一種 **為降低 FN 而犧牲 FP 的策略**，
在 heavy-occlusion 這種「FN 為主指標」的情境下，本質上是合理的。

---

## 5. 為什麼「SAM 降 FP」不等於「SAM 比較好」？

### 兩種模型的錯誤偏好（decision bias）

| 模型 | 決策偏好 | 結果 |
|----|----|----|
| U-Net | 寧可多切也不要漏切 | FN 低、FP 高 |
| SAM（自動 prompt） | 寧可不切也不要亂切 | FP 低、FN 高 |

因此：
> **SAM 降低 FP，只代表它比較保守，並不代表它更懂天空。**

---

## 6. Prompt 為什麼是 SAM 的核心？

### 關鍵實驗觀察

- 使用 Cursor 自動產生 prompt 時：
  - SAM 常出現 FN 偏高
- 使用人工互動、明確點在天空區域時：
  - SAM 可切出非常準確的天空
  - IoU 高、FN 低

### 原理說明

SAM 的本質不是「自動判斷語義」，而是：
> **在給定條件（prompt）的前提下，切出最合理的區域**

- prompt = 語義 anchor / 條件
- prompt 錯或模糊 → SAM 保守不切（FN ↑）
- prompt 正確 → SAM 只需專注於邊界生成（IoU / FN 表現佳）

---

## 7. 為什麼 CNN 是「像不像」層級，而 SAM 是「合理區域」層級？

### CNN / U-Net 的決策層級

- 基於局部 convolution filter
- 依賴 pixel-level 統計特徵
- supervision 也是 pixel label

→ 決策問題本質是：
> 「這一小塊 pixel，像不像我學過的天空？」

### SAM 的決策層級

- 先生成可能存在的 **完整區域（mask proposal）**
- supervision 來自大量真實世界的完整 mask
- 再在 prompt 條件下選擇區域

→ 決策問題本質是：
> 「這一整塊，看起來像不像一個合理存在的區域？」

---

## 8. 為什麼 fog / sea-sky 情境兩者都表現有限？

- fog / 海天一線：
  - pixel 上高度相似
  - 區域結構上也合理且連續

→ 不只 pixel 線索失效，連「區域合理性」也無法區分

這屬於：
> **問題本身的可分性上限，而非模型能力不足**

---

## 9. 正確的模型定位（結論）

- **U-Net**：
  - 主力、自動化
  - FN 導向（敢切）
  - 適合 heavy-occlusion、default 場景

- **SAM**：
  - 條件化、保守
  - FP 導向（不亂切）
  - 適合作為 no-sky / night 的輔助或 fallback

- **Prompt 是關鍵開關**：
  - 沒有好 prompt，SAM 只會保守
  - 有好 prompt，SAM 的 segmentation 能力非常強

---

## 10. 最終總結（一句話）

> **這不是兩個 segmentation 模型誰比較好，
> 而是「誰負責做決策、誰負責畫邊界」的問題。**

U-Net 擅長自動判斷「哪裡可能是天空」，
SAM 擅長在給定條件下，把區域切得非常好。

---

*This document is intended to be uploaded as experiment reasoning notes, not as a benchmark claim.*

