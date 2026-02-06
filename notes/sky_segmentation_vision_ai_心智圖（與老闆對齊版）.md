# 🎯 Sky Segmentation × Vision AI 心智圖（與老闆對齊）

> **核心一句話**：
> 不是選最強模型，而是判斷「問題壞在哪一層」，再選對工具。

---

## 一、問題定義層（最上游，老闆最在意）

### 天空是什麼問題？
- ✅ **語義區域（Semantic Region）**
- ❌ 不是 instance（不需要第幾片天空）

➡️ 問題類型：**Semantic Segmentation**

---

## 二、技術層級地圖（不要混層）

### Level 0：學習方法（How to learn）
- **Deep Learning**：學表示（representation）
  - CNN
  - Transformer

---

### Level 1：Backbone（眼睛，怎麼看圖）
- **CNN Backbone**
  - ResNet（有 skip connection，穩定、有效率）
- **Transformer Backbone**
  - ViT（patch + attention，看全局關係）

---

### Level 2：任務模型（交什麼作業）
- **Classification**：有沒有天空？
- **Detection（YOLO）**：天空大概在哪？
- **Segmentation（U-Net）**：天空精確形狀

---

### Level 3：Foundation / VLM 系統
- **DINO**：
  - Self-supervised
  - 學「視覺結構與一致性」
- **CLIP**：
  - Vision + Language
  - 學「語言定義的語義」
- **SAM**：
  - ViT + 大量 mask 訓練
  - 學「怎麼切一塊合理的區域」

---

## 三、老闆口中的 CV 是什麼？

### CV（工程語境）= OpenCV / 傳統影像處理
- 前處理：對比、去霧、gamma
- 後處理：補洞、平滑、幾何限制

➡️ **負責穩定，不負責理解**

---

## 四、CNN vs ViT vs VLM（能力分工）

- **CNN**：
  - 強在 pixel / 紋理
  - 穩定、快
  - ❌ 怕 domain shift

- **ViT**：
  - 強在全局關係
  - 吃資料 / pretrain
  - ⭕ 跨 domain 撐得久

- **VLM（DINO / CLIP / SAM）**：
  - 強在語義先驗
  - ❌ 成本高、不可控
  - ⭕ 補語義缺失

---

## 五、Failure Taxonomy（你今天最重要的產出）

### Type A：Night
- 壞在：**低階 pixel 線索**
- 語義結構：還在
- 結果：
  - U-Net ❌
  - VLM ⭕

---

### Type B：Fog
- 壞在：**語義線索弱化**
- 區域角色模糊
- 結果：
  - U-Net ❌
  - VLM ❌

---

### Type C：海天一線
- 壞在：**語義本身歧義**
- 問題不可分
- 結果：
  - 所有模型 ≈ ❌

➡️ **問題定義的上限，不是模型問題**

---

## 六、工程 Pipeline（成熟版）

```
Image
 ↓
Classifier（有沒有天空）
 ↓ yes
Anomaly Detection（是不是怪資料）
 ↓ normal
U-Net Segmentation（精準形狀）
 ↓ abnormal
VLM / 保守策略 / 不確定輸出
```

---

## 七、老闆每句話對應的意思（對齊表）

- 「先知道是什麼，再來強化」
  → 語義理解優先於影像增強

- 「DL 跟 VLM 都要走一遍」
  → 比較不同語義來源的極限

- 「要會講 failure mode」
  → Night / Fog / 海天一線 taxonomy

- 「不用鑽實作，要講概念」
  → CNN / ViT / VLM 的角色差異

---

## 八、一句你可以直接對老闆說的話

>「我現在把天空 segmentation 的問題，
> 拆成語義缺失、語義弱化、語義歧義三種 failure。
> CNN、VLM 的差異不是誰比較強，
> 而是它們各自能補哪一層，
> fog 與海天一線則是問題本身的上限。」

---

## 🎯 最終定錨

> **模型只是工具，
> failure taxonomy 才是決策依據。**

