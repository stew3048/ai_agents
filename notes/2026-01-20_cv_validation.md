# 2026-01-20 學習心得：CV Baseline 驗證

## 核心心得

我先用 toy dataset 驗證 segmentation pipeline，
傳統 CV 在這個受控場景下 IoU 有 90% 以上，
證明資料、對齊與指標計算都正確。

接下來我會逐步增加資料複雜度與真實世界變因，
觀察 CV baseline 的 failure mode，
再決定深度模型介入的時機。

---

## 今日完成事項

### 1. 傳統 CV Baseline 實作
- 檔案：`src/models/cv_baseline.py`
- 方法：HSV 色彩空間 + 顏色閾值
- 規則：找「藍色 (H: 90-130) 且 明亮 (V > 120)」的像素
- 後處理：Opening（去雜點）+ Closing（補洞）

### 2. 視覺化驗證
- 腳本：`scripts/run_cv_on_toy.py`
- 輸出：`outputs/cv_toy_vis/`
  - 原始圖片
  - 預測 mask
  - 預測 overlay（紅色=預測為天空）

### 3. 指標評估
- 腳本：`scripts/evaluate_cv_baseline.py`
- 輸出：`outputs/cv_toy_metrics.csv`
- 結果：
  - Pixel Accuracy: 94.88%
  - IoU: 90.67%
  - Dice: 92.02%

---

## 學到的概念

### 三個評估指標
| 指標 | 意義 |
|------|------|
| Pixel Accuracy | 整體猜對多少像素（容易被背景撐高） |
| IoU | 預測與 GT 的重疊程度（分割任務的標準指標） |
| Dice | 預測與 GT 的相似度（比 IoU 寬鬆） |

### 形態學後處理
- **Opening（開運算）**：先侵蝕再膨脹 → 去除小白點雜訊
- **Closing（閉運算）**：先膨脹再侵蝕 → 填補小黑洞

### 為什麼用 CV baseline 當第一個 baseline？
1. 不需要訓練，快速驗證 pipeline
2. 完全透明，知道它在做什麼
3. 方便 debug，能理解錯誤原因
4. 設定效能下限，深度學習至少要贏過它

---

## CV Baseline 的已知限制
1. 只能抓藍色天空（夕陽、陰天會失敗）
2. 藍色物體會被誤判（水面、藍色建築）
3. 規則太死，無法泛化

---

## 下一步計畫
1. 增加資料複雜度（不同顏色的天空）
2. 觀察 CV baseline 在哪些情況失敗
3. 評估是否需要引入深度學習模型
