import os, sys
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
notes_path = os.path.join(_root, "notes", "2026-03-19-模型訓練機制與Loss函數面試準備.md")

appendix = """
---

## 五、CNN Decoder 架構介紹

### 架構圖

```
DINOv2 輸出（凍結）
  最後 4 層 patch tokens，每層 384 維
  → concatenate → (B, 1536, h, w)     # h = H/14, w = W/14
        ↓
  Block 1：1×1 Conv, 1536 → 512       # 通道壓縮，降維
           + BatchNorm + ReLU
        ↓
  Block 2：3×3 Conv, 512 → 512        # 空間特徵萃取
           + BatchNorm + ReLU
        ↓
  Block 3：3×3 Conv, 512 → 256        # 繼續萃取 + 降維
           + BatchNorm + ReLU
        ↓
  Block 4：1×1 Conv, 256 → 1          # 輸出單通道 logit（無 BN、無 ReLU）
        ↓
  Bilinear Upsample → 還原至原圖尺寸
        ↓
  Sigmoid → 每個 pixel 的天空機率
```

### 程式碼對應

```python
self.decoder = nn.Sequential(
    conv_block(1536, 512, kernel_size=1),  # Block 1: 1x1, 降維
    conv_block(512,  512, kernel_size=3),  # Block 2: 3x3, 空間萃取
    conv_block(512,  256, kernel_size=3),  # Block 3: 3x3, 空間萃取 + 降維
    nn.Conv2d(256, 1, kernel_size=1),      # Block 4: 1x1, 輸出 logit
)
# conv_block = Conv2d + BatchNorm2d + ReLU
```

### 面試介紹說法

> 「Decoder 共四個 Block，每個 Block 由 Conv2d、BatchNorm、ReLU 組成。
>
> 第一個 Block 用 **1×1 卷積**把 1536 個 channel 壓縮到 512，目的是降維——DINOv2 輸出的 1536 維太多，先用 1×1 conv 做一次通道方向的線性組合，提煉出最有用的 512 個特徵。
>
> 第二、三個 Block 用 **3×3 卷積**，這是有感受野的空間卷積，負責萃取每個位置附近的局部空間結構，讓模型理解周圍的脈絡，512 → 512 → 256 逐步濃縮。
>
> 最後一個 Block 用 **1×1 卷積**把 256 個 channel 壓縮到 1，輸出每個位置的 logit 分數。
>
> 最後做一次 Bilinear Upsample，把 DINOv2 patch 尺度（原圖的 1/14）還原回原始圖片尺寸，再接 Sigmoid 得到每個 pixel 的天空機率。」

---

### 可能被追問的設計決策

**Q：為什麼第一層用 1×1 而不是 3×3？**
> 「1×1 conv 只做 channel 方向的線性組合，不涉及空間，目的純粹是降維。1536 → 512 的壓縮用 1×1 比 3×3 更高效，也避免在感受野還不夠大的情況下就做空間萃取。」

**Q：為什麼要接 BatchNorm？**
> 「BatchNorm 讓每層輸出的分布穩定在固定的均值和方差，避免梯度消失或爆炸，讓訓練更穩定、收斂更快。」

**Q：為什麼最後一層不接 BatchNorm 和 ReLU？**
> 「最後一層輸出的是 logit，要保留負值——因為 Sigmoid 需要負數來表達『不是天空』的低機率。接 ReLU 會把負值砍掉，接 BatchNorm 會改變分布，兩者都會破壞 logit 的語義，所以最後一層只用純 Conv。」
"""

with open(notes_path, 'a', encoding='utf-8') as f:
    f.write(appendix)
print(f"已補充至：{notes_path}")
