# 天空分割三模型評估總結

> 評估日期：2026-03-13
> 執行環境：`sky_sam2` conda 環境（Python 3.10）

---

## 一、評估腳本

**主腳本路徑：**

```
sky_segmentation/scripts/eval_custom_data_three_methods.py
```

---

## 二、腳本參數說明

### 全部可用參數

| 參數 | 類型 | 預設值 | 說明 |
|---|---|---|---|
| `--data_dir` | str | `data` | 舊版掃描模式的資料根目錄（含 `image/` 與 `mask/` 子目錄） |
| `--output_dir` | str | `output` | 輸出根目錄（overlay、CSV、Excel 存放位置） |
| `--device` | str | 自動偵測 | `cpu` 或 `cuda` |
| `--test_list` | str | `None` | test_list.txt 路徑，每行格式：`skyfinder_xxx/images/yyy.jpg` |
| `--base_data_dir` | str | `data/data` | test_list 模式下 `skyfinder_xxx/` 的根目錄 |
| `--no_grounding` | flag | False | 關閉 GroundingDINO+SAM 評估 |
| `--no_clipseg` | flag | False | 關閉 CLIPSeg 評估 |
| `--no_dino` | flag | False | 關閉 DINO-segmentation 評估 |
| `--filter_camera` | str | `None` | 只評估指定 camera，例如 `skyfinder_3888` |
| `--merge_existing` | str | `None` | 指定已存在 CSV 路徑，將新結果合併後重算平均並覆寫 |

### 常用呼叫範例

#### 標準：跑全部 test_list 的三個模型

```bash
C:\Users\yiching\Anaconda3\envs\sky_sam2\python.exe scripts/eval_custom_data_three_methods.py \
  --test_list data/test_list.txt \
  --base_data_dir data/data \
  --output_dir output/test_overlay
```

#### 補評估特定 camera 並回填現有 Excel

```bash
C:\Users\yiching\Anaconda3\envs\sky_sam2\python.exe scripts/eval_custom_data_three_methods.py \
  --test_list data/test_list.txt \
  --base_data_dir data/data \
  --output_dir output/test_overlay \
  --filter_camera skyfinder_3888 \
  --merge_existing output/test_overlay/test_data_metrics.csv
```

#### 只跑單一方法（加速測試用）

```bash
# 只跑 DINO-segmentation
... --no_grounding --no_clipseg

# 只跑 GroundingDINO+SAM
... --no_clipseg --no_dino
```

#### 未來新資料集（test_list_2）

```bash
C:\Users\yiching\Anaconda3\envs\sky_sam2\python.exe scripts/eval_custom_data_three_methods.py \
  --test_list data/test_list_2.txt \
  --base_data_dir data/data \
  --output_dir output/testlist2_overlay
```

---

## 三、評估資料說明

### 資料結構

```
sky_segmentation/
├── data/
│   ├── test_list.txt              ← 指定評估影像清單
│   └── data/                      ← 實際影像與 mask 存放位置
│       ├── skyfinder_10066/
│       │   ├── images/            ← 原始影像（.jpg）
│       │   └── masks/             ← GT mask（.png，與影像同名）
│       ├── skyfinder_10870/
│       ├── skyfinder_1093/
│       ├── skyfinder_21444/
│       ├── skyfinder_3888/
│       ├── skyfinder_4795/
│       ├── skyfinder_9112/
│       ├── skyfinder_9291/
│       └── skyfinder_9483/
```

### test_list.txt 組成（共 116 張）

| Camera | 張數 |
|---|---|
| skyfinder_10066 | 5 |
| skyfinder_10870 | 10 |
| skyfinder_1093 | 9 |
| skyfinder_21444 | 10 |
| skyfinder_3888 | 21 |
| skyfinder_4795 | 32 |
| skyfinder_9112 | 10 |
| skyfinder_9291 | 9 |
| skyfinder_9483 | 10 |
| **合計** | **116** |

---

## 四、三個評估模型說明

| 模型 | 類型 | Checkpoint / 模型來源 |
|---|---|---|
| **GroundingDINO + SAM** | Zero-shot，文字提示偵測＋SAM 分割 | HuggingFace `IDEA-Research/grounding-dino-tiny` + `facebook/sam2-hiera-small` |
| **CLIPSeg** | Zero-shot，CLIP 視覺語言分割 | HuggingFace `CIDAS/clipseg-rd64-refined` |
| **DINO-segmentation** | In-domain 訓練模型 | `outputs/train_dino_segmentation_20260214_101339/checkpoints/best.pth` |

---

## 五、評估指標定義

| 指標 | 公式 | 說明 |
|---|---|---|
| **IoU** | TP / (TP + FP + FN) | 預測與 GT 的交集比聯集，最綜合的指標 |
| **Precision** | TP / (TP + FP) | 預測為天空中，真正是天空的比例 |
| **Recall** | TP / (TP + FN) | GT 中是天空，且被正確預測出來的比例 |
| **FP rate** | FP / (TN + FP) | 非天空像素中被誤標為天空的比例 |
| **FN rate** | FN / (TP + FN) | 天空像素中被漏標的比例（= 1 - Recall） |

---

## 六、評估結果（116 張）

### 量化指標

| 方法 | IoU ↑ | Precision ↑ | Recall ↑ | FP rate ↓ | FN rate ↓ |
|---|---|---|---|---|---|
| GroundingDINO+SAM | 0.7282 | 0.7393 | **0.9030** | 0.1322 | 0.0970 |
| CLIPSeg | 0.7619 | 0.8510 | 0.8883 | 0.0510 | 0.1117 |
| **DINO-segmentation** | **0.9204** | **0.8823** | 0.8632 | **0.0105** | **0.0506** |

### 各方法特性分析

#### GroundingDINO+SAM：「撒大網」型

GDino 用文字提示（"sky"）找到 bounding box，SAM 再將框內可分割的區域全部框進來，傾向**過度預測（over-segmentation）**。

- ✅ Recall 最高（0.903）：幾乎不漏掉任何天空像素
- ❌ FP rate 最高（0.132）：大量非天空區域被誤標
- ❌ Precision 最低（0.739）：預測出的像素中有 26% 不是天空
- ❌ IoU 最低（0.728）：FP 過多嚴重拖累整體分數

#### CLIPSeg：平衡型（Zero-shot）

透過 CLIP 的視覺語言對齊能力做分割，不需要訓練資料，表現介於兩者之間。

- FP rate（0.051）與 FN rate（0.112）相對平衡
- 無需任何訓練即達到 IoU 0.762，是 zero-shot 方法中表現穩定的選擇

#### DINO-segmentation：「精準狙擊」型

基於 DINOv2 在本資料集上 in-domain 訓練，學到了天空的精確邊界，傾向**保守預測（under-segmentation）**。

- ✅ IoU 最高（0.920）：FP 與 FN 都控制得最好
- ✅ FP rate 最低（0.011）：幾乎零誤報
- ⚠️ Recall 略低（0.863）：極少數邊緣天空像素未被標到

### 為什麼 IoU 最好的模型 Recall 反而最低？

IoU 同時懲罰 FP 和 FN：

```
IoU = TP / (TP + FP + FN)
```

GDino+SAM 雖然 FN 很少（Recall 高），但 **FP 極多**，把分母撐大，IoU 因此大幅下降。
DINO-seg 雖然 FN 比 GDino 多一些（Recall 略低），但幾乎沒有 FP，IoU 反而最高。

```
直覺示例（假設 GT sky = 1000 px）：

GDino+SAM：TP=900, FP=400, FN=100
  Recall = 900/1000 = 0.90  ← 高
  IoU    = 900/1400 = 0.64  ← 被 FP 嚴重壓低

DINO-seg：TP=860, FP=10, FN=140
  Recall = 860/1000 = 0.86  ← 略低
  IoU    = 860/1010 = 0.85  ← FP 極少，IoU 反而更高
```

模型行為光譜：

```
     過度預測（撒大網）  ←──────────────────→  保守精準
     GDino+SAM            CLIPSeg            DINO-seg

Recall：  高  ──────────────────────────────→  低
FP rate： 高  ──────────────────────────────→  低
IoU：     低  ──────────────────────────────→  高
```

---

## 七、輸出檔案位置

```
output/test_overlay/
├── test_data_metrics.csv        ← 348 筆 per-image 指標 + 3 方法 summary
├── test_data_metrics.xlsx       ← 同上，Excel 格式（可直接開啟）
├── grounding_dino_sam_overlays/ ← 116 張 overlay 圖
├── clipseg_overlays/            ← 116 張 overlay 圖
└── dino_segmentation_overlays/  ← 116 張 overlay 圖
```

> **Overlay 色彩說明：** 綠色 = TP（正確預測天空）、紅色 = FP（誤標非天空）、藍色 = FN（漏標天空）

---

## 八、深度分析：Camera 難度、最適方法、差異最大影像

> 分析腳本：`sky_segmentation/analyze_results.py`
> 資料來源：`output/test_overlay/test_data_metrics.csv`（348 筆，116 張 × 3 方法）
>
> ⚠️ **skyfinder_21444 已從以下分析中排除**
> 該 camera 的 GT mask 全為黑色（所有影像均無天空），GDino 與 CLIPSeg 的 `load_gt_mask` 遇到全黑 mask 會回傳 `has_gt=False`，導致 IoU 無法計算。DINO 雖對全黑 GT 回傳 IoU=1.0（因預測也全黑），但此結果無實質意義，不納入難度排名。

---

### Q1：哪個 Camera 最難預測？

以三個方法的 IoU 平均值排序（由低到高）：

| 排名（難→易）| Camera | 三方法平均 IoU | GDino | CLIPSeg | DINO |
|---|---|---|---|---|---|
| 🔴 最難 | **skyfinder_10870** | 0.6222 | 0.3523 | 0.6026 | 0.9117 |
| 🔴 | **skyfinder_9291** | 0.7016 | 0.6105 | 0.6349 | 0.8594 |
| 🟡 | skyfinder_3888 | 0.7764 | 0.6413 | 0.7648 | 0.9231 |
| 🟡 | skyfinder_10066 | 0.7862 | 0.7719 | 0.8590 | 0.7277 |
| 🟡 | skyfinder_4795 | 0.8169 | 0.7524 | 0.7649 | 0.9333 |
| 🟢 | skyfinder_1093 | 0.8669 | 0.8303 | 0.8720 | 0.8984 |
| 🟢 | skyfinder_9112 | 0.8817 | 0.9609 | 0.7637 | 0.9205 |
| 🟢 最易 | **skyfinder_9483** | 0.9377 | 0.9688 | 0.8708 | 0.9736 |

**觀察：**

- `skyfinder_10870` 三方法平均 IoU 只有 **0.622**，是最難的場景。GDino 只拿到 0.35，幾乎完全失效；CLIPSeg 也只有 0.60。只有靠 in-domain 訓練的 DINO 才達到 0.91，顯示這種場景對 zero-shot 方法極度困難（推測場景光線或天空形狀較特殊）。
- `skyfinder_9483` 是最容易的場景，三方法都在 0.87 以上，即使是 zero-shot 的 GDino 也達到 0.97。
- 整體規律：DINO-seg 在各 camera 的表現都很穩定；GDino 和 CLIPSeg 則隨場景差異大幅波動。

---

### Q2：每個 Camera 最適合哪個方法？

（以各 camera 內 IoU 最高的方法為「最適合」）

| Camera | 最適合方法 | 最佳 IoU | GDino | CLIPSeg | DINO | 備註 |
|---|---|---|---|---|---|---|
| skyfinder_10066 | **CLIPSeg** | 0.8590 | 0.7719 | **0.8590** | 0.7277 | ⭐ DINO 在此反而最差 |
| skyfinder_10870 | DINO | 0.9117 | 0.3523 | 0.6026 | **0.9117** | GDino 幾近失效 |
| skyfinder_1093 | DINO | 0.8984 | 0.8303 | 0.8720 | **0.8984** | 三方法差距小 |
| skyfinder_3888 | DINO | 0.9231 | 0.6413 | 0.7648 | **0.9231** | |
| skyfinder_4795 | DINO | 0.9333 | 0.7524 | 0.7649 | **0.9333** | |
| skyfinder_9112 | **GDino** | 0.9609 | **0.9609** | 0.7637 | 0.9205 | ⭐ 唯一 GDino 勝出的場景 |
| skyfinder_9291 | DINO | 0.8594 | 0.6105 | 0.6349 | **0.8594** | |
| skyfinder_9483 | DINO | 0.9736 | 0.9688 | 0.8708 | **0.9736** | GDino 緊追其後（差 0.005）|

**觀察：**

- DINO-segmentation 在 8 個 camera 中贏了 **6 個**，但並非無敵。
- `skyfinder_9112` 是唯一 **GDino 勝出**的場景（0.961 vs DINO 0.921）。推測這個場景的天空輪廓清晰直觀，bounding box 偵測法反而更精準。
- `skyfinder_10066` 是唯一 **CLIPSeg 勝出且 DINO 最差**的場景。DINO 在此僅 0.728，低於 CLIPSeg（0.859）和 GDino（0.772），值得查看這 5 張影像的特性。
- 結論：**若有訓練資料，DINO 幾乎是最佳選擇；若無訓練資料，CLIPSeg 整體比 GDino 更穩定**。

---

### Q3：方法差異最顯著的影像

（最適合放在報告中展示各方法優劣的影像）

#### 類型一：GDino 完全失效（IoU=0），DINO 表現完美

| 排名 | 影像路徑 | GDino IoU | CLIPSeg IoU | DINO IoU | 差距 |
|---|---|---|---|---|---|
| 1 | `skyfinder_4795/654.jpg` | **0.0000** | 0.8355 | **0.9370** | 0.937 |
| 2 | `skyfinder_4795/664.jpg` | **0.0000** | 0.5465 | **0.9360** | 0.936 |
| 3 | `skyfinder_4795/671.jpg` | **0.0000** | 0.6420 | **0.9321** | 0.932 |
| 4 | `skyfinder_10870/100.jpg` | **0.0000** | 0.7280 | **0.9248** | 0.925 |
| 5 | `skyfinder_3888/1472.jpg` | **0.0000** | 0.6280 | **0.9225** | 0.923 |

這類影像最能展示 in-domain 訓練的絕對優勢：GDino 完全偵測不到天空（IoU=0，表示完全沒有框到任何天空區域），DINO 卻接近完美。

#### 類型二：三方法都有顯著差距（適合展示梯度差異）

| 影像路徑 | GDino IoU | CLIPSeg IoU | DINO IoU |
|---|---|---|---|
| `skyfinder_10870/093.jpg` | 0.2666 | 0.4919 | **0.9203** |
| `skyfinder_10870/094.jpg` | 0.2682 | 0.4923 | **0.9095** |
| `skyfinder_10870/096.jpg` | 0.2646 | 0.4142 | **0.9023** |

這類影像三個方法的 IoU 呈現明顯梯度（GDino < CLIPSeg < DINO），適合在報告中用並排圖表對比。

#### 類型三：⭐ 反例——CLIPSeg 勝出，DINO 表現最差

| 影像路徑 | GDino IoU | CLIPSeg IoU | DINO IoU |
|---|---|---|---|
| `skyfinder_1093/084.jpg` | 0.0000 | **0.7063** | 0.2503 |

這張影像極為特殊：CLIPSeg 是三者中表現最好的，而 DINO（通常最強）只有 0.25。這說明 **in-domain 訓練並非萬能**，某些特殊場景仍可能是 zero-shot 方法的強項。報告中加入這張可增加論述的完整性。

---

### 報告選圖建議

| 目的 | 建議影像 | 說明 |
|---|---|---|
| 展示最大方法差異 | `skyfinder_4795/654.jpg` | GDino=0.00 / CLIPSeg=0.84 / DINO=0.94 |
| 展示梯度差異 | `skyfinder_10870/093.jpg` | GDino=0.27 / CLIPSeg=0.49 / DINO=0.92 |
| 展示反例（DINO 失靈） | `skyfinder_1093/084.jpg` | GDino=0.00 / CLIPSeg=0.71 / DINO=0.25 |

---

## 九、視覺化分析頁面（HTML Report）

### 腳本位置

`
sky_segmentation/build_analysis_page.py
`

### 執行方式

`ash
python build_analysis_page.py
`

輸出位置：output/test_overlay/analyze/index.html
可攜打包：output/test_overlay/analyze.zip（含所有圖片，解壓後直接用瀏覽器開啟）

---

### 頁面結構

| 區塊 | 說明 |
|---|---|
| 五個極端情境逐一比較 | 原始影像 ＋ 三個方法 overlay，各附指標 badge 與手寫評語 |
| 極端天空情境雷達圖 | Model_Iou-Comparison.png，右側附各方法文字分析 |
| 全資料集平均指標表 | 含 IoU / Precision / Recall / FP rate / FN rate / TNR（負樣本） |

---

### 五個選定情境

| 情境描述 | Camera | 影像 |
|---|---|---|
| 濃霧場景 | skyfinder_10066 | 046.jpg |
| 夜間建築物 | skyfinder_10870 | 096.jpg |
| 天海一線 | skyfinder_3888 | 1518.jpg |
| 夜間強遮擋 | skyfinder_4795 | 653.jpg |
| 日間建築物 | skyfinder_9112 | 097.jpg |

---

### 全資料集平均指標計算方式

#### IoU 均值
- **GroundingDINO+SAM / CLIPSeg**：僅計算有有效輸出的 **106 張**（排除 skyfinder_21444 無輸出之 10 張）
- **DINOv2 + CNN**：計算完整 **116 張**（skyfinder_21444 的 10 張 IoU=1.0 納入）

理由：DINOv2 + CNN 在無天空影像上正確輸出空遮罩（IoU=1.0），數值有意義；
其他兩個方法推論失敗（無輸出），強制計為 0 會扭曲該方法在有天空場景的真實能力。

#### TNR（True Negative Rate / 特異度）
- 定義：TNR = 1 − FP rate
- 衡量「沒有天空時，模型能正確判斷沒有天空」的能力
- **僅針對 skyfinder_21444 的 10 張無天空影像單獨計算**，不混入整體 IoU 均值

#### 負樣本測試結果（skyfinder_21444，10 張 GT 全黑）

| 方法 | TNR | 說明 |
|---|---|---|
| DINOv2 + CNN | **1.0000** | 完美判定為空 |
| GroundingDINO+SAM | 推論失敗 | 無有效輸出，fp_rate 無法計算 |
| CLIPSeg | 推論失敗 | 無有效輸出，fp_rate 無法計算 |

#### 方法顯示名稱對應（CSV key → HTML 顯示）

| CSV key | HTML 顯示名稱 |
|---|---|
| GroundingDINO+SAM | GroundingDINO+SAM |
| CLIPSeg | CLIPSeg |
| DINO-segmentation | DINOv2 + CNN |

> 修改顯示名稱只需編輯 uild_analysis_page.py 中的 DISPLAY_NAMES 字典，不影響任何資料查找邏輯。
