import matplotlib.pyplot as plt
import numpy as np

# 1. 準備數據
categories = ['Heavy Fog', 'Night Reflection', 'Sky-Sea Line', 'Night Obstruction', 'Day Buildings']
N = len(categories)

# 各模型的 IoU 數據
gdino_iou = [0.6799, 0.2646, 0.5356, 0.1275, 0.9624]
clipseg_iou = [0.8269, 0.4142, 0.5511, 0.6235, 0.8451]
dino_iou = [0.6041, 0.9023, 0.9134, 0.9274, 0.9334]

# 為了讓雷達圖閉合，需要重複第一個數值
gdino_iou += gdino_iou[:1]
clipseg_iou += clipseg_iou[:1]
dino_iou += dino_iou[:1]

# 計算角度
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

# 2. 開始繪圖
plt.figure(figsize=(10, 8))
ax = plt.subplot(111, polar=True)

# 畫出軸線
plt.xticks(angles[:-1], categories, color='black', size=12)

# 設定 y 軸範圍 (IoU 0~1)
ax.set_rlabel_position(30)
plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=10)
plt.ylim(0, 1.1)

# 3. 繪製各模型區域
# DINO-segmentation (你的模型 - 藍色)
ax.plot(angles, dino_iou, linewidth=2.5, linestyle='solid', label='DINO-segmentation', color='#1f77b4')
ax.fill(angles, dino_iou, '#1f77b4', alpha=0.2)

# CLIPSeg (橘色)
ax.plot(angles, clipseg_iou, linewidth=2, linestyle='solid', label='CLIPSeg', color='#ff7f0e')
ax.fill(angles, clipseg_iou, '#ff7f0e', alpha=0.1)

# GDino+SAM (綠色)
ax.plot(angles, gdino_iou, linewidth=2, linestyle='solid', label='GDino+SAM', color='#2ca02c')
ax.fill(angles, gdino_iou, '#2ca02c', alpha=0.1)

# 4. 美化圖表
plt.title('Sky Segmentation Robustness Analysis (IoU Comparison)', size=18, pad=30)
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

# 儲存圖片
plt.tight_layout()
plt.savefig('sky_segmentation_radar.png', dpi=300)
print("雷達圖已成功儲存為 sky_segmentation_radar.png")
