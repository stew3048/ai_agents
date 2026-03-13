"""
對 10066_046（濃霧）測試多種 ensemble/後處理策略，比較 IoU。

策略：
  A. DINOv2+CNN baseline (threshold=0.5)
  B. Residual Compensation: max(DINO, CLIPSeg*0.8)  (threshold=0.5)
  C. Soft blend: 0.7*DINO + 0.3*CLIPSeg            (threshold=0.5)
  D. DINOv2 lower threshold (threshold=0.3)
  E. Mask Expansion: DINO mask + dilation 補 CLIPSeg>0.35 區域

輸出：output/prob_maps/fog_strategy_comparison.png
"""
import os, sys
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

import numpy as np
import torch
from PIL import Image
import torchvision.transforms.functional as TF
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.ndimage import binary_dilation
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

from models.dinov2_segmentation_decoder import create_dinov2_segmentation_model

# ─── 設定 ─────────────────────────────────────────────────────────────────────
IMAGE_PATH = os.path.join(_root, "data", "image", "10066_046.jpg")
MASK_PATH  = os.path.join(_root, "data", "mask",  "10066_mask.png")
OUTPUT_DIR = os.path.join(_root, "output", "prob_maps")
CKPT_DINO  = os.path.join(_root, "outputs",
               "train_dino_segmentation_20260214_101339",
               "checkpoints", "best.pth")
CLIPSEG_ID = "CIDAS/clipseg-rd64-refined"
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"設備: {device}")

# ─── 載入模型 ──────────────────────────────────────────────────────────────────
print("載入 DINOv2+CNN ...")
dino_model = create_dinov2_segmentation_model()
state = torch.load(CKPT_DINO, map_location='cpu')
if 'model_state_dict' in state:
    state = state['model_state_dict']
dino_model.load_state_dict(state)
dino_model.to(device).eval()

print("載入 CLIPSeg ...")
processor = CLIPSegProcessor.from_pretrained(CLIPSEG_ID)
clipseg   = CLIPSegForImageSegmentation.from_pretrained(CLIPSEG_ID).to(device).eval()
print("載入完成\n")

# ─── 推論 ─────────────────────────────────────────────────────────────────────
image_pil = Image.open(IMAGE_PATH).convert('RGB')
W, H = image_pil.size
image_t = TF.to_tensor(image_pil).unsqueeze(0).to(device)

# DINOv2 prob map
with torch.no_grad():
    logits = dino_model(image_t)
dino_prob = torch.sigmoid(logits).squeeze().cpu().numpy()
dino_prob = np.array(
    Image.fromarray((np.clip(dino_prob,0,1)*255).astype(np.uint8)).resize((W,H), Image.BILINEAR),
    dtype=np.float32) / 255.0

# CLIPSeg prob map
inputs = processor(text=['sky'], images=image_pil, return_tensors='pt', padding=True)
inputs = {k: v.to(device) if hasattr(v,'to') else v for k,v in inputs.items()}
with torch.no_grad():
    out = clipseg(**inputs)
logits_cs = out.logits.squeeze().cpu().float().numpy()
if logits_cs.ndim != 2:
    logits_cs = logits_cs[0]
clip_prob = 1.0 / (1.0 + np.exp(-logits_cs.astype(np.float64))).astype(np.float32)
clip_prob = np.array(
    Image.fromarray((clip_prob*255).astype(np.uint8)).resize((W,H), Image.BILINEAR),
    dtype=np.float32) / 255.0

# GT
gt_np = np.array(Image.open(MASK_PATH).convert('L').resize((W,H), Image.NEAREST),
                  dtype=np.float32) / 255.0
gt_bin = gt_np > 0.5

# ─── 各策略的 final_prob / final_mask ─────────────────────────────────────────
def iou(pred, gt):
    TP = (pred & gt).sum(); FP = (pred & ~gt).sum(); FN = (~pred & gt).sum()
    return TP / (TP + FP + FN + 1e-8)

strategies = {}

# A. Baseline
prob_A = dino_prob.copy()
mask_A = prob_A > 0.5
strategies['A. DINOv2 baseline\n(thr=0.5)'] = (prob_A, mask_A)

# B. Residual Compensation
prob_B = np.maximum(dino_prob, clip_prob * 0.8)
mask_B = prob_B > 0.5
strategies['B. max(DINO, CLIP×0.8)\n(thr=0.5)'] = (prob_B, mask_B)

# C. Soft blend
prob_C = 0.7 * dino_prob + 0.3 * clip_prob
mask_C = prob_C > 0.5
strategies['C. 0.7×DINO + 0.3×CLIP\n(thr=0.5)'] = (prob_C, mask_C)

# D. Lower threshold on DINOv2
prob_D = dino_prob.copy()
mask_D = prob_D > 0.30
strategies['D. DINOv2 only\n(thr=0.30)'] = (prob_D, mask_D)

# E. Mask Expansion
struct = np.ones((31, 31), dtype=bool)   # dilation radius ~15px
dilated = binary_dilation(mask_A, structure=struct)
uncertain_band = dilated & ~mask_A
supplement = uncertain_band & (clip_prob > 0.35)
mask_E = mask_A | supplement
prob_E = dino_prob.copy()  # 機率圖不變，只改 mask
strategies['E. Mask Expansion\n(dil=15px, clip>0.35)'] = (prob_E, mask_E)

# ─── 結果列印 ─────────────────────────────────────────────────────────────────
print(f"{'策略':<35} {'IoU':>6}  {'Precision':>9}  {'Recall':>7}  {'sky%':>6}")
print("-" * 70)
iou_scores = {}
for name, (_, mask) in strategies.items():
    TP = (mask & gt_bin).sum()
    FP = (mask & ~gt_bin).sum()
    FN = (~mask & gt_bin).sum()
    i   = TP/(TP+FP+FN+1e-8)
    prec= TP/(TP+FP+1e-8)
    rec = TP/(TP+FN+1e-8)
    iou_scores[name] = i
    label = name.replace('\n', ' ')
    print(f"  {label:<33} {i:.4f}  {prec:.4f}     {rec:.4f}  {mask.mean():.3f}")

# ─── 視覺化 ───────────────────────────────────────────────────────────────────
n_cols = 3 + len(strategies)   # 原圖 + GT + CLIPSeg prob + 各策略
fig, axes = plt.subplots(2, 4, figsize=(24, 12))
axes = axes.flatten()

def make_overlay(img_np, pred_bin, gt_bin, alpha=0.5):
    overlay = img_np.astype(np.float32).copy()
    TP = gt_bin & pred_bin;  FP = ~gt_bin & pred_bin;  FN = gt_bin & ~pred_bin
    for mask, color, a in [(TP,[50,255,50],alpha*0.3),(FP,[255,50,50],alpha*0.9),(FN,[0,100,255],alpha)]:
        m3 = np.stack([mask]*3, axis=-1)
        overlay = np.where(m3, overlay*(1-a)+np.array(color,dtype=np.float32)*a, overlay)
    return np.clip(overlay,0,255).astype(np.uint8)

img_np = np.array(image_pil)

# 原圖
axes[0].imshow(img_np); axes[0].set_title('原圖', fontsize=12); axes[0].axis('off')
# GT
axes[1].imshow(gt_bin, cmap='gray', vmin=0, vmax=1); axes[1].set_title('GT Mask', fontsize=12); axes[1].axis('off')
# CLIPSeg prob
im2 = axes[2].imshow(clip_prob, cmap='jet', vmin=0, vmax=1)
axes[2].set_title(f'CLIPSeg 機率圖\n(entropy≈0.49)', fontsize=11); axes[2].axis('off')
plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

# 各策略 overlay
for idx, (name, (prob, mask)) in enumerate(strategies.items()):
    ax = axes[3 + idx]
    overlay = make_overlay(img_np, mask, gt_bin)
    ax.imshow(overlay)
    iou_val = iou_scores[name]
    ax.set_title(f'{name}\nIoU={iou_val:.4f}', fontsize=10)
    ax.axis('off')

# 圖例
patches = [
    mpatches.Patch(color=[c/255 for c in [50,255,50]], label='TP (正確天空)'),
    mpatches.Patch(color=[c/255 for c in [255,50,50]], label='FP (誤判)'),
    mpatches.Patch(color=[c/255 for c in [0,100,255]], label='FN (漏判)'),
]
fig.legend(handles=patches, loc='lower center', ncol=3, fontsize=11, framealpha=0.8)
fig.suptitle('濃霧場景 10066_046 — 各 Ensemble 策略比較', fontsize=14, y=1.01)
plt.tight_layout()

out_path = os.path.join(OUTPUT_DIR, "fog_strategy_comparison.png")
plt.savefig(out_path, dpi=130, bbox_inches='tight')
print(f"\n圖片已儲存：{out_path}")
