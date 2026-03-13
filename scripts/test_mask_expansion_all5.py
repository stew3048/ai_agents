"""
對 data/image 全部 5 張圖，比較：
  A. DINOv2 baseline (thr=0.5)
  E. Mask Expansion (dil=15px, CLIPSeg>0.35)

輸出：output/prob_maps/mask_expansion_all5.png
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

PAIRS = [
    ("10066_046.jpg", "10066_mask.png", "濃霧"),
    ("10870_040.jpg", "10870_mask.png", "夜間建築物"),
    ("3888_1480.jpg", "3888_mask.png",  "天海一線"),
    ("4795_017.jpg",  "4795_mask.png",  "夜間強遮擋"),
    ("9483_017.jpg",  "9483_mask.png",  "日間建築物"),
]
IMAGE_DIR  = os.path.join(_root, "data", "image")
MASK_DIR   = os.path.join(_root, "data", "mask")
OUTPUT_DIR = os.path.join(_root, "output", "prob_maps")
CKPT_DINO  = os.path.join(_root, "outputs",
               "train_dino_segmentation_20260214_101339",
               "checkpoints", "best.pth")
CLIPSEG_ID = "CIDAS/clipseg-rd64-refined"
DILATION_RADIUS = 15
CLIP_THRESHOLD  = 0.35
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

# ─── Helper ───────────────────────────────────────────────────────────────────
struct = np.ones((DILATION_RADIUS*2+1, DILATION_RADIUS*2+1), dtype=bool)

def compute_metrics(pred, gt):
    TP = (pred & gt).sum(); FP = (pred & ~gt).sum(); FN = (~pred & gt).sum()
    iou  = TP / (TP + FP + FN + 1e-8)
    prec = TP / (TP + FP + 1e-8)
    rec  = TP / (TP + FN + 1e-8)
    return iou, prec, rec

def make_overlay(img_np, pred, gt, alpha=0.5):
    overlay = img_np.astype(np.float32).copy()
    TP = gt & pred; FP = ~gt & pred; FN = gt & ~pred
    for mask, color, a in [(TP,[50,255,50],alpha*0.3),(FP,[255,50,50],alpha*0.9),(FN,[0,100,255],alpha)]:
        m3 = np.stack([mask]*3, axis=-1)
        overlay = np.where(m3, overlay*(1-a)+np.array(color,dtype=np.float32)*a, overlay)
    return np.clip(overlay, 0, 255).astype(np.uint8)

# ─── 逐張推論 ─────────────────────────────────────────────────────────────────
results = []

for img_fname, mask_fname, scene in PAIRS:
    print(f"處理: {img_fname} ({scene})")
    img_path  = os.path.join(IMAGE_DIR, img_fname)
    mask_path = os.path.join(MASK_DIR,  mask_fname)

    image_pil = Image.open(img_path).convert('RGB')
    W, H = image_pil.size
    image_t = TF.to_tensor(image_pil).unsqueeze(0).to(device)

    # DINOv2 prob
    with torch.no_grad():
        logits = dino_model(image_t)
    dino_prob = torch.sigmoid(logits).squeeze().cpu().numpy()
    dino_prob = np.array(
        Image.fromarray((np.clip(dino_prob,0,1)*255).astype(np.uint8)).resize((W,H), Image.BILINEAR),
        dtype=np.float32) / 255.0

    # CLIPSeg prob
    inputs = processor(text=['sky'], images=image_pil, return_tensors='pt', padding=True)
    inputs = {k: v.to(device) if hasattr(v,'to') else v for k,v in inputs.items()}
    with torch.no_grad():
        out = clipseg(**inputs)
    logits_cs = out.logits.squeeze().cpu().float().numpy()
    if logits_cs.ndim != 2:
        logits_cs = logits_cs[0]
    clip_prob = (1.0 / (1.0 + np.exp(-logits_cs.astype(np.float64)))).astype(np.float32)
    clip_prob = np.array(
        Image.fromarray((clip_prob*255).astype(np.uint8)).resize((W,H), Image.BILINEAR),
        dtype=np.float32) / 255.0

    # GT
    gt_bin = np.array(
        Image.open(mask_path).convert('L').resize((W,H), Image.NEAREST),
        dtype=np.float32) / 255.0 > 0.5

    # Strategy A
    mask_A = dino_prob > 0.5
    iou_A, prec_A, rec_A = compute_metrics(mask_A, gt_bin)

    # Strategy E
    dilated = binary_dilation(mask_A, structure=struct)
    uncertain_band = dilated & ~mask_A
    supplement = uncertain_band & (clip_prob > CLIP_THRESHOLD)
    mask_E = mask_A | supplement
    iou_E, prec_E, rec_E = compute_metrics(mask_E, gt_bin)

    diff = iou_E - iou_A
    sign = "▲" if diff > 0 else "▼"
    print(f"  Baseline IoU={iou_A:.4f}  →  Mask Expansion IoU={iou_E:.4f}  {sign} {abs(diff):.4f}")

    results.append(dict(
        scene=scene, fname=img_fname,
        img_np=np.array(image_pil), gt_bin=gt_bin,
        mask_A=mask_A, mask_E=mask_E,
        iou_A=iou_A, prec_A=prec_A, rec_A=rec_A,
        iou_E=iou_E, prec_E=prec_E, rec_E=rec_E,
    ))

# ─── 彙總表 ───────────────────────────────────────────────────────────────────
print(f"\n{'場景':<12} {'Baseline IoU':>12} {'Expansion IoU':>13} {'差異':>8}  {'Prec_E':>7}  {'Rec_E':>7}")
print("-" * 65)
for r in results:
    diff = r['iou_E'] - r['iou_A']
    sign = "▲" if diff > 0 else "▼"
    print(f"  {r['scene']:<10} {r['iou_A']:>12.4f} {r['iou_E']:>13.4f} {sign}{abs(diff):>7.4f}  {r['prec_E']:>7.4f}  {r['rec_E']:>7.4f}")

# ─── 視覺化（5 列 × 3 欄：原圖 / Baseline overlay / Expansion overlay）──────
fig, axes = plt.subplots(5, 3, figsize=(15, 25))
col_titles = ['原圖', 'A. DINOv2 Baseline', 'E. Mask Expansion']
for col, title in enumerate(col_titles):
    axes[0][col].set_title(title, fontsize=13, pad=8)

for row, r in enumerate(results):
    # 原圖
    axes[row][0].imshow(r['img_np'])
    axes[row][0].set_ylabel(r['scene'], fontsize=11, rotation=0, labelpad=60, va='center')
    axes[row][0].axis('off')
    # Baseline
    ov_A = make_overlay(r['img_np'], r['mask_A'], r['gt_bin'])
    axes[row][1].imshow(ov_A)
    axes[row][1].set_title(f"IoU={r['iou_A']:.4f}", fontsize=10)
    axes[row][1].axis('off')
    # Expansion
    ov_E = make_overlay(r['img_np'], r['mask_E'], r['gt_bin'])
    axes[row][2].imshow(ov_E)
    diff = r['iou_E'] - r['iou_A']
    sign = "▲" if diff > 0 else "▼"
    axes[row][2].set_title(f"IoU={r['iou_E']:.4f}  {sign}{abs(diff):.4f}", fontsize=10)
    axes[row][2].axis('off')

patches = [
    mpatches.Patch(color=[c/255 for c in [50,255,50]], label='TP'),
    mpatches.Patch(color=[c/255 for c in [255,50,50]], label='FP'),
    mpatches.Patch(color=[c/255 for c in [0,100,255]], label='FN'),
]
fig.legend(handles=patches, loc='lower center', ncol=3, fontsize=12, framealpha=0.8)
plt.tight_layout()

out_path = os.path.join(OUTPUT_DIR, "mask_expansion_all5.png")
plt.savefig(out_path, dpi=120, bbox_inches='tight')
print(f"\n圖片已儲存：{out_path}")
