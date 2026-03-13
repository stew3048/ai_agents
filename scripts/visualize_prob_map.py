"""
產生濃霧場景（10066_046）三個模型的機率熱力圖並排比較。

輸出：output/prob_maps/10066_046_prob_comparison.png
色彩說明：藍→綠→黃→紅  =  低信心 → 高信心（JET colormap）
"""

import os, sys
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms.functional as TF
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

from models.dinov2_segmentation_decoder import create_dinov2_segmentation_model
from models.dinov2_clipseg_decoder import DINOv2CLIPSegModel

# ─── 設定 ─────────────────────────────────────────────────────────────────────
IMAGE_PATH  = os.path.join(_root, "data", "image", "10066_046.jpg")
MASK_PATH   = os.path.join(_root, "data", "mask",  "10066_mask.png")
OUTPUT_DIR  = os.path.join(_root, "output", "prob_maps")
CKPT_DINO   = os.path.join(_root, "outputs",
                "train_dino_segmentation_20260214_101339",
                "checkpoints", "best.pth")
CKPT_FUSED  = os.path.join(_root, "outputs",
                "train_dino_clipseg_20260313_184550",
                "checkpoints", "best.pth")
CLIPSEG_ID  = "CIDAS/clipseg-rd64-refined"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── 載入模型 ──────────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"設備: {device}")

def load_dino_model():
    model = create_dinov2_segmentation_model()
    state = torch.load(CKPT_DINO, map_location='cpu')
    if 'model_state_dict' in state:
        state = state['model_state_dict']
    model.load_state_dict(state)
    return model.to(device).eval()

def load_fused_model():
    model = DINOv2CLIPSegModel()
    state = torch.load(CKPT_FUSED, map_location='cpu')
    if 'model_state_dict' in state:
        state = state['model_state_dict']
    model.load_state_dict(state)
    return model.to(device).eval()

print("載入 DINOv2+CNN ...")
dino_model  = load_dino_model()
print("載入 DINOv2+CLIPSeg ...")
fused_model = load_fused_model()
print("載入 CLIPSeg ...")
processor  = CLIPSegProcessor.from_pretrained(CLIPSEG_ID)
clipseg    = CLIPSegForImageSegmentation.from_pretrained(CLIPSEG_ID).to(device).eval()
print("所有模型載入完成\n")

# ─── 讀取影像 ──────────────────────────────────────────────────────────────────
image_pil = Image.open(IMAGE_PATH).convert('RGB')
image_t   = TF.to_tensor(image_pil).unsqueeze(0).to(device)
H_orig, W_orig = image_pil.size[1], image_pil.size[0]

gt_pil  = Image.open(MASK_PATH).convert('L')
gt_np   = (np.array(gt_pil, dtype=np.float32) / 255.0)
gt_np   = np.array(Image.fromarray((gt_np * 255).astype(np.uint8)).resize(
              (W_orig, H_orig), Image.NEAREST), dtype=np.float32) / 255.0

# ─── 1. CLIPSeg 機率圖 ─────────────────────────────────────────────────────────
inputs = processor(text=['sky'], images=image_pil, return_tensors='pt', padding=True)
inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}
with torch.no_grad():
    out = clipseg(**inputs)
logits_cs = out.logits.squeeze().cpu().float().numpy()
if logits_cs.ndim != 2:
    logits_cs = logits_cs[0]
prob_clipseg = 1.0 / (1.0 + np.exp(-logits_cs.astype(np.float64)))
prob_clipseg = np.array(
    Image.fromarray((prob_clipseg * 255).astype(np.uint8)).resize((W_orig, H_orig), Image.BILINEAR),
    dtype=np.float32) / 255.0
heatmap_t = torch.tensor(prob_clipseg, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

# ─── 2. DINOv2+CNN 機率圖 ──────────────────────────────────────────────────────
with torch.no_grad():
    logits_dino = dino_model(image_t)
prob_dino = torch.sigmoid(logits_dino).squeeze().cpu().numpy()
prob_dino = np.array(
    Image.fromarray((np.clip(prob_dino, 0, 1) * 255).astype(np.uint8)).resize((W_orig, H_orig), Image.BILINEAR),
    dtype=np.float32) / 255.0

# ─── 3. DINOv2+CLIPSeg 機率圖 ─────────────────────────────────────────────────
with torch.no_grad():
    logits_fused = fused_model(image_t, heatmap_t)
prob_fused = torch.sigmoid(logits_fused).squeeze().cpu().numpy()
prob_fused = np.array(
    Image.fromarray((np.clip(prob_fused, 0, 1) * 255).astype(np.uint8)).resize((W_orig, H_orig), Image.BILINEAR),
    dtype=np.float32) / 255.0

# ─── 繪圖 ──────────────────────────────────────────────────────────────────────
titles = [
    "原圖 (10066_046)",
    "GT Mask",
    "CLIPSeg 機率圖",
    "DINOv2+CNN 機率圖",
    "DINOv2+CLIPSeg 機率圖",
]
maps = [
    np.array(image_pil),
    gt_np,
    prob_clipseg,
    prob_dino,
    prob_fused,
]

fig, axes = plt.subplots(1, 5, figsize=(22, 5))
fig.suptitle("濃霧場景 (10066_046) — 各模型機率熱力圖比較\n"
             "JET colormap：藍(低信心) → 綠/黃 → 紅(高信心)", fontsize=13, y=1.02)

for ax, title, data in zip(axes, titles, maps):
    if title.startswith("原圖"):
        ax.imshow(data)
    elif title == "GT Mask":
        ax.imshow(data, cmap='gray', vmin=0, vmax=1)
    else:
        im = ax.imshow(data, cmap='jet', vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title, fontsize=11, fontproperties='SimHei' if sys.platform=='win32' else None)
    ax.axis('off')

plt.tight_layout()
out_path = os.path.join(OUTPUT_DIR, "10066_046_prob_comparison.png")
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"已儲存：{out_path}")

# ─── 列印各模型信心統計 ────────────────────────────────────────────────────────
print("\n=== 機率圖統計（越接近 0.5 表示越不確定）===")
for name, prob in [("CLIPSeg", prob_clipseg), ("DINOv2+CNN", prob_dino), ("DINOv2+CLIPSeg", prob_fused)]:
    entropy = -(prob * np.log(prob + 1e-8) + (1 - prob) * np.log(1 - prob + 1e-8))
    print(f"  {name:<20}  mean={prob.mean():.3f}  std={prob.std():.3f}  "
          f"mean_entropy={entropy.mean():.3f}  sky_ratio={( prob>0.5).mean():.3f}")
