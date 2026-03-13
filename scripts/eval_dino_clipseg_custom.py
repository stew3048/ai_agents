"""
用 DINOv2CLIPSegModel 評估 data/image & data/mask 的 5 張影像。

輸出
----
output/dino_clipseg/
  ├── {name}_overlay.png
  └── metrics.csv

Overlay 色彩：🟢 TP（正確天空）　🔴 FP（誤判）　🔵 FN（漏判）
"""

import os, sys, csv
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms.functional as TF
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation  # type: ignore

from models.dinov2_clipseg_decoder import DINOv2CLIPSegModel

# ─── 設定 ─────────────────────────────────────────────────────────────────────
CHECKPOINT = os.path.join(
    _root, "outputs",
    "train_dino_clipseg_20260313_184550",
    "checkpoints", "best.pth"
)
IMAGE_DIR  = os.path.join(_root, "data", "image")
MASK_DIR   = os.path.join(_root, "data", "mask")
OUTPUT_DIR = os.path.join(_root, "output", "dino_clipseg")
CLIPSEG_ID = "CIDAS/clipseg-rd64-refined"

# 影像 → mask 對應（以檔名前綴手動配對）
PAIRS = [
    ("10066_046.jpg",  "10066_mask.png"),
    ("10870_040.jpg",  "10870_mask.png"),
    ("3888_1480.jpg",  "3888_mask.png"),
    ("4795_017.jpg",   "4795_mask.png"),
    ("9483_017.jpg",   "9483_mask.png"),
]

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─── Helper ───────────────────────────────────────────────────────────────────

def get_clipseg_heatmap(processor, clipseg_model, image_path, device):
    """回傳 CLIPSeg sky sigmoid heatmap (H, W) float32，0~1。"""
    image = Image.open(image_path).convert('RGB')
    inputs = processor(text=['sky'], images=image, return_tensors='pt', padding=True)
    inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}
    with torch.no_grad():
        out = clipseg_model(**inputs)
    logits = out.logits.squeeze().cpu().float().numpy()
    if logits.ndim != 2:
        logits = logits[0]
    heatmap = 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))
    return heatmap.astype(np.float32)


def compute_metrics(pred_bin, gt_bin):
    """計算 IoU / Precision / Recall / FP_rate / FN_rate。"""
    TP = int((pred_bin & gt_bin).sum())
    FP = int((pred_bin & ~gt_bin).sum())
    FN = int((~pred_bin & gt_bin).sum())
    TN = int((~pred_bin & ~gt_bin).sum())
    iou  = TP / (TP + FP + FN + 1e-8) if (TP + FP + FN) > 0 else None
    prec = TP / (TP + FP + 1e-8)       if (TP + FP) > 0       else None
    rec  = TP / (TP + FN + 1e-8)       if (TP + FN) > 0       else None
    fpr  = FP / (FP + TN + 1e-8)       if (FP + TN) > 0       else None
    fnr  = FN / (FN + TP + 1e-8)       if (FN + TP) > 0       else None
    return dict(iou=iou, precision=prec, recall=rec, fp_rate=fpr, fn_rate=fnr)


def create_overlay(image_path, pred_bin, gt_bin, alpha=0.5):
    """TP=綠、FP=紅、FN=藍 疊圖。"""
    img_np = np.array(Image.open(image_path).convert('RGB'), dtype=np.float32)
    h, w = img_np.shape[:2]

    def resize_mask(m):
        if m.shape != (h, w):
            pil = Image.fromarray(m.astype(np.uint8) * 255).resize((w, h), Image.NEAREST)
            return np.array(pil) > 127
        return m

    pred_bin = resize_mask(pred_bin)
    gt_bin   = resize_mask(gt_bin)

    TP = gt_bin  &  pred_bin
    FP = ~gt_bin &  pred_bin
    FN = gt_bin  & ~pred_bin

    overlay = img_np.copy()
    green = np.array([50, 255, 50],  dtype=np.float32)
    red   = np.array([255, 50, 50],  dtype=np.float32)
    blue  = np.array([0,  100, 255], dtype=np.float32)

    for mask, color, a in [(TP, green, alpha * 0.3), (FP, red, alpha * 0.9), (FN, blue, alpha)]:
        m3 = np.stack([mask] * 3, axis=-1)
        overlay = np.where(m3, overlay * (1 - a) + color * a, overlay)

    return Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"設備: {device}")

    # 載入 DINOv2CLIPSeg 模型
    print(f"載入 DINOv2CLIPSegModel: {CHECKPOINT}")
    model = DINOv2CLIPSegModel()
    state = torch.load(CHECKPOINT, map_location='cpu')
    if 'model_state_dict' in state:
        state = state['model_state_dict']
    elif 'model' in state:
        state = state['model']
    model.load_state_dict(state)
    model.to(device).eval()
    print("  模型載入完成")

    # 載入 CLIPSeg
    print(f"載入 CLIPSeg: {CLIPSEG_ID}")
    processor = CLIPSegProcessor.from_pretrained(CLIPSEG_ID)
    clipseg   = CLIPSegForImageSegmentation.from_pretrained(CLIPSEG_ID)
    clipseg.to(device).eval()
    print("  CLIPSeg 載入完成\n")

    results = []

    for img_fname, mask_fname in PAIRS:
        img_path  = os.path.join(IMAGE_DIR, img_fname)
        mask_path = os.path.join(MASK_DIR,  mask_fname)
        name      = os.path.splitext(img_fname)[0]

        print(f"評估: {img_fname}")

        # ── CLIPSeg heatmap ─────────────────────────────────
        heatmap_np = get_clipseg_heatmap(processor, clipseg, img_path, device)
        heatmap_t  = torch.tensor(heatmap_np).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)

        # ── DINOv2CLIPSeg 推論 ───────────────────────────────
        image_pil = Image.open(img_path).convert('RGB')
        image_t   = TF.to_tensor(image_pil).unsqueeze(0).to(device)      # (1,3,H,W)
        heatmap_t = heatmap_t.to(device)

        with torch.no_grad():
            logits = model(image_t, heatmap_t)                            # (1,1,H,W)
        pred_prob = torch.sigmoid(logits).squeeze().cpu().numpy()         # (H,W)
        pred_bin  = pred_prob > 0.5

        # ── GT mask ──────────────────────────────────────────
        gt_pil  = Image.open(mask_path).convert('L')
        gt_np   = np.array(gt_pil, dtype=np.float32)
        if gt_np.max() > 1:
            gt_np = gt_np / 255.0
        gt_bin  = gt_np > 0.5

        # ── 指標 ─────────────────────────────────────────────
        m = compute_metrics(pred_bin, gt_bin)
        print(f"  IoU={m['iou']:.4f}  Prec={m['precision']:.4f}  "
              f"Recall={m['recall']:.4f}  FP={m['fp_rate']:.4f}  FN={m['fn_rate']:.4f}")

        # ── Overlay ──────────────────────────────────────────
        overlay = create_overlay(img_path, pred_bin, gt_bin)
        overlay_path = os.path.join(OUTPUT_DIR, f"{name}_overlay.png")
        overlay.save(overlay_path)
        print(f"  Overlay → {overlay_path}")

        results.append(dict(image=img_fname, **m))

    # ── CSV ──────────────────────────────────────────────────
    csv_path = os.path.join(OUTPUT_DIR, "metrics.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['image','iou','precision','recall','fp_rate','fn_rate'])
        writer.writeheader()
        writer.writerows(results)

        # 平均
        valid = [r for r in results if r['iou'] is not None]
        if valid:
            avg = {k: sum(r[k] for r in valid) / len(valid)
                   for k in ['iou','precision','recall','fp_rate','fn_rate']}
            writer.writerow(dict(image='AVERAGE', **avg))

    print(f"\nMetrics → {csv_path}")
    print("\n=== 平均指標 ===")
    for k, v in avg.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
