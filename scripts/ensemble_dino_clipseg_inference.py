"""
Ensemble Inference：DINOv2Segmentation + CLIPSeg 加權融合推論

- 模型：同時載入 DINOv2Segmentation 與 CLIPSeg，同一張圖分別推論後融合
- 融合模式（預設 spatial）：
  - 空間加權 (spatial)：上半部 (y < 0.4) 用 0.5*DINO + 0.5*CLIP 補足天空漏檢；
    下半部 (y >= 0.4) 用 0.9*DINO + 0.1*CLIP 壓制反光
  - 均勻 (uniform)：整張圖 P_final = w_dino*P_dino + w_clip*P_clip（可 --w_dino/--w_clip）
- 二值化門檻：固定 0.5
- 輸出：融合二值化 Mask + Overlay（TP=綠、FP=紅、FN=藍）
"""

import os
import sys
import argparse
import numpy as np
import torch
from PIL import Image
from collections import defaultdict

_script_dir = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_script_dir)
sys.path.insert(0, _root)
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

from models import create_dinov2_segmentation_model
from scripts.train_in_domain import load_list_file

try:
    from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
    CLIPSEG_AVAILABLE = True
except ImportError:
    CLIPSEG_AVAILABLE = False


def get_p_dino(model_dino, img_tensor, device, orig_h, orig_w):
    """
    取得 DINOv2Segmentation 的機率圖，並 resize 至 (orig_h, orig_w)。
    img_tensor: (1, 3, H, W) 0~1，已為 14 的倍數或會由模型內部對齊。
    """
    with torch.no_grad():
        logits = model_dino(img_tensor)
    p = torch.sigmoid(logits).cpu().numpy().squeeze()
    if p.ndim == 3:
        p = p[0]
    if p.shape[0] != orig_h or p.shape[1] != orig_w:
        p = np.array(
            Image.fromarray((np.clip(p, 0, 1) * 255).astype(np.uint8)).resize((orig_w, orig_h), Image.BILINEAR),
            dtype=np.float32,
        ) / 255.0
    return p


def get_p_clip(processor, model_clip, image_pil, device, orig_h, orig_w):
    """
    取得 CLIPSeg sky 的機率圖（sigmoid(logits)），resize 至 (orig_h, orig_w)。
    """
    inputs = processor(
        text=['sky'],
        images=image_pil,
        return_tensors='pt',
        padding=True,
    )
    inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}
    with torch.no_grad():
        out = model_clip(**inputs)
    logits = out.logits.cpu().float().numpy().squeeze()
    if logits.ndim != 2:
        logits = logits[0]
    p = 1.0 / (1.0 + np.exp(-np.asarray(logits, dtype=np.float64)))
    p = np.asarray(np.clip(p, 0.0, 1.0), dtype=np.float32)
    if p.shape[0] != orig_h or p.shape[1] != orig_w:
        p = np.array(
            Image.fromarray((np.clip(p, 0, 1) * 255).astype(np.uint8)).resize((orig_w, orig_h), Image.BILINEAR),
            dtype=np.float32,
        ) / 255.0
    return p


def spatial_weighted_fusion(p_dino, p_clip, y_split=0.4, w_upper_dino=0.5, w_lower_dino=0.9):
    """
    空間加權融合：依垂直位置 y 使用不同權重。
    y 正規化為 [0,1]，0=影像頂、1=影像底。
    - y < y_split（上半部）：w_dino=w_upper_dino, w_clip=1-w_upper_dino（預設 0.5/0.5）
    - y >= y_split（下半部）：w_dino=w_lower_dino, w_clip=1-w_lower_dino（預設 0.9/0.1）

    p_dino, p_clip: (H, W) float32 機率圖
    回傳: (H, W) float32 融合機率圖
    """
    H, W = p_dino.shape
    # 每個 row 的 y 正規化：row 0 → 0, row H-1 → 1
    y_norm = np.linspace(0.0, 1.0, H, dtype=np.float32)
    upper = y_norm < y_split
    w_dino = np.where(upper, w_upper_dino, w_lower_dino).reshape(-1, 1)
    w_clip = np.where(upper, 1.0 - w_upper_dino, 1.0 - w_lower_dino).reshape(-1, 1)
    p_final = w_dino * p_dino + w_clip * p_clip
    return np.clip(p_final.astype(np.float32), 0.0, 1.0)


def main():
    parser = argparse.ArgumentParser(description='Ensemble: DINOv2Segmentation + CLIPSeg')
    parser.add_argument('--test_list', type=str, default=os.path.join(_root, 'outputs', 'test_list_small.txt'))
    parser.add_argument('--data_dir', type=str, default=os.path.join(_root, 'data'))
    parser.add_argument('--output_dir', type=str, default=os.path.join(_root, 'outputs', 'ensemble_dino_clipseg_masks'),
                        help='融合後的二值化 Mask 輸出目錄')
    parser.add_argument('--dino_checkpoint', type=str,
                        default=os.path.join(_root, 'outputs', 'train_dino_segmentation_20260214_101339', 'checkpoints', 'best.pth'))
    parser.add_argument('--clipseg_model', type=str, default='CIDAS/clipseg-rd64-refined')
    parser.add_argument('--image_size', type=int, nargs=2, default=[160, 160], help='DINO 輸入尺寸（會對齊 14 倍數）')
    parser.add_argument('--fusion', type=str, default='spatial', choices=['spatial', 'uniform'],
                        help='spatial=上半 0.5/0.5、下半 0.9/0.1；uniform=整張固定 w_dino/w_clip')
    parser.add_argument('--w_dino', type=float, default=0.7, help='uniform 時 DINO 權重')
    parser.add_argument('--w_clip', type=float, default=0.3, help='uniform 時 CLIPSeg 權重')
    parser.add_argument('--y_split', type=float, default=0.4, help='spatial 時上半/下半分界（y 正規化）')
    parser.add_argument('--threshold', type=float, default=0.5, help='二值化門檻')
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()

    if not CLIPSEG_AVAILABLE:
        print('請安裝: pip install transformers timm')
        sys.exit(1)

    device = torch.device(args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'))
    image_size = tuple(args.image_size)
    os.makedirs(args.output_dir, exist_ok=True)

    # 載入 DINOv2Segmentation
    print('載入 DINOv2Segmentation...')
    model_dino = create_dinov2_segmentation_model().to(device)
    ck = torch.load(args.dino_checkpoint, map_location=device)
    model_dino.load_state_dict(ck['model_state_dict'])
    model_dino.eval()

    # 載入 CLIPSeg
    print('載入 CLIPSeg...')
    processor_clip = CLIPSegProcessor.from_pretrained(args.clipseg_model)
    model_clip = CLIPSegForImageSegmentation.from_pretrained(args.clipseg_model).to(device)
    model_clip.eval()

    if args.fusion == 'spatial':
        print(f'融合: 空間加權 — 上半 (y<{args.y_split}) 0.5*DINO+0.5*CLIP，下半 0.9*DINO+0.1*CLIP')
    else:
        print(f'融合: 均勻 — DINO={args.w_dino}, CLIP={args.w_clip}')
    print(f'二值化門檻: {args.threshold}')
    print(f'輸出目錄: {args.output_dir}')
    overlay_dir = os.path.join(args.output_dir, 'overlays')
    os.makedirs(overlay_dir, exist_ok=True)
    print(f'Overlay 目錄: {overlay_dir}')
    print()

    test_list = load_list_file(args.test_list, base_data_dir=args.data_dir)
    print(f'待推論: {len(test_list)} 張')
    print()

    for item in test_list:
        cid = item['camera_id']
        camera_folder = f'skyfinder_{cid}'
        img_path = os.path.join(args.data_dir, camera_folder, 'images', item['image'])
        mask_path = os.path.join(args.data_dir, camera_folder, 'masks', item['mask'])
        if not os.path.exists(img_path):
            print(f'  跳過（缺圖）: {img_path}')
            continue

        image_pil = Image.open(img_path).convert('RGB')
        orig_h, orig_w = np.array(image_pil).shape[:2]

        # DINO：resize 到 image_size（160x160 已是 14 倍數），tensor 0~1
        img_np = np.array(image_pil).astype(np.float32) / 255.0
        img_t = torch.from_numpy(img_np.transpose(2, 0, 1)).unsqueeze(0).to(device)
        if img_t.shape[2:] != image_size:
            img_t = torch.nn.functional.interpolate(img_t, size=image_size, mode='bilinear', align_corners=False)

        p_dino = get_p_dino(model_dino, img_t, device, orig_h, orig_w)
        p_clip = get_p_clip(processor_clip, model_clip, image_pil, device, orig_h, orig_w)

        # 加權融合：空間加權或均勻
        if args.fusion == 'spatial':
            p_final = spatial_weighted_fusion(
                p_dino, p_clip,
                y_split=args.y_split,
                w_upper_dino=0.5,
                w_lower_dino=0.9,
            )
        else:
            p_final = (args.w_dino * p_dino + args.w_clip * p_clip).astype(np.float32)
            p_final = np.clip(p_final, 0.0, 1.0)

        # 固定門檻二值化
        mask_binary = (p_final >= args.threshold).astype(np.uint8)  # 0 or 1
        mask_uint8 = (mask_binary * 255).astype(np.uint8)

        # 保存二值化 Mask
        name_base = os.path.splitext(item['image'])[0]
        out_name = f"camera_{cid}_{name_base}_ensemble_mask.png"
        out_path = os.path.join(args.output_dir, out_name)
        Image.fromarray(mask_uint8).save(out_path)
        print(f'  已保存: {out_name}')

        # Overlay：一律產出。有 GT 則 TP=綠、FP=紅、FN=藍；無 GT 則僅預測疊綠
        overlay_np = np.array(image_pil, dtype=np.float32)
        pred_b = mask_binary.astype(bool)
        if os.path.exists(mask_path):
            gt_img = Image.open(mask_path).convert('L')
            gt_np = np.array(gt_img)
            gt_b = (gt_np > 127) if gt_np.max() > 1 else (gt_np.astype(np.float32) > 0.5)
            if gt_b.shape[:2] != (orig_h, orig_w):
                gt_b = np.array(Image.fromarray((gt_b.astype(np.uint8) * 255)).resize((orig_w, orig_h), Image.NEAREST), dtype=bool)
            fn_mask = gt_b & (~pred_b)
            fp_mask = pred_b & (~gt_b)
            tp_mask = gt_b & pred_b
            blue = np.array([0, 100, 255], dtype=np.float32)
            red = np.array([255, 50, 50], dtype=np.float32)
            green = np.array([50, 255, 50], dtype=np.float32)
            fn_3d = np.stack([fn_mask] * 3, axis=-1)
            fp_3d = np.stack([fp_mask] * 3, axis=-1)
            tp_3d = np.stack([tp_mask] * 3, axis=-1)
            overlay_np = np.where(tp_3d, overlay_np * (1 - 0.3 * 0.5) + green * (0.3 * 0.5), overlay_np)
            overlay_np = np.where(fn_3d, overlay_np * 0.5 + blue * 0.5, overlay_np)
            overlay_np = np.where(fp_3d, overlay_np * (1 - 0.9 * 0.5) + red * (0.9 * 0.5), overlay_np)
        else:
            pred_3d = np.stack([pred_b] * 3, axis=-1)
            green = np.array([50, 255, 50], dtype=np.float32)
            overlay_np = np.where(pred_3d, overlay_np * 0.7 + green * 0.3, overlay_np)
        overlay_np = np.clip(overlay_np, 0, 255).astype(np.uint8)
        overlay_path = os.path.join(overlay_dir, f"camera_{cid}_{name_base}_overlay.png")
        Image.fromarray(overlay_np).save(overlay_path)

    print()
    print('完成。融合 Mask 目錄:', args.output_dir)


if __name__ == '__main__':
    main()
