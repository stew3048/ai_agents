"""
一次性 Val 分析：載入 checkpoint，只跑 validation split（camera=9483），
輸出 overall / night-ish / non-night 摘要，並產出 outputs/val_9483_grouped_metrics.csv。

- mean_luma：image 為 sRGB [0,1]，先 sRGB→linear 再 0.2126*R+0.7152*G+0.0722*B 取平均（直接用在 sRGB 會高估暗部）
- night-ish: mean_luma < T；non-night: >= T。T=0.10（linear 空間），若 night-ish < 10 張則改用 p25
- 不做任何訓練

用法:
  python scripts/eval_val_only.py
  python scripts/eval_val_only.py --checkpoint outputs/train_multi_camera_XXX/checkpoints/best.pth
"""

import os
import sys
import argparse
import csv
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

# 專案根目錄
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

from models import create_unet_model
from utils.dataset import get_dataloader, load_splits_from_json
from utils.metrics import calculate_metrics
from utils.image_utils import mean_luma_linear
from train_multi_camera import create_overlay, _safe_overlay_suffix


def find_latest_best_checkpoint():
    """找最新的 train_multi_camera_* 下的 checkpoints/best.pth"""
    outputs_dir = Path('outputs')
    candidates = list(outputs_dir.glob('train_multi_camera_*/checkpoints/best.pth'))
    if not candidates:
        return None
    return str(max(candidates, key=os.path.getmtime))


def main():
    parser = argparse.ArgumentParser(description='Val 一次性分析：overall / night-ish / non-night + val_9483_grouped_metrics.csv')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='checkpoint 路徑；未指定則用最新的 best.pth')
    parser.add_argument('--splits', type=str, default='outputs/multi_camera_splits.json',
                        help='multi_camera_splits.json 路徑')
    parser.add_argument('--image_size', type=int, nargs=2, default=[160, 160],
                        help='與訓練時一致，預設 160 160')
    parser.add_argument('--batch_size', type=int, default=2, help='batch size')
    parser.add_argument('--T', type=float, default=0.10, help='luma 閾值（linear）；若 night-ish<10 則改用 p25')
    args = parser.parse_args()

    image_size = tuple(args.image_size)
    json_file = args.splits
    T_default = args.T

    if args.checkpoint:
        ckpt_path = args.checkpoint
        if not os.path.isfile(ckpt_path):
            print(f"[錯誤] 找不到 checkpoint: {ckpt_path}")
            sys.exit(1)
    else:
        ckpt_path = find_latest_best_checkpoint()
        if not ckpt_path:
            print("[錯誤] 找不到任何 outputs/train_multi_camera_*/checkpoints/best.pth")
            print("       請以 --checkpoint 指定路徑")
            sys.exit(1)

    print("=" * 60)
    print("  Val 一次性分析（eval_val_only）")
    print("=" * 60)
    print()
    print(f"  Checkpoint:   {ckpt_path}")
    print(f"  Splits:       {json_file}  (split=val, camera=9483)")
    print(f"  Image size:   {image_size}")
    print()

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    use_amp = use_cuda
    print(f"  裝置: {'GPU' if use_cuda else 'CPU'}, AMP: {'on' if use_amp else 'off'}")
    print()

    val_list = load_splits_from_json(json_file, 'val')
    print(f"  Val: {len(val_list)} 張")
    val_loader = get_dataloader(
        split_list=val_list,
        batch_size=args.batch_size,
        shuffle=False,
        transform=False,
        image_size=image_size,
        num_workers=0
    )
    print()

    model = create_unet_model(n_channels=3, n_classes=1)
    ck = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ck['model_state_dict'])
    model = model.to(device)
    model.eval()
    print(f"  已載入 Epoch: {ck.get('epoch', 'N/A')}, Val IoU: {ck.get('val_iou', 'N/A'):.4f}" if isinstance(ck.get('val_iou'), (int, float)) else f"  已載入 Epoch: {ck.get('epoch', 'N/A')}")
    print()

    # 一輪：每張 mean_luma, iou, tp, fp, tn, fn
    smooth = 1e-6
    per_image = []
    bs = getattr(val_loader, 'batch_size', 1)
    ds = getattr(val_loader, 'dataset', None)
    paths = getattr(ds, 'image_paths', None) if ds else None

    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(tqdm(val_loader, desc='  Val', leave=True)):
            images = images.to(device)
            masks = masks.to(device)

            if use_amp:
                with torch.amp.autocast('cuda'):
                    outputs = model(images)
            else:
                outputs = model(images)

            pred_probs = torch.sigmoid(outputs)
            pred_b = (pred_probs > 0.5).float()
            gt_b = (masks > 0.5).float()

            for i in range(images.shape[0]):
                gidx = batch_idx * bs + i
                mean_luma = mean_luma_linear(images[i])
                filename = os.path.basename(paths[gidx]) if paths and gidx < len(paths) else f"idx{gidx}"

                m = calculate_metrics(outputs[i:i+1], masks[i:i+1])
                p, g = pred_b[i], gt_b[i]
                tp = ((p == 1) & (g == 1)).sum().item()
                fp = ((p == 1) & (g == 0)).sum().item()
                tn = ((p == 0) & (g == 0)).sum().item()
                fn = ((p == 0) & (g == 1)).sum().item()

                # 留 overlay 用：image [C,H,W], gt [1,H,W], pred [1,H,W]
                img = images[i].cpu()
                gt = masks[i].cpu()
                if gt.dim() == 2:
                    gt = gt.unsqueeze(0)
                pred_m = (pred_probs[i] > 0.5).float().cpu()
                if pred_m.dim() == 2:
                    pred_m = pred_m.unsqueeze(0)

                per_image.append({
                    'filename': filename,
                    'luma': mean_luma,
                    'iou': m['iou'],
                    'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
                    'image': img, 'gt_mask': gt, 'pred_mask': pred_m,
                })

            del images, masks, outputs, pred_probs, pred_b, gt_b
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if not per_image:
        print("[錯誤] Val 無資料")
        sys.exit(1)

    # T：先 T_default (0.10)，若 night-ish < 10 則 p25
    mean_lumas = [p['luma'] for p in per_image]
    T = T_default
    night_count = sum(1 for m in mean_lumas if m < T)
    if night_count < 10:
        T = float(np.percentile(mean_lumas, 25))
        night_count = sum(1 for m in mean_lumas if m < T)
    non_night_count = len(mean_lumas) - night_count

    for p in per_image:
        p['group'] = 'night-ish' if p['luma'] < T else 'non-night'

    def _pool(items):
        tp = sum(x['tp'] for x in items)
        fp = sum(x['fp'] for x in items)
        tn = sum(x['tn'] for x in items)
        fn = sum(x['fn'] for x in items)
        iou = tp / (tp + fp + fn + smooth)
        dice = (2 * tp) / (2 * tp + fp + fn + smooth)
        pixel_acc = (tp + tn) / (tp + tn + fp + fn + smooth)
        fp_rate = fp / (tn + fp + smooth)
        fn_rate = fn / (tp + fn + smooth)
        return iou, dice, pixel_acc, fp_rate, fn_rate

    iou_o, dice_o, pac_o, fp_o, fn_o = _pool(per_image)
    night_list = [p for p in per_image if p['group'] == 'night-ish']
    non_list = [p for p in per_image if p['group'] == 'non-night']

    if night_list:
        iou_n, dice_n, pac_n, fp_n, fn_n = _pool(night_list)
    else:
        iou_n, dice_n, pac_n, fp_n, fn_n = 0.0, 0.0, 0.0, 0.0, 0.0
    if non_list:
        iou_nn, dice_nn, pac_nn, fp_nn, fn_nn = _pool(non_list)
    else:
        iou_nn, dice_nn, pac_nn, fp_nn, fn_nn = 0.0, 0.0, 0.0, 0.0, 0.0

    # 寫出 outputs/val_9483_grouped_metrics.csv：filename, luma, group, iou, fp, fn（fp/fn 為該張 pixel 計數）
    out_path = os.path.join('outputs', 'val_9483_grouped_metrics.csv')
    os.makedirs('outputs', exist_ok=True)
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['filename', 'luma', 'group', 'iou', 'fp', 'fn'])
        for p in per_image:
            w.writerow([p['filename'], f"{p['luma']:.6f}", p['group'], f"{p['iou']:.6f}", int(p['fp']), int(p['fn'])])
    print(f"  已寫入 {out_path}")
    print()

    # 各組 top-10 最差 IoU overlay → night-ish/、non-night/
    err_base = os.path.join('outputs', 'val_9483_errors')
    for group, sub in [('night-ish', 'night-ish'), ('non-night', 'non-night')]:
        items = [p for p in per_image if p['group'] == group]
        if not items:
            continue
        worst = sorted(items, key=lambda x: x['iou'])[:10]
        out_dir = os.path.join(err_base, sub)
        os.makedirs(out_dir, exist_ok=True)
        for i, r in enumerate(worst):
            overlay = create_overlay(r['image'], r['gt_mask'], r['pred_mask'])
            suf = _safe_overlay_suffix(r['filename'])
            name = f"worst_{i+1}_iou_{r['iou']:.4f}_{suf}.png" if suf else f"worst_{i+1}_iou_{r['iou']:.4f}.png"
            overlay.save(os.path.join(out_dir, name))
        print(f"  已存 {group} 最差 {len(worst)} 張 → {out_dir}")
    print()

    # 終端摘要：overall | night-ish | non-night
    t_src = f"{T:.3f}" if T == T_default else f"p25={T:.3f}"
    print("[luma] T=%s, night-ish=%d, non-night=%d" % (t_src, night_count, non_night_count))
    print()
    print("               IoU    Dice   PixelAcc   FP rate   FN rate")
    print("  overall   %6.4f  %6.4f  %8.4f  %8.4f  %8.4f" % (iou_o, dice_o, pac_o, fp_o, fn_o))
    print("  night-ish %6.4f  %6.4f  %8.4f  %8.4f  %8.4f" % (iou_n, dice_n, pac_n, fp_n, fn_n))
    print("  non-night %6.4f  %6.4f  %8.4f  %8.4f  %8.4f" % (iou_nn, dice_nn, pac_nn, fp_nn, fn_nn))
    print()
    print("=" * 60)
    print("  完成")
    print("=" * 60)


if __name__ == '__main__':
    main()
