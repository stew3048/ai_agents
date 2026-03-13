"""
Test (camera 10870) 依 mean_luma 分 night / day，輸出各組 mean±std、CSV、各組 top-10 最差 IoU overlay。

- 用 best checkpoint 推論 10870 test
- mean_luma：image 為 sRGB [0,1]，先 sRGB→linear 再 0.2126*R+0.7152*G+0.0722*B 取平均（直接用在 sRGB 會高估暗部）
- night: luma < T；day: luma >= T。T=0.10（linear 空間），若 night < 10 張則改用 p25
- 各組：IoU / Dice / PixelAcc / FP rate / FN rate 之平均與標準差
- outputs/test_10870_grouped_metrics.csv：filename, luma, group, iou, fp, fn
- 各組最差 10 張 overlay：outputs/test_10870_errors/night/、outputs/test_10870_errors/day/

用法:
  python scripts/eval_test_night_day.py
  python scripts/eval_test_night_day.py --checkpoint outputs/train_multi_camera_XXX/checkpoints/best.pth
"""

import os
import sys
import argparse
import csv
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

from models import create_unet_model
from utils.dataset import get_dataloader, load_splits_from_json
from utils.metrics import calculate_metrics
from utils.image_utils import mean_luma_linear
from train_multi_camera import create_overlay, _safe_overlay_suffix


def find_latest_best_checkpoint():
    outputs_dir = Path('outputs')
    candidates = list(outputs_dir.glob('train_multi_camera_*/checkpoints/best.pth'))
    if not candidates:
        return None
    return str(max(candidates, key=os.path.getmtime))


def main():
    parser = argparse.ArgumentParser(description='Test 10870 night/day 分組分析 + CSV + 最差 10 overlay')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--splits', type=str, default='outputs/multi_camera_splits.json')
    parser.add_argument('--image_size', type=int, nargs=2, default=[160, 160])
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--T', type=float, default=0.20, help='luma 閾值；若 night<10 則改用 p25')
    args = parser.parse_args()

    image_size = tuple(args.image_size)
    T_default = args.T

    if args.checkpoint:
        ckpt_path = args.checkpoint
        if not os.path.isfile(ckpt_path):
            print(f"[錯誤] 找不到 checkpoint: {ckpt_path}")
            sys.exit(1)
    else:
        ckpt_path = find_latest_best_checkpoint()
        if not ckpt_path:
            print("[錯誤] 找不到 best.pth，請用 --checkpoint 指定")
            sys.exit(1)

    print("=" * 60)
    print("  Test 10870 night / day 分組分析")
    print("=" * 60)
    print()
    print(f"  Checkpoint:   {ckpt_path}")
    print(f"  Splits:       {args.splits}  (split=test, camera=10870)")
    print(f"  Image size:   {image_size}")
    print()

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    use_amp = use_cuda
    print(f"  裝置: {'GPU' if use_cuda else 'CPU'}, AMP: {'on' if use_amp else 'off'}")
    print()

    test_list = load_splits_from_json(args.splits, 'test')
    print(f"  Test: {len(test_list)} 張")
    test_loader = get_dataloader(
        split_list=test_list,
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
    vi = ck.get('val_iou')
    print(f"  已載入 Epoch: {ck.get('epoch','N/A')}, Val IoU: {vi:.4f}" if isinstance(vi, (int, float)) else f"  已載入 Epoch: {ck.get('epoch','N/A')}")
    print()

    smooth = 1e-6
    per_image = []
    bs = getattr(test_loader, 'batch_size', 1)
    ds = getattr(test_loader, 'dataset', None)
    paths = getattr(ds, 'image_paths', None) if ds else None

    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(tqdm(test_loader, desc='  Test', leave=True)):
            images = images.to(device)
            masks = masks.to(device)

            if use_amp:
                with torch.amp.autocast('cuda'):
                    outputs = model(images)
            else:
                outputs = model(images)

            pred_probs = torch.sigmoid(outputs)
            pred_b = (pred_probs > 0.5).float()

            for i in range(images.shape[0]):
                gidx = batch_idx * bs + i
                mean_luma = mean_luma_linear(images[i])

                filename = os.path.basename(paths[gidx]) if paths and gidx < len(paths) else f"idx{gidx}"

                m = calculate_metrics(outputs[i:i+1], masks[i:i+1])
                p = pred_b[i]
                g = (masks[i] > 0.5).float()
                if g.dim() == 3:
                    g = g.squeeze(0)
                if p.dim() == 3:
                    p = p.squeeze(0)
                tp = ((p == 1) & (g == 1)).sum().item()
                fp = ((p == 1) & (g == 0)).sum().item()
                tn = ((p == 0) & (g == 0)).sum().item()
                fn = ((p == 0) & (g == 1)).sum().item()

                dice = (2 * tp) / (2 * tp + fp + fn + smooth)
                pac = (tp + tn) / (tp + tn + fp + fn + smooth)
                fp_rate = fp / (tn + fp + smooth)
                fn_rate = fn / (tp + fn + smooth)

                # 留 overlay 用：image [C,H,W], gt [1,H,W], pred [1,H,W]
                img = images[i].cpu()
                gt = masks[i].cpu()
                if gt.dim() == 2:
                    gt = gt.unsqueeze(0)
                pred_m = pred_probs[i].cpu()
                if pred_m.dim() == 2:
                    pred_m = pred_m.unsqueeze(0)
                pred_m = (pred_m > 0.5).float()

                per_image.append({
                    'filename': filename,
                    'luma': mean_luma,
                    'iou': m['iou'],
                    'dice': dice,
                    'pixel_acc': pac,
                    'fp_rate': fp_rate,
                    'fn_rate': fn_rate,
                    'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
                    'image': img, 'gt_mask': gt, 'pred_mask': pred_m,
                })

            del images, masks, outputs, pred_probs, pred_b
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if not per_image:
        print("[錯誤] Test 無資料")
        sys.exit(1)

    # T：0.20，若 night < 10 則 p25
    mean_lumas = [p['luma'] for p in per_image]
    T = T_default
    night_count = sum(1 for m in mean_lumas if m < T)
    if night_count < 10:
        T = float(np.percentile(mean_lumas, 25))
        night_count = sum(1 for m in mean_lumas if m < T)
    day_count = len(mean_lumas) - night_count

    for p in per_image:
        p['group'] = 'night' if p['luma'] < T else 'day'

    print("[luma] 閾值 T=%s, night=%d, day=%d" % (
        f"{T:.3f}" if T == T_default else f"p25={T:.3f}", night_count, day_count))
    print()

    # 各組 mean ± std
    night_list = [p for p in per_image if p['group'] == 'night']
    day_list = [p for p in per_image if p['group'] == 'day']

    def _mean_std(arr):
        a = np.array(arr, dtype=float)
        if len(a) == 0:
            return np.nan, np.nan
        m = np.mean(a)
        s = np.std(a, ddof=1) if len(a) >= 2 else 0.0
        return m, s

    for label, items in [('night', night_list), ('day', day_list)]:
        if not items:
            print(f"  {label}: 0 張，略過")
            continue
        iou_m, iou_s = _mean_std([x['iou'] for x in items])
        dice_m, dice_s = _mean_std([x['dice'] for x in items])
        pac_m, pac_s = _mean_std([x['pixel_acc'] for x in items])
        fp_m, fp_s = _mean_std([x['fp_rate'] for x in items])
        fn_m, fn_s = _mean_std([x['fn_rate'] for x in items])
        print(f"  [{label}] n=%d" % len(items))
        print(f"      IoU:      %7.4f ± %7.4f" % (iou_m, iou_s))
        print(f"      Dice:     %7.4f ± %7.4f" % (dice_m, dice_s))
        print(f"      PixelAcc: %7.4f ± %7.4f" % (pac_m, pac_s))
        print(f"      FP rate:  %7.4f ± %7.4f" % (fp_m, fp_s))
        print(f"      FN rate:  %7.4f ± %7.4f" % (fn_m, fn_s))
        print()

    # CSV: filename, luma, group, iou, fp, fn
    out_csv = os.path.join('outputs', 'test_10870_grouped_metrics.csv')
    os.makedirs('outputs', exist_ok=True)
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['filename', 'luma', 'group', 'iou', 'fp', 'fn'])
        for p in per_image:
            w.writerow([p['filename'], f"{p['luma']:.6f}", p['group'], f"{p['iou']:.6f}", int(p['fp']), int(p['fn'])])
    print(f"  已寫入 {out_csv}")
    print()

    # 各組 top-10 最差 IoU overlay → night/、day/
    err_base = os.path.join('outputs', 'test_10870_errors')
    for group, sub in [('night', 'night'), ('day', 'day')]:
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
    print("=" * 60)
    print("  完成")
    print("=" * 60)


if __name__ == '__main__':
    main()
