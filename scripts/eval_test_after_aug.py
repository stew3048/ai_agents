"""
Training 使用低光/低對比 augmentation 後，用 best checkpoint 在 test 10870 做分組評估：
- night / day（T=0.20 固定）、dusk（p25~p50）
- overlay: night_after_aug（worst IoU top-10 + worst FN top-10）、dusk_after_aug（worst IoU top-10）
- 終端印出「改前 vs 改後」對照表

用法:
  python scripts/eval_test_after_aug.py
  python scripts/eval_test_after_aug.py --checkpoint outputs/train_multi_camera_XXX/checkpoints/best.pth
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

# 改前 baseline（2026-01-25 無 aug：T=0.20 night/day）
BEFORE = {
    "overall_iou": 0.6960,
    "night_iou": 0.6299,
    "night_fn": 0.2901,
    "day_iou": 0.8135,
    "dusk_iou": None,
    "dusk_fn": None,
}

# 上一輪 after-aug（2026-01-25 強 aug：brightness/contrast -0.45~0.20, gamma 60~140, blur 3~5, noise, p=0.75/0.55/0.25/0.20）
AFTER_AUG_ROUND1 = {
    "overall_iou": 0.5979,
    "night_iou": 0.5668,
    "night_fn": 0.0289,
    "day_iou": 0.6602,
    "dusk_iou": 0.5709,
    "dusk_fn": 0.0313,
}


def find_latest_best_checkpoint():
    outputs_dir = Path('outputs')
    candidates = list(outputs_dir.glob('train_multi_camera_*/checkpoints/best.pth'))
    if not candidates:
        return None
    return str(max(candidates, key=os.path.getmtime))


def _mean_std(arr):
    a = np.array(arr, dtype=float)
    if len(a) == 0:
        return np.nan, np.nan
    return float(np.mean(a)), float(np.std(a, ddof=1)) if len(a) >= 2 else 0.0


def _pool(items, smooth=1e-6):
    tp = sum(x['tp'] for x in items)
    fp = sum(x['fp'] for x in items)
    tn = sum(x['tn'] for x in items)
    fn = sum(x['fn'] for x in items)
    iou = tp / (tp + fp + fn + smooth)
    fn_rate = fn / (tp + fn + smooth)
    fp_rate = fp / (tn + fp + smooth)
    return iou, fp_rate, fn_rate


def main():
    parser = argparse.ArgumentParser(description='Test 10870 分組評估（aug 後）+ dusk + 改前/改後對照')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--splits', type=str, default='outputs/multi_camera_splits.json')
    parser.add_argument('--image_size', type=int, nargs=2, default=[160, 160])
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--T', type=float, default=0.20, help='night/day 閾值，固定 0.20')
    args = parser.parse_args()

    image_size = tuple(args.image_size)
    T = args.T

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
    print("  Test 10870 分組評估（night / day / dusk）— after aug")
    print("=" * 60)
    print()
    print(f"  Checkpoint:   {ckpt_path}")
    print(f"  Splits:       {args.splits}  (split=test, camera=10870)")
    print(f"  T (night/day): {T}")
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
                p, g = pred_b[i], (masks[i] > 0.5).float()
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
                img = images[i].cpu()
                gt = masks[i].cpu()
                if gt.dim() == 2:
                    gt = gt.unsqueeze(0)
                pred_m = (pred_probs[i] > 0.5).float().cpu()
                if pred_m.dim() == 2:
                    pred_m = pred_m.unsqueeze(0)
                per_image.append({
                    'filename': filename, 'luma': mean_luma, 'iou': m['iou'], 'dice': dice, 'pixel_acc': pac,
                    'fp_rate': fp_rate, 'fn_rate': fn_rate, 'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
                    'image': img, 'gt_mask': gt, 'pred_mask': pred_m,
                })
            del images, masks, outputs, pred_probs, pred_b
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if not per_image:
        print("[錯誤] Test 無資料")
        sys.exit(1)

    # night / day：T=0.20 固定
    for p in per_image:
        p['group'] = 'night' if p['luma'] < T else 'day'
    night_list = [p for p in per_image if p['group'] == 'night']
    day_list = [p for p in per_image if p['group'] == 'day']

    # dusk：p25 <= luma < p50
    mean_lumas = [p['luma'] for p in per_image]
    p25 = float(np.percentile(mean_lumas, 25))
    p50 = float(np.percentile(mean_lumas, 50))
    dusk_list = [p for p in per_image if p25 <= p['luma'] < p50]

    # overall（pooled）
    overall_iou, _, _ = _pool(per_image)

    # night / day：mean±std
    print(f"[luma] T={T:.2f}, night={len(night_list)}, day={len(day_list)}; dusk (p25~p50): n={len(dusk_list)}, p25={p25:.4f}, p50={p50:.4f}")
    print()

    for label, items in [('night', night_list), ('day', day_list)]:
        if not items:
            print(f"  [{label}] n=0 略過")
            continue
        iou_m, iou_s = _mean_std([x['iou'] for x in items])
        dice_m, dice_s = _mean_std([x['dice'] for x in items])
        pac_m, pac_s = _mean_std([x['pixel_acc'] for x in items])
        fp_m, fp_s = _mean_std([x['fp_rate'] for x in items])
        fn_m, fn_s = _mean_std([x['fn_rate'] for x in items])
        print(f"  [{label}] n={len(items)}")
        print(f"      IoU:      %7.4f ± %7.4f" % (iou_m, iou_s))
        print(f"      Dice:     %7.4f ± %7.4f" % (dice_m, dice_s))
        print(f"      PixelAcc: %7.4f ± %7.4f" % (pac_m, pac_s))
        print(f"      FP rate:  %7.4f ± %7.4f" % (fp_m, fp_s))
        print(f"      FN rate:  %7.4f ± %7.4f" % (fn_m, fn_s))
        print()

    # night / day / dusk：pooled IoU / FP rate / FN rate（用於簡短報表）
    night_iou_pooled, night_fp_pooled, night_fn_pooled = _pool(night_list) if night_list else (np.nan, np.nan, np.nan)
    day_iou_pooled, day_fp_pooled, day_fn_pooled = _pool(day_list) if day_list else (np.nan, np.nan, np.nan)
    if dusk_list:
        dusk_iou, dusk_fp, dusk_fn = _pool(dusk_list)
        print(f"  [dusk] n={len(dusk_list)} (p25~p50)")
        print(f"      IoU:      %7.4f (pooled)" % dusk_iou)
        print(f"      FP rate:  %7.4f (pooled)" % dusk_fp)
        print(f"      FN rate:  %7.4f (pooled)" % dusk_fn)
        print()
    else:
        dusk_iou = dusk_fp = dusk_fn = np.nan

    # 簡短報表：day/night/dusk 的 FP/FN（pooled）
    print("=" * 80)
    print("  簡短報表：Day / Night / Dusk 的 FP/FN（pooled）")
    print("=" * 80)
    print()
    
    # 計算絕對 FP/FN 數量（pooled）
    def _get_pooled_counts(items):
        if not items:
            return 0, 0, 0, 0
        tp = sum(x['tp'] for x in items)
        fp = sum(x['fp'] for x in items)
        tn = sum(x['tn'] for x in items)
        fn = sum(x['fn'] for x in items)
        return tp, fp, tn, fn
    
    night_tp, night_fp_abs, night_tn, night_fn_abs = _get_pooled_counts(night_list)
    day_tp, day_fp_abs, day_tn, day_fn_abs = _get_pooled_counts(day_list)
    dusk_tp, dusk_fp_abs, dusk_tn, dusk_fn_abs = _get_pooled_counts(dusk_list)
    
    print(f"  [day] n={len(day_list)}")
    print(f"      FP (pooled): {day_fp_abs:,} (rate: {day_fp_pooled:.4f})")
    print(f"      FN (pooled): {day_fn_abs:,} (rate: {day_fn_pooled:.4f})")
    print(f"      IoU (pooled): {day_iou_pooled:.4f}")
    print()
    
    print(f"  [night] n={len(night_list)}")
    print(f"      FP (pooled): {night_fp_abs:,} (rate: {night_fp_pooled:.4f})")
    print(f"      FN (pooled): {night_fn_abs:,} (rate: {night_fn_pooled:.4f})")
    print(f"      IoU (pooled): {night_iou_pooled:.4f}")
    print()
    
    if dusk_list:
        print(f"  [dusk] n={len(dusk_list)}")
        print(f"      FP (pooled): {dusk_fp_abs:,} (rate: {dusk_fp:.4f})")
        print(f"      FN (pooled): {dusk_fn_abs:,} (rate: {dusk_fn:.4f})")
        print(f"      IoU (pooled): {dusk_iou:.4f}")
        print()
    
    # 分析 day IoU 下降的原因（與改前 baseline 比較）
    print("=" * 80)
    print("  Day IoU 下降原因分析（本輪 vs 改前 baseline）")
    print("=" * 80)
    print()
    
    before_day_iou = BEFORE.get("day_iou")
    if before_day_iou is not None and not np.isnan(day_iou_pooled):
        day_iou_drop = before_day_iou - day_iou_pooled
        print(f"  Day IoU 變化：{before_day_iou:.4f} → {day_iou_pooled:.4f} (下降 {day_iou_drop:.4f})")
        print()
        
        # 分析：FP 增加？FN 增加？還是邊界偏移？
        # 由於沒有改前的 FP/FN 數據，我們只能分析本輪的情況
        # 但可以從 IoU 公式推斷：IoU = TP / (TP + FP + FN)
        # 如果 IoU 下降，可能是：
        # 1. FP 增加（分母增加）
        # 2. FN 增加（分母增加）
        # 3. TP 減少（分子減少，可能伴隨 FN 增加）
        # 4. 邊界偏移（IoU 掉但 FP/FN 沒很誇張）
        
        print("  分析：")
        print(f"    - Day FP rate: {day_fp_pooled:.4f} (絕對值: {day_fp_abs:,})")
        print(f"    - Day FN rate: {day_fn_pooled:.4f} (絕對值: {day_fn_abs:,})")
        print()
        
        # 判斷主要問題
        if day_fp_pooled > 0.15:  # FP rate 較高
            print("    → 主要問題：FP 增加（將非天空誤認為天空）")
        elif day_fn_pooled > 0.10:  # FN rate 較高
            print("    → 主要問題：FN 增加（將天空誤認為非天空）")
        elif day_fp_pooled < 0.10 and day_fn_pooled < 0.10:
            print("    → 可能原因：邊界偏移（IoU 下降但 FP/FN 不誇張）")
            print("      可能是預測邊界與真實邊界有輕微偏移，導致 IoU 下降")
        else:
            print("    → 混合問題：FP 和 FN 都有一定程度的增加")
        print()
    else:
        print("  無法分析：缺少改前 baseline 的 day IoU 數據")
        print()
    
    print("=" * 80)
    print()

    # 收集改後數值（用於對照表）
    night_iou_m = _mean_std([x['iou'] for x in night_list])[0] if night_list else np.nan
    night_fn_m = _mean_std([x['fn_rate'] for x in night_list])[0] if night_list else np.nan
    day_iou_m = _mean_std([x['iou'] for x in day_list])[0] if day_list else np.nan

    # CSV（與原版相容：filename, luma, group, iou, fp, fn；group 僅 night/day）
    out_csv = os.path.join('outputs', 'test_10870_after_aug_metrics.csv')
    os.makedirs('outputs', exist_ok=True)
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['filename', 'luma', 'group', 'iou', 'fp', 'fn'])
        for p in per_image:
            w.writerow([p['filename'], f"{p['luma']:.6f}", p['group'], f"{p['iou']:.6f}", int(p['fp']), int(p['fn'])])
    print(f"  已寫入 {out_csv}")
    print()

    # Overlay: night_after_aug（worst IoU top-10 + worst FN top-10）
    err = os.path.join('outputs', 'test_10870_errors')
    night_dir = os.path.join(err, 'night_after_aug')
    os.makedirs(night_dir, exist_ok=True)
    if night_list:
        by_iou = sorted(night_list, key=lambda x: x['iou'])[:10]
        for i, r in enumerate(by_iou):
            ov = create_overlay(r['image'], r['gt_mask'], r['pred_mask'])
            suf = _safe_overlay_suffix(r['filename'])
            n = f"worst_iou_{i+1}_iou_{r['iou']:.4f}_{suf}.png" if suf else f"worst_iou_{i+1}_iou_{r['iou']:.4f}.png"
            ov.save(os.path.join(night_dir, n))
        by_fn = sorted(night_list, key=lambda x: x['fn_rate'], reverse=True)[:10]
        for i, r in enumerate(by_fn):
            ov = create_overlay(r['image'], r['gt_mask'], r['pred_mask'])
            suf = _safe_overlay_suffix(r['filename'])
            n = f"worst_fn_{i+1}_fnr_{r['fn_rate']:.4f}_{suf}.png" if suf else f"worst_fn_{i+1}_fnr_{r['fn_rate']:.4f}.png"
            ov.save(os.path.join(night_dir, n))
        print(f"  已存 night_after_aug: worst IoU 10 + worst FN 10 → {night_dir}")

    # Overlay: dusk_after_aug（worst IoU top-10）
    dusk_dir = os.path.join(err, 'dusk_after_aug')
    os.makedirs(dusk_dir, exist_ok=True)
    if dusk_list:
        by_iou = sorted(dusk_list, key=lambda x: x['iou'])[:10]
        for i, r in enumerate(by_iou):
            ov = create_overlay(r['image'], r['gt_mask'], r['pred_mask'])
            suf = _safe_overlay_suffix(r['filename'])
            n = f"worst_{i+1}_iou_{r['iou']:.4f}_{suf}.png" if suf else f"worst_{i+1}_iou_{r['iou']:.4f}.png"
            ov.save(os.path.join(dusk_dir, n))
        print(f"  已存 dusk_after_aug: worst IoU 10 → {dusk_dir}")
    print()

    # 改前 vs 上一輪 after-aug vs 本輪 對照表
    after = {
        "overall_iou": float(overall_iou),
        "night_iou": float(night_iou_m) if not np.isnan(night_iou_m) else None,
        "night_fn": float(night_fn_m) if not np.isnan(night_fn_m) else None,
        "day_iou": float(day_iou_m) if not np.isnan(day_iou_m) else None,
        "dusk_iou": float(dusk_iou) if dusk_list and not np.isnan(dusk_iou) else None,
        "dusk_fn": float(dusk_fn) if dusk_list and not np.isnan(dusk_fn) else None,
    }

    def _v(x):
        return f"{x:.4f}" if x is not None else "—"

    print("=" * 80)
    print("  改前 vs 上一輪 after-aug vs 本輪 對照表")
    print("=" * 80)
    print()
    print("  %-18s %12s %12s %12s" % ("", "改前", "上一輪 after-aug", "本輪"))
    print("  %-18s %12s %12s %12s" % ("overall test IoU", _v(BEFORE["overall_iou"]), _v(AFTER_AUG_ROUND1["overall_iou"]), _v(after["overall_iou"])))
    print("  %-18s %12s %12s %12s" % ("night test IoU", _v(BEFORE["night_iou"]), _v(AFTER_AUG_ROUND1["night_iou"]), _v(after["night_iou"])))
    print("  %-18s %12s %12s %12s" % ("night FN", _v(BEFORE["night_fn"]), _v(AFTER_AUG_ROUND1["night_fn"]), _v(after["night_fn"])))
    print("  %-18s %12s %12s %12s" % ("day test IoU", _v(BEFORE["day_iou"]), _v(AFTER_AUG_ROUND1["day_iou"]), _v(after["day_iou"])))
    print("  %-18s %12s %12s %12s" % ("dusk IoU", _v(BEFORE["dusk_iou"]), _v(AFTER_AUG_ROUND1["dusk_iou"]), _v(after["dusk_iou"])))
    print("  %-18s %12s %12s %12s" % ("dusk FN", _v(BEFORE["dusk_fn"]), _v(AFTER_AUG_ROUND1["dusk_fn"]), _v(after["dusk_fn"])))
    print()
    print("  成功判準（本輪 vs 改前）：")
    print("    - night FN 明顯低於改前（目標 ≤ 0.18）")
    print("    - day IoU 不再大掉（下降 ≤ 0.03）")
    print("    - overall IoU 回升（至少 ≥ 0.66，往 0.70 靠）")
    print("=" * 80)
    print("  完成")
    print("=" * 80)


if __name__ == '__main__':
    main()
