"""
對 test 10870 做 threshold sweep，找出最佳 threshold
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
from train_multi_camera import create_overlay


def evaluate_with_threshold(model, dataloader, device, threshold, use_amp=False, T=0.20):
    """用指定 threshold 評估，返回 overall/night/day 的 metrics"""
    model.eval()
    smooth = 1e-6
    records = []
    
    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc=f'  T={threshold:.2f}', leave=False):
            images = images.to(device)
            masks = masks.to(device)
            if use_amp:
                with torch.amp.autocast('cuda'):
                    outputs = model(images)
            else:
                outputs = model(images)
            pred_probs = torch.sigmoid(outputs)
            pred_b = (pred_probs > threshold).float()
            
            for i in range(images.shape[0]):
                luma = mean_luma_linear(images[i])
                p, g = pred_b[i], (masks[i] > 0.5).float()
                if p.dim() == 3:
                    p = p.squeeze(0)
                if g.dim() == 3:
                    g = g.squeeze(0)
                tp = ((p == 1) & (g == 1)).sum().item()
                fp = ((p == 1) & (g == 0)).sum().item()
                tn = ((p == 0) & (g == 0)).sum().item()
                fn = ((p == 0) & (g == 1)).sum().item()
                records.append({
                    'luma': luma,
                    'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
                    'image': images[i].cpu(),
                    'gt_mask': masks[i].cpu(),
                    'pred_probs': pred_probs[i].cpu(),
                })
            del images, masks, outputs, pred_probs, pred_b
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # 分組
    night = [r for r in records if r['luma'] < T]
    day = [r for r in records if r['luma'] >= T]
    
    def _pool_metrics(items):
        if not items:
            return {'iou': 0.0, 'fp_rate': 0.0, 'fn_rate': 0.0}
        tp = sum(x['tp'] for x in items)
        fp = sum(x['fp'] for x in items)
        tn = sum(x['tn'] for x in items)
        fn = sum(x['fn'] for x in items)
        iou = tp / (tp + fp + fn + smooth)
        fp_rate = fp / (tn + fp + smooth)
        fn_rate = fn / (tp + fn + smooth)
        return {'iou': iou, 'fp_rate': fp_rate, 'fn_rate': fn_rate}
    
    overall = _pool_metrics(records)
    night_metrics = _pool_metrics(night)
    day_metrics = _pool_metrics(day)
    
    return {
        'overall': overall,
        'night': night_metrics,
        'day': day_metrics,
        'records': records,  # 保留原始記錄用於 overlay
    }


def main():
    parser = argparse.ArgumentParser(description='Threshold sweep on test 10870')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--splits', type=str, default='outputs/multi_camera_splits.json')
    parser.add_argument('--image_size', type=int, nargs=2, default=[160, 160])
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--T', type=float, default=0.20, help='night/day 閾值')
    parser.add_argument('--thresholds', type=str, default='0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95')
    args = parser.parse_args()
    
    image_size = tuple(args.image_size)
    T = args.T
    thresholds = [float(t) for t in args.thresholds.split(',')]
    
    print("=" * 60)
    print("  Threshold Sweep on Test 10870")
    print("=" * 60)
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Thresholds: {thresholds}")
    print(f"  T (night/day): {T}")
    print()
    
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    use_amp = use_cuda
    print(f"  裝置: {'GPU' if use_cuda else 'CPU'}, AMP: {'on' if use_amp else 'off'}")
    print()
    
    # 載入資料
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
    
    # 載入模型
    model = create_unet_model(n_channels=3, n_classes=1)
    ck = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ck['model_state_dict'])
    model = model.to(device)
    model.eval()
    print(f"  已載入 Epoch: {ck.get('epoch', 'N/A')}, Val IoU: {ck.get('val_iou', 'N/A'):.4f}" if isinstance(ck.get('val_iou'), (int, float)) else f"  已載入 Epoch: {ck.get('epoch', 'N/A')}")
    print()
    
    # Sweep
    results = []
    best_overall_t = None
    best_overall_iou = -1
    best_night_t = None
    best_night_iou = -1
    best_records = None  # 保存最佳 threshold 的記錄用於 overlay
    
    print("執行 threshold sweep...")
    for t in thresholds:
        metrics = evaluate_with_threshold(model, test_loader, device, t, use_amp, T)
        results.append({
            'threshold': t,
            'overall_iou': metrics['overall']['iou'],
            'overall_fp': metrics['overall']['fp_rate'],
            'overall_fn': metrics['overall']['fn_rate'],
            'night_iou': metrics['night']['iou'],
            'night_fp': metrics['night']['fp_rate'],
            'night_fn': metrics['night']['fn_rate'],
            'day_iou': metrics['day']['iou'],
            'day_fp': metrics['day']['fp_rate'],
            'day_fn': metrics['day']['fn_rate'],
        })
        
        if metrics['overall']['iou'] > best_overall_iou:
            best_overall_iou = metrics['overall']['iou']
            best_overall_t = t
            if t == 0.5:  # 保存 t=0.5 的記錄
                best_records = metrics['records']
        
        if metrics['night']['iou'] > best_night_iou:
            best_night_iou = metrics['night']['iou']
            best_night_t = t
    
    # 重新載入資料以獲取 t=0.5 和 best_night_t 的完整記錄（用於 overlay）
    print("  重新載入資料以生成 overlay...")
    metrics_t05 = evaluate_with_threshold(model, test_loader, device, 0.5, use_amp, T)
    best_records = metrics_t05['records']
    best_night_metrics = evaluate_with_threshold(model, test_loader, device, best_night_t, use_amp, T)
    best_night_records = best_night_metrics['records']
    
    # 輸出 CSV
    csv_path = 'outputs/threshold_sweep_10870.csv'
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['threshold', 'overall_iou', 'overall_fp', 'overall_fn',
                                               'night_iou', 'night_fp', 'night_fn',
                                               'day_iou', 'day_fp', 'day_fn'])
        writer.writeheader()
        writer.writerows(results)
    print(f"  已寫入 {csv_path}")
    print()
    
    # 找出最佳 threshold 的 metrics
    best_overall_row = next(r for r in results if r['threshold'] == best_overall_t)
    best_night_row = next(r for r in results if r['threshold'] == best_night_t)
    
    print("=" * 60)
    print("  最佳 Threshold 建議")
    print("=" * 60)
    print(f"  Overall IoU 最佳: t={best_overall_t:.2f}")
    print(f"    IoU: {best_overall_row['overall_iou']:.4f}")
    print(f"    FP:  {best_overall_row['overall_fp']:.4f}")
    print(f"    FN:  {best_overall_row['overall_fn']:.4f}")
    print()
    print(f"  Night IoU 最佳: t={best_night_t:.2f}")
    print(f"    IoU: {best_night_row['night_iou']:.4f}")
    print(f"    FP:  {best_night_row['night_fp']:.4f}")
    print(f"    FN:  {best_night_row['night_fn']:.4f}")
    print()
    
    # 輸出對照 overlay（5 張 night 最差樣本）
    if best_records and best_night_records:
        # 找出 night 最差的 5 張（用 t=0.5 的結果）
        night_records_t05 = [r for r in best_records if r['luma'] < T]
        night_records_t05.sort(key=lambda x: (
            x['fn'] / (x['tp'] + x['fn'] + 1e-6)  # FN rate
        ), reverse=True)
        worst_5 = night_records_t05[:5]
        
        # 找出對應的 best_night_t 記錄
        ds = getattr(test_loader, 'dataset', None)
        paths = getattr(ds, 'image_paths', None) if ds else None
        
        output_dir = Path('outputs/threshold_debug')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"  輸出 5 張 night 最差樣本對照 overlay 到 {output_dir}")
        for idx, record_t05 in enumerate(worst_5):
            # 找到對應的 best_night_t 記錄（用 luma 匹配）
            record_best = next((r for r in best_night_records if abs(r['luma'] - record_t05['luma']) < 1e-6), None)
            if record_best is None:
                continue
            
            # t=0.5 overlay
            pred_t05 = (record_t05['pred_probs'] > 0.5).float()
            if pred_t05.dim() == 2:
                pred_t05 = pred_t05.unsqueeze(0)
            overlay_t05 = create_overlay(record_t05['image'], record_t05['gt_mask'], pred_t05)
            
            # best_t overlay
            pred_best = (record_best['pred_probs'] > best_night_t).float()
            if pred_best.dim() == 2:
                pred_best = pred_best.unsqueeze(0)
            overlay_best = create_overlay(record_best['image'], record_best['gt_mask'], pred_best)
            
            # 保存
            from PIL import Image
            luma_str = f"{record_t05['luma']:.4f}".replace('.', '_')
            # create_overlay 返回的是 [H, W, 3] numpy array，值範圍 [0, 1]
            if overlay_t05.max() <= 1.0:
                overlay_t05 = (overlay_t05 * 255).astype(np.uint8)
            if overlay_best.max() <= 1.0:
                overlay_best = (overlay_best * 255).astype(np.uint8)
            Image.fromarray(overlay_t05).save(
                output_dir / f"worst_{idx+1}_t0.5_luma{luma_str}.png"
            )
            Image.fromarray(overlay_best).save(
                output_dir / f"worst_{idx+1}_t{best_night_t:.2f}_luma{luma_str}.png"
            )
        
        print(f"  已保存到 {output_dir}")
    
    print()
    print("=" * 60)
    print("  完成")
    print("=" * 60)


if __name__ == '__main__':
    main()
