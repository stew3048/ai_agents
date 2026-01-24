"""
用訓練最佳 checkpoint (best.pth) 在 Test set 上評估，並輸出最好/最差各 5 張 overlay。

用法:
  python eval_multi_camera_test.py
  python eval_multi_camera_test.py --checkpoint outputs/train_multi_camera_XXX/checkpoints/best.pth

- 若未指定 --checkpoint，會自動找最新的 outputs/train_multi_camera_*/checkpoints/best.pth
- Overlay 存到該 run 目錄下的 test_overlays/
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from PIL import Image
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

from models import create_unet_model
from utils.dataset import get_dataloader, load_splits_from_json
from utils.metrics import calculate_metrics
from train_multi_camera import create_overlay, evaluate_with_predictions


def find_latest_best_checkpoint():
    """找最新的 train_multi_camera_* 下的 checkpoints/best.pth"""
    outputs_dir = Path('outputs')
    candidates = list(outputs_dir.glob('train_multi_camera_*/checkpoints/best.pth'))
    if not candidates:
        return None
    return str(max(candidates, key=os.path.getmtime))


def main():
    parser = argparse.ArgumentParser(description='Multi-Camera U-Net: 用 best checkpoint 跑 Test 並輸出 overlay')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='best.pth 路徑；未指定則用最新的 train_multi_camera_*/checkpoints/best.pth')
    parser.add_argument('--splits', type=str, default='outputs/multi_camera_splits.json',
                        help='multi_camera_splits.json 路徑')
    parser.add_argument('--image_size', type=int, nargs=2, default=[160, 160],
                        help='與訓練時一致的 image_size，預設 160 160')
    args = parser.parse_args()

    image_size = tuple(args.image_size)
    json_file = args.splits

    # === 解析 checkpoint ===
    if args.checkpoint:
        best_path = args.checkpoint
        if not os.path.isfile(best_path):
            print(f"[錯誤] 找不到 checkpoint: {best_path}")
            sys.exit(1)
        run_dir = str(Path(best_path).resolve().parent.parent)  # checkpoints -> run dir
    else:
        best_path = find_latest_best_checkpoint()
        if not best_path:
            print("[錯誤] 找不到任何 outputs/train_multi_camera_*/checkpoints/best.pth")
            print("       請先完成訓練，或使用 --checkpoint 指定路徑")
            sys.exit(1)
        run_dir = str(Path(best_path).resolve().parent.parent)

    overlay_dir = os.path.join(run_dir, 'test_overlays')
    os.makedirs(overlay_dir, exist_ok=True)

    print("=" * 60)
    print("  Multi-Camera U-Net：Test 評估")
    print("=" * 60)
    print()
    print(f"  Checkpoint:   {best_path}")
    print(f"  Run 目錄:     {run_dir}")
    print(f"  Splits:       {json_file}")
    print(f"  Image size:   {image_size}")
    print(f"  Overlay 輸出: {overlay_dir}")
    print()

    # === 設備 ===
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    use_amp = use_cuda
    print(f"  裝置: {'GPU' if use_cuda else 'CPU'}, AMP: {'on' if use_amp else 'off'}")
    print()

    # === 載入 test split ===
    print("載入 Test split...")
    test_list = load_splits_from_json(json_file, 'test')
    print(f"  Test: {len(test_list)} 張")
    print()

    # === DataLoader ===
    test_loader = get_dataloader(
        split_list=test_list,
        batch_size=4,
        shuffle=False,
        transform=False,
        image_size=image_size,
        num_workers=0
    )
    print(f"  Test batches: {len(test_loader)}")
    print()

    # === 模型與 checkpoint ===
    print("載入模型與 best checkpoint...")
    model = create_unet_model(n_channels=3, n_classes=1)
    ck = torch.load(best_path, map_location=device)
    model.load_state_dict(ck['model_state_dict'])
    model = model.to(device)
    model.eval()
    print(f"  Epoch: {ck.get('epoch', 'N/A')}, Val IoU: {ck.get('val_iou', 'N/A'):.4f}" if isinstance(ck.get('val_iou'), (int, float)) else f"  Epoch: {ck.get('epoch', 'N/A')}")
    print()

    # === 整體 Test 指標 (evaluate) ===
    criterion = nn.BCEWithLogitsLoss()
    test_loss, test_metrics = _evaluate(model, test_loader, criterion, device, use_amp)
    print("=" * 60)
    print("  Test 結果（整體）")
    print("=" * 60)
    print()
    print(f"  Test Loss:     {test_loss:.4f}")
    print(f"  Test IoU:      {test_metrics['iou']:.4f}")
    print(f"  Test Dice:     {test_metrics['dice']:.4f}")
    print(f"  Test PixelAcc: {test_metrics['pixel_acc']:.4f}")
    print(f"  Test FP Rate:  {test_metrics['fp_rate']:.4f}  (非天空誤判為天空)")
    print(f"  Test FN Rate:  {test_metrics['fn_rate']:.4f}  (天空漏檢)")
    print()

    # === 每張圖預測與 IoU，用於 best/worst 5；並補上每張的 fp_rate, fn_rate ===
    print("計算每張圖 IoU 並生成 overlay...")
    all_results = evaluate_with_predictions(model, test_loader, device, use_amp)
    _smooth = 1e-6
    for r in all_results:
        p = (r['pred_mask'] > 0.5).float()
        g = (r['gt_mask'] > 0.5).float()
        fp = ((p == 1) & (g == 0)).sum().item()
        fn = ((p == 0) & (g == 1)).sum().item()
        tn = (g == 0).sum().item()
        tp = (g == 1).sum().item()
        r['fp_rate'] = fp / (tn + _smooth)
        r['fn_rate'] = fn / (tp + _smooth)
    all_results.sort(key=lambda x: x['iou'], reverse=True)

    best_5 = all_results[:5]
    worst_5 = all_results[-5:]

    # 儲存 best 5
    print("  儲存最好的 5 張 overlay...")
    for i, r in enumerate(best_5):
        overlay = create_overlay(r['image'], r['gt_mask'], r['pred_mask'])
        overlay.save(os.path.join(overlay_dir, f'best_{i+1}_iou_{r["iou"]:.4f}.png'))

    # 儲存 worst 5
    print("  儲存最差的 5 張 overlay...")
    for i, r in enumerate(worst_5):
        overlay = create_overlay(r['image'], r['gt_mask'], r['pred_mask'])
        overlay.save(os.path.join(overlay_dir, f'worst_{i+1}_iou_{r["iou"]:.4f}.png'))

    print(f"  Overlay 已儲存至: {overlay_dir}")
    print()

    # === 印出 best / worst 5 的 IoU ===
    print("  最好的 5 張 (IoU):")
    for i, r in enumerate(best_5, 1):
        print(f"    {i}. {r['iou']:.4f}")
    print("  最差的 5 張 (IoU, FP Rate, FN Rate):")
    for i, r in enumerate(worst_5, 1):
        print(f"    {i}. IoU={r['iou']:.4f}  FP={r['fp_rate']:.4f}  FN={r['fn_rate']:.4f}")
    print()

    print("=" * 60)
    print("  完成")
    print("=" * 60)
    print(f"  Test 整體: IoU={test_metrics['iou']:.4f}, Dice={test_metrics['dice']:.4f}, PixelAcc={test_metrics['pixel_acc']:.4f}, FP Rate={test_metrics['fp_rate']:.4f}, FN Rate={test_metrics['fn_rate']:.4f}")
    print(f"  Overlay:   {overlay_dir}")
    print()


def _evaluate(model, dataloader, criterion, device, use_amp=False):
    """與 train_multi_camera.evaluate 相同邏輯，並累計 FP/FN rate。"""
    model.eval()
    running_loss = 0.0
    total_iou = 0.0
    total_dice = 0.0
    total_pixel_acc = 0.0
    run_fp, run_fn, run_neg, run_pos = 0.0, 0.0, 0.0, 0.0
    num_batches = 0
    smooth = 1e-6

    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc='  Test', leave=True):
            images = images.to(device)
            masks = masks.to(device)
            if use_amp:
                with torch.amp.autocast('cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, masks)
            else:
                outputs = model(images)
                loss = criterion(outputs, masks)
            running_loss += loss.item()
            m = calculate_metrics(outputs, masks)
            total_iou += m['iou']
            total_dice += m['dice']
            total_pixel_acc += m['pixel_acc']

            pred_b = (torch.sigmoid(outputs) > 0.5).float()
            gt_b = (masks > 0.5).float()
            run_fp += ((pred_b == 1) & (gt_b == 0)).sum().item()
            run_fn += ((pred_b == 0) & (gt_b == 1)).sum().item()
            run_neg += (gt_b == 0).sum().item()
            run_pos += (gt_b == 1).sum().item()

            num_batches += 1
            del images, masks, outputs, loss
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    n = num_batches if num_batches > 0 else 1
    avg_loss = running_loss / n
    fp_rate = run_fp / (run_neg + smooth)
    fn_rate = run_fn / (run_pos + smooth)
    metrics = {
        'iou': total_iou / n,
        'dice': total_dice / n,
        'pixel_acc': total_pixel_acc / n,
        'fp_rate': fp_rate,
        'fn_rate': fn_rate
    }
    return avg_loss, metrics


if __name__ == '__main__':
    main()
