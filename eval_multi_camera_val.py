"""
僅對 Val 跑 evaluate_val_with_luma_groups，輸出 overall / night-ish / non-night 與 val_9483_luma_groups.csv。

用法:
  python eval_multi_camera_val.py
  python eval_multi_camera_val.py --checkpoint outputs/train_multi_camera_XXX/checkpoints/best.pth

- 若未指定 --checkpoint，會自動找最新的 outputs/train_multi_camera_*/checkpoints/best.pth
- 寫出 outputs/val_9483_luma_groups.csv（filename, mean_luma, group）
"""

import os
import sys
import argparse
import csv
import torch
import torch.nn as nn
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

from models import create_unet_model
from utils.dataset import get_dataloader, load_splits_from_json
from train_multi_camera import evaluate_val_with_luma_groups


def find_latest_best_checkpoint():
    """找最新的 train_multi_camera_* 下的 checkpoints/best.pth"""
    outputs_dir = Path('outputs')
    candidates = list(outputs_dir.glob('train_multi_camera_*/checkpoints/best.pth'))
    if not candidates:
        return None
    return str(max(candidates, key=os.path.getmtime))


def main():
    parser = argparse.ArgumentParser(description='Val 評估：evaluate_val_with_luma_groups + val_9483_luma_groups.csv')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='best.pth 路徑；未指定則用最新的 train_multi_camera_*/checkpoints/best.pth')
    parser.add_argument('--splits', type=str, default='outputs/multi_camera_splits.json',
                        help='multi_camera_splits.json 路徑')
    parser.add_argument('--image_size', type=int, nargs=2, default=[160, 160],
                        help='與訓練時一致的 image_size，預設 160 160')
    parser.add_argument('--batch_size', type=int, default=2, help='val batch size')
    args = parser.parse_args()

    image_size = tuple(args.image_size)
    json_file = args.splits

    if args.checkpoint:
        best_path = args.checkpoint
        if not os.path.isfile(best_path):
            print(f"[錯誤] 找不到 checkpoint: {best_path}")
            sys.exit(1)
    else:
        best_path = find_latest_best_checkpoint()
        if not best_path:
            print("[錯誤] 找不到任何 outputs/train_multi_camera_*/checkpoints/best.pth")
            print("       請先完成訓練，或使用 --checkpoint 指定路徑")
            sys.exit(1)

    print("=" * 60)
    print("  Val 評估（evaluate_val_with_luma_groups）")
    print("=" * 60)
    print()
    print(f"  Checkpoint:   {best_path}")
    print(f"  Splits:       {json_file}")
    print(f"  Image size:   {image_size}")
    print(f"  Batch size:   {args.batch_size}")
    print()

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    use_amp = use_cuda
    print(f"  裝置: {'GPU' if use_cuda else 'CPU'}, AMP: {'on' if use_amp else 'off'}")
    print()

    print("載入 Val split...")
    val_list = load_splits_from_json(json_file, 'val')
    print(f"  Val: {len(val_list)} 張")
    print()

    val_loader = get_dataloader(
        split_list=val_list,
        batch_size=args.batch_size,
        shuffle=False,
        transform=False,
        image_size=image_size,
        num_workers=0
    )
    print(f"  Val batches: {len(val_loader)}")
    print()

    print("載入模型與 checkpoint...")
    model = create_unet_model(n_channels=3, n_classes=1)
    ck = torch.load(best_path, map_location=device)
    model.load_state_dict(ck['model_state_dict'])
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    vi = ck.get('val_iou')
    print(f"  Epoch: {ck.get('epoch', 'N/A')}, Val IoU: {vi:.4f}" if isinstance(vi, (int, float)) else f"  Epoch: {ck.get('epoch', 'N/A')}")
    print()

    print("執行 evaluate_val_with_luma_groups...")
    val_loss, val_metrics, luma_info = evaluate_val_with_luma_groups(
        model, val_loader, criterion, device, use_amp, T_default=0.20
    )

    print()
    print("=" * 60)
    print("  Val 結果")
    print("=" * 60)
    print()
    print(f"  Val Loss:     {val_loss:.4f}")
    print(f"  Val IoU:      {val_metrics['iou']:.4f}")
    print(f"  Val Dice:     {val_metrics['dice']:.4f}")
    print(f"  Val PixelAcc: {val_metrics['pixel_acc']:.4f}")
    print(f"  Val FP Rate:  {val_metrics['fp_rate']:.4f}  (非天空誤判為天空)")
    print(f"  Val FN Rate:  {val_metrics['fn_rate']:.4f}  (天空漏檢)")
    print(f"  Val (night-ish)   IoU: {val_metrics['iou_night']:.4f}  Dice: {val_metrics['dice_night']:.4f}  PixelAcc: {val_metrics['pixel_acc_night']:.4f}  FP: {val_metrics['fp_rate_night']:.4f}  FN: {val_metrics['fn_rate_night']:.4f}")
    print(f"  Val (non-night)   IoU: {val_metrics['iou_non_night']:.4f}  Dice: {val_metrics['dice_non_night']:.4f}  PixelAcc: {val_metrics['pixel_acc_non_night']:.4f}  FP: {val_metrics['fp_rate_non_night']:.4f}  FN: {val_metrics['fn_rate_non_night']:.4f}")
    print(f"  [luma] T={luma_info.get('T', 0):.3f}, night-ish={luma_info.get('night_count', 0)}, non-night={luma_info.get('non_night_count', 0)}")
    print()

    out_path = os.path.join('outputs', 'val_9483_luma_groups.csv')
    os.makedirs('outputs', exist_ok=True)
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['filename', 'mean_luma', 'group'])
        for p in luma_info.get('per_image', []):
            w.writerow([p['filename'], f"{p['mean_luma']:.6f}", p['group']])
    print(f"  已寫入 {out_path}")
    print()

    print("=" * 60)
    print("  完成")
    print("=" * 60)


if __name__ == '__main__':
    main()
