"""
使用 in-domain split 訓練 SkySegModel（DINOv2 + 兩層解碼頭）

- 資料：outputs/train_list.txt（2535 筆）、outputs/val_list.txt（315 筆）
- 模型：models/sky_seg_dinov2_linear.py（Backbone 凍結，只訓練 Fusion + Prediction）
- Loss：BCEWithLogitsLoss + Dice（CombinedLoss），輸出 logits
- 參數：batch_size=2, epochs=10, lr=1e-4, image_size=(160,160)

輸出：
- outputs/train_dino_linear_YYYYMMDD_HHMMSS/checkpoints/best.pth
- outputs/train_dino_linear_YYYYMMDD_HHMMSS/training_log.csv
"""

import os
import csv
import sys

_script_dir = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_script_dir)
sys.path.insert(0, _root)

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

import torch
import torch.optim as optim
from datetime import datetime

from models import create_sky_seg_dinov2_linear
from utils.dataset import get_dataloader
from scripts.train_in_domain import (
    load_list_file,
    CombinedLoss,
    train_epoch,
    evaluate,
    save_checkpoint,
)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='訓練 DINOv2 兩層解碼頭（in-domain split）')
    parser.add_argument('--device', type=str, default=None, help='cpu / cuda，預設自動偵測')
    args = parser.parse_args()

    print("=" * 60)
    print("  DINOv2 兩層解碼頭 in-domain 訓練")
    print("=" * 60)
    print()

    batch_size = 2
    epochs = 10
    learning_rate = 1e-4
    seed = 42
    image_size = (160, 160)
    num_workers = 0

    if args.device == 'cpu':
        device = torch.device('cpu')
        use_amp = False
    elif args.device == 'cuda':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        use_amp = device.type == 'cuda'
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        use_amp = device.type == 'cuda'

    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)

    print(f"設備: {device}, 混合精度: {use_amp}")
    print(f"batch_size={batch_size}, epochs={epochs}, lr={learning_rate}, image_size={image_size}")
    print()

    train_list_file = os.path.join(_root, "outputs", "train_list.txt")
    val_list_file = os.path.join(_root, "outputs", "val_list.txt")

    print(f"載入 train list: {train_list_file}")
    train_split_list = load_list_file(train_list_file, base_data_dir=os.path.join(_root, "data"))
    print(f"  共 {len(train_split_list)} 張")
    print(f"載入 val list: {val_list_file}")
    val_split_list = load_list_file(val_list_file, base_data_dir=os.path.join(_root, "data"))
    print(f"  共 {len(val_split_list)} 張")
    print()

    base_data_dir = os.path.join(_root, "data")
    train_loader = get_dataloader(
        batch_size=batch_size,
        shuffle=True,
        transform=True,
        image_size=image_size,
        num_workers=num_workers,
        split_list=train_split_list,
        base_data_dir=base_data_dir,
    )
    val_loader = get_dataloader(
        batch_size=batch_size,
        shuffle=False,
        transform=False,
        image_size=image_size,
        num_workers=num_workers,
        split_list=val_split_list,
        base_data_dir=base_data_dir,
    )
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    print()

    print("建立模型: SkySegModel (DINOv2 + Fusion 384->256 + Prediction 256->1)")
    model = create_sky_seg_dinov2_linear().to(device)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  總參數: {total:,}, 可訓練: {trainable:,}")
    print()

    criterion = CombinedLoss(bce_weight=1.0, dice_weight=1.0)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(_root, "outputs", f"train_dino_linear_{timestamp}")
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"輸出目錄: {output_dir}")
    print()

    training_log = []
    best_val_iou = 0.0

    print("開始訓練...")
    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}/{epochs}")
        train_loss, train_iou = train_epoch(model, train_loader, criterion, optimizer, device, use_amp)
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device, use_amp)
        val_iou = val_metrics['iou']

        log_entry = {
            'epoch': epoch,
            'train_loss': train_loss,
            'train_iou': train_iou,
            'val_loss': val_loss,
            'val_iou': val_iou,
            'val_dice': val_metrics['dice'],
            'val_pixel_acc': val_metrics['pixel_acc'],
            'val_fp_rate': val_metrics['fp_rate'],
            'val_fn_rate': val_metrics['fn_rate'],
        }
        training_log.append(log_entry)

        print(f"  Train Loss: {train_loss:.4f}, Train IoU: {train_iou:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}, Val Dice: {val_metrics['dice']:.4f}")
        print(f"  Val FP: {val_metrics['fp_rate']:.4f}, Val FN: {val_metrics['fn_rate']:.4f}")

        is_best = val_iou > best_val_iou
        if is_best:
            best_val_iou = val_iou
        save_checkpoint(model, optimizer, epoch, val_iou, checkpoint_dir, is_best=is_best)
        print()

    log_file = os.path.join(output_dir, "training_log.csv")
    with open(log_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=training_log[0].keys())
        writer.writeheader()
        writer.writerows(training_log)

    print("=" * 60)
    print("  訓練完成")
    print("=" * 60)
    print(f"Best Val IoU: {best_val_iou:.4f}")
    print(f"輸出目錄: {output_dir}")
    print(f"Best checkpoint: {os.path.join(checkpoint_dir, 'best.pth')}")
    print()


if __name__ == "__main__":
    main()
