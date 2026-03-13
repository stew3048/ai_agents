"""
繪製訓練曲線：train/val loss、accuracy（pixel_acc / IoU）
讀取 training_log.csv，輸出 PNG 到同目錄。
"""
import os
import sys
import argparse
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def load_log(csv_path):
    """讀取 training_log.csv，回傳 dict of lists."""
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return None
    out = {k: [] for k in rows[0].keys()}
    for row in rows:
        for k in out:
            try:
                out[k].append(float(row[k]))
            except ValueError:
                out[k].append(row[k])
    return out

def main():
    parser = argparse.ArgumentParser(description='繪製訓練曲線')
    parser.add_argument('log_dir', type=str, nargs='?',
                        default=os.path.join(os.path.dirname(__file__), '..', 'outputs', 'train_dino_linear_20260214_022830'),
                        help='訓練輸出目錄（內含 training_log.csv）')
    args = parser.parse_args()

    csv_path = os.path.join(args.log_dir, 'training_log.csv')
    if not os.path.exists(csv_path):
        print(f'找不到: {csv_path}')
        sys.exit(1)

    data = load_log(csv_path)
    if data is None:
        print('CSV 為空')
        sys.exit(1)
    epochs = data['epoch']

    fig, ax1 = plt.subplots(figsize=(8, 5))

    # Left: Loss
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='C0')
    ax1.plot(epochs, data['train_loss'], 'o-', color='C0', label='Train Loss', linewidth=2, markersize=6)
    ax1.plot(epochs, data['val_loss'], 's-', color='C1', label='Val Loss', linewidth=2, markersize=6)
    ax1.tick_params(axis='y', labelcolor='C0')
    ax1.legend(loc='upper right')
    ax1.set_ylim(bottom=0)
    ax1.grid(True, alpha=0.3)

    # Right: Accuracy (Pixel Acc & IoU)
    ax2 = ax1.twinx()
    ax2.set_ylabel('Accuracy / IoU', color='C2')
    ax2.plot(epochs, data['val_pixel_acc'], '^-', color='C2', label='Val Pixel Acc', linewidth=2, markersize=6)
    ax2.plot(epochs, data['train_iou'], 'd-', color='C3', label='Train IoU', linewidth=2, markersize=6)
    ax2.plot(epochs, data['val_iou'], 'v-', color='C4', label='Val IoU', linewidth=2, markersize=6)
    ax2.tick_params(axis='y', labelcolor='C2')
    ax2.legend(loc='center right')
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, alpha=0.3)

    plt.title('DINOv2 Linear Head In-Domain Training')
    fig.tight_layout()
    out_path = os.path.join(args.log_dir, 'training_curves.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'已儲存: {out_path}')


if __name__ == '__main__':
    main()
