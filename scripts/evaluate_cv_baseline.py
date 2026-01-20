"""
CV Baseline 評估腳本
計算 CV baseline 在 toy dataset 上的效能指標

輸出：
- outputs/cv_toy_metrics.csv：每張圖的指標
- 終端機顯示平均指標

指標說明：
- Pixel Accuracy: 猜對的像素比例（整體正確率）
- IoU: 預測與 GT 的重疊程度（交集/聯集）
- Dice: 預測與 GT 的相似度（醫學影像常用）
"""

import os
import sys
import csv
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.dataset import get_dataloader
from src.models.cv_baseline import predict_sky_mask_cv


def compute_pixel_accuracy(pred, gt):
    """
    計算 Pixel Accuracy
    
    Pixel Accuracy = (TP + TN) / Total
                   = 猜對的像素數 / 總像素數
    
    Args:
        pred: numpy array [H, W], values {0, 1}
        gt: numpy array [H, W], values {0, 1}
    
    Returns:
        float: Pixel Accuracy (0~1)
    """
    correct = (pred == gt).sum()
    total = pred.size
    return correct / total


def compute_iou(pred, gt, smooth=1e-6):
    """
    計算 IoU (Intersection over Union)
    
    IoU = (Pred ∩ GT) / (Pred ∪ GT)
        = 交集 / 聯集
    
    Args:
        pred: numpy array [H, W], values {0, 1}
        gt: numpy array [H, W], values {0, 1}
        smooth: 避免除以 0 的小數值
    
    Returns:
        float: IoU (0~1)
    """
    intersection = ((pred == 1) & (gt == 1)).sum()
    union = ((pred == 1) | (gt == 1)).sum()
    return (intersection + smooth) / (union + smooth)


def compute_dice(pred, gt, smooth=1e-6):
    """
    計算 Dice Coefficient
    
    Dice = 2 × (Pred ∩ GT) / (|Pred| + |GT|)
         = 2 × 交集 / (預測面積 + GT 面積)
    
    Args:
        pred: numpy array [H, W], values {0, 1}
        gt: numpy array [H, W], values {0, 1}
        smooth: 避免除以 0 的小數值
    
    Returns:
        float: Dice (0~1)
    """
    intersection = ((pred == 1) & (gt == 1)).sum()
    pred_area = (pred == 1).sum()
    gt_area = (gt == 1).sum()
    return (2 * intersection + smooth) / (pred_area + gt_area + smooth)


def gt_tensor_to_numpy(gt_tensor):
    """
    將 GT mask tensor 轉換為 numpy {0, 1}
    
    Args:
        gt_tensor: torch tensor [1, H, W], float32, values {0.0, 1.0}
    
    Returns:
        numpy array [H, W], uint8, values {0, 1}
    """
    # [1, H, W] -> [H, W]
    gt_np = gt_tensor.squeeze(0).numpy()
    # 轉成 {0, 1}
    gt_np = (gt_np > 0.5).astype(np.uint8)
    return gt_np


def main():
    # === 設定 ===
    images_dir = 'data/toy/images'
    masks_dir = 'data/toy/masks'
    output_csv = 'outputs/cv_toy_metrics.csv'
    
    os.makedirs('outputs', exist_ok=True)
    
    print('=' * 60)
    print('  CV Baseline Evaluation on Toy Dataset')
    print('=' * 60)
    print()
    
    # === 載入 DataLoader ===
    print('[1] Loading DataLoader...')
    dataloader = get_dataloader(
        images_dir=images_dir,
        masks_dir=masks_dir,
        batch_size=50,  # 一次載入全部
        shuffle=False,
        transform=False
    )
    
    images, gt_masks = next(iter(dataloader))
    num_samples = images.shape[0]
    print(f'    Loaded {num_samples} images')
    print()
    
    # === 評估每張圖 ===
    print('[2] Evaluating...')
    print('-' * 60)
    
    results = []
    
    for idx in range(num_samples):
        sample_id = f'{idx+1:03d}'
        
        # 取得 image 和 GT
        image_tensor = images[idx]
        gt_tensor = gt_masks[idx]
        
        # CV baseline 預測
        pred_mask = predict_sky_mask_cv(image_tensor)  # [H, W], {0, 1}
        
        # GT 轉換
        gt_mask = gt_tensor_to_numpy(gt_tensor)  # [H, W], {0, 1}
        
        # 計算指標
        pixel_acc = compute_pixel_accuracy(pred_mask, gt_mask)
        iou = compute_iou(pred_mask, gt_mask)
        dice = compute_dice(pred_mask, gt_mask)
        
        results.append({
            'sample_id': sample_id,
            'pixel_accuracy': pixel_acc,
            'iou': iou,
            'dice': dice
        })
        
        # 每 10 張顯示一次
        if (idx + 1) % 10 == 0 or idx == num_samples - 1:
            print(f'    Processed {idx + 1}/{num_samples}')
    
    print('-' * 60)
    print()
    
    # === 計算平均 ===
    avg_pixel_acc = np.mean([r['pixel_accuracy'] for r in results])
    avg_iou = np.mean([r['iou'] for r in results])
    avg_dice = np.mean([r['dice'] for r in results])
    
    # === 輸出 CSV ===
    print('[3] Saving results...')
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['sample_id', 'pixel_accuracy', 'iou', 'dice'])
        writer.writeheader()
        for r in results:
            writer.writerow({
                'sample_id': r['sample_id'],
                'pixel_accuracy': f"{r['pixel_accuracy']:.4f}",
                'iou': f"{r['iou']:.4f}",
                'dice': f"{r['dice']:.4f}"
            })
        # 加入平均值
        writer.writerow({
            'sample_id': 'AVERAGE',
            'pixel_accuracy': f"{avg_pixel_acc:.4f}",
            'iou': f"{avg_iou:.4f}",
            'dice': f"{avg_dice:.4f}"
        })
    
    print(f'    Saved to: {output_csv}')
    print()
    
    # === 顯示結果 ===
    print('=' * 60)
    print('  RESULTS SUMMARY')
    print('=' * 60)
    print()
    print(f'  Total samples: {num_samples}')
    print()
    print('  +------------------+----------+')
    print('  | Metric           | Value    |')
    print('  +------------------+----------+')
    print(f'  | Pixel Accuracy   | {avg_pixel_acc*100:6.2f}%  |')
    print(f'  | IoU              | {avg_iou*100:6.2f}%  |')
    print(f'  | Dice             | {avg_dice*100:6.2f}%  |')
    print('  +------------------+----------+')
    print()
    
    # === 解讀指南 ===
    print('=' * 60)
    print('  HOW TO INTERPRET')
    print('=' * 60)
    print()
    print('  [Pixel Accuracy]')
    print('    - How many pixels are correctly classified?')
    print('    - Can be misleading if sky is small portion')
    print()
    print('  [IoU] (most important for segmentation)')
    print('    - How much does prediction overlap with GT?')
    print('    - 0% = no overlap, 100% = perfect match')
    print('    - >50% is decent, >70% is good')
    print()
    print('  [Dice]')
    print('    - Similar to IoU but slightly more lenient')
    print('    - Always >= IoU')
    print()
    print('=' * 60)


if __name__ == '__main__':
    main()
