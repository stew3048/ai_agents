"""
CV Baseline on Cloudy Sky Dataset
測試 CV baseline 對有雲天空的處理能力

重點：
- 雲是白色/灰色，不是藍色
- 但雲仍然是「天空」的一部分，應該被標記為 1
- CV baseline 用藍色 HSV 範圍偵測天空
- 預期：雲會被漏掉（FN），因為雲不是藍色
"""

import os
import sys
import csv
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.dataset import get_dataloader
from src.models.cv_baseline import predict_sky_mask_cv


def create_comparison_overlay(image_np, pred_mask, gt_mask):
    """建立比較 overlay"""
    result = image_np.astype(np.float32).copy()
    
    tp = (pred_mask == 1) & (gt_mask == 1)
    fp = (pred_mask == 1) & (gt_mask == 0)
    fn = (pred_mask == 0) & (gt_mask == 1)
    
    alpha = 0.5
    
    # Green = TP (correct sky)
    # Red = FP (ground marked as sky)
    # Blue = FN (sky marked as ground) <- clouds will show up here!
    result[tp] = result[tp] * (1 - alpha) + np.array([0, 255, 0]) * alpha
    result[fp] = result[fp] * (1 - alpha) + np.array([255, 0, 0]) * alpha
    result[fn] = result[fn] * (1 - alpha) + np.array([0, 0, 255]) * alpha
    
    return result.clip(0, 255).astype(np.uint8)


def tensor_to_numpy_uint8(tensor):
    arr = tensor.permute(1, 2, 0).numpy()
    arr = (arr * 255).clip(0, 255).astype(np.uint8)
    return arr


def gt_tensor_to_numpy(gt_tensor):
    gt_np = gt_tensor.squeeze(0).numpy()
    gt_np = (gt_np > 0.5).astype(np.uint8)
    return gt_np


def compute_metrics(pred, gt):
    """計算指標"""
    smooth = 1e-6
    
    pixel_acc = (pred == gt).sum() / pred.size
    
    intersection = ((pred == 1) & (gt == 1)).sum()
    union = ((pred == 1) | (gt == 1)).sum()
    iou = (intersection + smooth) / (union + smooth)
    
    pred_area = (pred == 1).sum()
    gt_area = (gt == 1).sum()
    dice = (2 * intersection + smooth) / (pred_area + gt_area + smooth)
    
    # FP 和 FN
    fp = ((pred == 1) & (gt == 0)).sum()
    fn = ((pred == 0) & (gt == 1)).sum()
    tn = ((pred == 0) & (gt == 0)).sum()
    tp = intersection
    
    fpr = fp / (fp + tn + smooth)
    fnr = fn / (fn + tp + smooth)  # 這個會顯示雲被漏掉的比例
    
    return pixel_acc, iou, dice, fpr, fnr, fn


def main():
    # === 設定 ===
    images_dir = 'data/toy_cloudy/images'
    masks_dir = 'data/toy_cloudy/masks'
    output_vis_dir = 'outputs/cv_toy_cloudy_vis'
    output_metrics_csv = 'outputs/cv_toy_cloudy_metrics.csv'
    info_csv = 'data/toy_cloudy/dataset_info.csv'
    
    os.makedirs(output_vis_dir, exist_ok=True)
    
    print('=' * 60)
    print('  CV Baseline on Cloudy Sky Dataset')
    print('=' * 60)
    print()
    print('  Test Focus:')
    print('    - Clouds are WHITE/GRAY, not blue')
    print('    - Clouds are PART OF SKY (mask = 1)')
    print('    - CV baseline detects BLUE using HSV')
    print('    - Expected: Clouds will be MISSED (FN)')
    print()
    
    # 讀取資料集資訊
    cloud_info = {}
    if os.path.exists(info_csv):
        with open(info_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                cloud_info[row['image_id']] = {
                    'sky_type': row['sky_type'],
                    'cloud_coverage': float(row['cloud_coverage']),
                }
    
    # === 載入 DataLoader ===
    print('[1] Loading DataLoader...')
    dataloader = get_dataloader(
        images_dir=images_dir,
        masks_dir=masks_dir,
        batch_size=50,
        shuffle=False,
        transform=False
    )
    
    images, gt_masks = next(iter(dataloader))
    num_samples = images.shape[0]
    print(f'    Loaded {num_samples} images')
    print()
    
    # === 評估 ===
    print('[2] Evaluating CV baseline on cloudy images...')
    print('-' * 60)
    
    results = []
    
    for idx in range(num_samples):
        sample_id = f'{idx+1:03d}'
        
        image_tensor = images[idx]
        gt_tensor = gt_masks[idx]
        
        image_np = tensor_to_numpy_uint8(image_tensor)
        gt_mask = gt_tensor_to_numpy(gt_tensor)
        pred_mask = predict_sky_mask_cv(image_tensor)
        
        pixel_acc, iou, dice, fpr, fnr, fn_pixels = compute_metrics(pred_mask, gt_mask)
        
        # 取得雲覆蓋率
        coverage = cloud_info.get(sample_id, {}).get('cloud_coverage', 0)
        
        results.append({
            'sample_id': sample_id,
            'cloud_coverage': coverage,
            'pixel_accuracy': pixel_acc,
            'iou': iou,
            'dice': dice,
            'fpr': fpr,
            'fnr': fnr,
            'fn_pixels': fn_pixels,
        })
        
        # 儲存視覺化
        Image.fromarray(image_np).save(
            os.path.join(output_vis_dir, f'{sample_id}_1_image.png')
        )
        
        Image.fromarray((gt_mask * 255).astype(np.uint8)).save(
            os.path.join(output_vis_dir, f'{sample_id}_2_gt_mask.png')
        )
        
        Image.fromarray((pred_mask * 255).astype(np.uint8)).save(
            os.path.join(output_vis_dir, f'{sample_id}_3_pred_mask.png')
        )
        
        comparison = create_comparison_overlay(image_np, pred_mask, gt_mask)
        Image.fromarray(comparison).save(
            os.path.join(output_vis_dir, f'{sample_id}_4_comparison.png')
        )
        
        status = 'OK' if iou > 0.9 else 'WARN' if iou > 0.7 else 'FAIL'
        print(f'    {sample_id}: Cloud={coverage*100:3.0f}% | IoU={iou*100:5.1f}% | FNR={fnr*100:5.1f}% [{status}]')
    
    print('-' * 60)
    print()
    
    # === 儲存 CSV ===
    print('[3] Saving metrics...')
    with open(output_metrics_csv, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['sample_id', 'cloud_coverage', 'pixel_accuracy', 'iou', 'dice', 'fpr', 'fnr']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                'sample_id': r['sample_id'],
                'cloud_coverage': f"{r['cloud_coverage']:.2f}",
                'pixel_accuracy': f"{r['pixel_accuracy']:.4f}",
                'iou': f"{r['iou']:.4f}",
                'dice': f"{r['dice']:.4f}",
                'fpr': f"{r['fpr']:.4f}",
                'fnr': f"{r['fnr']:.4f}",
            })
    print(f'    Saved to: {output_metrics_csv}')
    print()
    
    # === 統計結果 ===
    avg_iou = np.mean([r['iou'] for r in results])
    avg_fnr = np.mean([r['fnr'] for r in results])
    
    print('=' * 60)
    print('  RESULTS SUMMARY')
    print('=' * 60)
    print()
    print('  Overall metrics:')
    print(f'    Average IoU: {avg_iou*100:.2f}%')
    print(f'    Average FNR: {avg_fnr*100:.2f}% (cloud pixels missed)')
    print()
    
    # === 按雲覆蓋率分析 ===
    print('  IoU by Cloud Coverage:')
    print('  +------------------+-------+-------+--------+')
    print('  | Cloud Coverage   | Count | IoU   | FNR    |')
    print('  +------------------+-------+-------+--------+')
    
    coverage_levels = [
        (0.0, '0% (clear)'),
        (0.15, '15% (few)'),
        (0.35, '35% (scattered)'),
        (0.55, '55% (broken)'),
        (0.75, '75% (overcast)'),
    ]
    
    for cov, label in coverage_levels:
        level_results = [r for r in results if abs(r['cloud_coverage'] - cov) < 0.05]
        if level_results:
            level_iou = np.mean([r['iou'] for r in level_results])
            level_fnr = np.mean([r['fnr'] for r in level_results])
            count = len(level_results)
            print(f'  | {label:16} | {count:5} | {level_iou*100:5.1f}% | {level_fnr*100:5.1f}%  |')
    
    print('  +------------------+-------+-------+--------+')
    print()
    
    # === 關鍵發現 ===
    print('=' * 60)
    print('  KEY FINDINGS')
    print('=' * 60)
    print()
    print('  Hypothesis:')
    print('    - CV baseline uses BLUE HSV range to detect sky')
    print('    - Clouds are WHITE/GRAY, outside blue range')
    print('    - Therefore, clouds should be MISSED (FN)')
    print()
    
    # 計算無雲 vs 有雲的差異
    no_cloud = [r for r in results if r['cloud_coverage'] == 0]
    with_cloud = [r for r in results if r['cloud_coverage'] > 0]
    
    if no_cloud and with_cloud:
        no_cloud_iou = np.mean([r['iou'] for r in no_cloud])
        with_cloud_iou = np.mean([r['iou'] for r in with_cloud])
        
        print('  Comparison:')
        print(f'    Clear sky (0% cloud):  IoU = {no_cloud_iou*100:.1f}%')
        print(f'    Cloudy sky (>0% cloud): IoU = {with_cloud_iou*100:.1f}%')
        print(f'    Difference: {(no_cloud_iou - with_cloud_iou)*100:.1f}%')
        print()
        
        if no_cloud_iou > with_cloud_iou + 0.05:
            print('  Result: HYPOTHESIS CONFIRMED')
            print('    Clouds cause IoU to drop!')
            print('    CV baseline cannot detect white/gray clouds.')
        else:
            print('  Result: UNEXPECTED')
            print('    Cloud coverage did not significantly affect IoU.')
    
    print()
    print('  Overlay Legend:')
    print('    GREEN = Correct sky detection (TP)')
    print('    BLUE  = Sky missed (FN) <- CLOUDS APPEAR HERE')
    print('    RED   = False sky detection (FP)')
    print()
    print(f'  Output saved to: {output_vis_dir}/')
    print()
    print('=' * 60)


if __name__ == '__main__':
    main()
