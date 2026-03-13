"""
CV Baseline on Real SkyFinder Dataset
在真實資料上測試 CV baseline

SkyFinder 資料特點：
- 真實戶外 webcam 圖片
- 640x480 原始尺寸
- 各種天氣、光線條件
- 複雜的天空邊界（山脈、建築、樹木）
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
    """計算所有指標"""
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
    
    fpr = fp / (fp + tn + smooth)
    fnr = fn / (fn + intersection + smooth)
    
    return pixel_acc, iou, dice, fpr, fnr


def main():
    # === 設定 ===
    images_dir = 'data/skyfinder_sample/images'
    masks_dir = 'data/skyfinder_sample/masks'
    output_vis_dir = 'outputs/cv_skyfinder_vis'
    output_metrics_csv = 'outputs/cv_skyfinder_metrics.csv'
    
    os.makedirs(output_vis_dir, exist_ok=True)
    
    print('=' * 60)
    print('  CV Baseline on REAL SkyFinder Dataset')
    print('=' * 60)
    print()
    
    # 檢查原始圖片尺寸
    sample_img = Image.open(os.path.join(images_dir, '001.jpg'))
    original_size = sample_img.size
    print(f'  Original image size: {original_size[0]} x {original_size[1]}')
    print(f'  DataLoader target:   256 x 256')
    print()
    print('  This is REAL outdoor webcam data with:')
    print('    - Various weather conditions')
    print('    - Different lighting (sunrise, noon, sunset)')
    print('    - Complex sky boundaries (mountains, buildings)')
    print()
    
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
    print(f'    Resized to: {images.shape[2]} x {images.shape[3]}')
    print()
    
    # === 評估 ===
    print('[2] Evaluating CV baseline on real images...')
    print('-' * 60)
    
    results = []
    
    for idx in range(num_samples):
        sample_id = f'{idx+1:03d}'
        
        image_tensor = images[idx]
        gt_tensor = gt_masks[idx]
        
        image_np = tensor_to_numpy_uint8(image_tensor)
        gt_mask = gt_tensor_to_numpy(gt_tensor)
        pred_mask = predict_sky_mask_cv(image_tensor)
        
        pixel_acc, iou, dice, fpr, fnr = compute_metrics(pred_mask, gt_mask)
        
        results.append({
            'sample_id': sample_id,
            'pixel_accuracy': pixel_acc,
            'iou': iou,
            'dice': dice,
            'fpr': fpr,
            'fnr': fnr,
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
        
        status = 'OK' if iou > 0.7 else 'WARN' if iou > 0.4 else 'FAIL'
        print(f'    {sample_id}: IoU={iou*100:5.1f}% | FPR={fpr*100:4.1f}% | FNR={fnr*100:4.1f}% [{status}]')
    
    print('-' * 60)
    print()
    
    # === 儲存 CSV ===
    print('[3] Saving metrics...')
    with open(output_metrics_csv, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['sample_id', 'pixel_accuracy', 'iou', 'dice', 'fpr', 'fnr']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                'sample_id': r['sample_id'],
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
    avg_dice = np.mean([r['dice'] for r in results])
    avg_acc = np.mean([r['pixel_accuracy'] for r in results])
    avg_fpr = np.mean([r['fpr'] for r in results])
    avg_fnr = np.mean([r['fnr'] for r in results])
    
    ok_count = sum(1 for r in results if r['iou'] > 0.7)
    warn_count = sum(1 for r in results if 0.4 <= r['iou'] <= 0.7)
    fail_count = sum(1 for r in results if r['iou'] < 0.4)
    
    print('=' * 60)
    print('  RESULTS SUMMARY (Real SkyFinder Data)')
    print('=' * 60)
    print()
    print(f'  Original size:  {original_size[0]} x {original_size[1]}')
    print(f'  Resized to:     256 x 256')
    print(f'  Total samples:  {num_samples}')
    print()
    print('  +------------------------+----------+')
    print('  | Metric                 | Value    |')
    print('  +------------------------+----------+')
    print(f'  | Pixel Accuracy         | {avg_acc*100:6.2f}%  |')
    print(f'  | IoU                    | {avg_iou*100:6.2f}%  |')
    print(f'  | Dice                   | {avg_dice*100:6.2f}%  |')
    print(f'  | False Positive Rate    | {avg_fpr*100:6.2f}%  |')
    print(f'  | False Negative Rate    | {avg_fnr*100:6.2f}%  |')
    print('  +------------------------+----------+')
    print()
    print('  Status breakdown:')
    print(f'    OK   (IoU > 70%): {ok_count}')
    print(f'    WARN (40-70%):    {warn_count}')
    print(f'    FAIL (< 40%):     {fail_count}')
    print()
    
    # === IoU 分布 ===
    ious = [r['iou'] for r in results]
    print('  IoU Distribution:')
    print(f'    Min:    {min(ious)*100:.1f}%')
    print(f'    Max:    {max(ious)*100:.1f}%')
    print(f'    Median: {np.median(ious)*100:.1f}%')
    print(f'    Std:    {np.std(ious)*100:.1f}%')
    print()
    
    # === 關鍵發現 ===
    print('=' * 60)
    print('  KEY FINDINGS ON REAL DATA')
    print('=' * 60)
    print()
    print('  Real-world challenges observed:')
    print('    - Overexposed sky (sun glare) -> not blue')
    print('    - Cloudy/hazy conditions -> gray/white sky')
    print('    - Complex mountain silhouettes')
    print('    - Timestamp text overlay on images')
    print()
    
    if avg_iou > 0.7:
        print('  Result: ACCEPTABLE')
        print('    CV baseline works reasonably on this camera.')
    elif avg_iou > 0.4:
        print('  Result: MODERATE')
        print('    CV baseline has issues, investigate failure cases.')
    else:
        print('  Result: POOR')
        print('    CV baseline fails on real data as expected.')
    
    print()
    print(f'  Output saved to: {output_vis_dir}/')
    print()
    print('=' * 60)


if __name__ == '__main__':
    main()
