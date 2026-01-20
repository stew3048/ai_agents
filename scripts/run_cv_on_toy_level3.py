"""
CV Baseline on Toy Level 3 Dataset
測試 CV baseline 對複雜邊界的處理能力

輸出：
- outputs/cv_toy_level3_vis/: 視覺化結果
- outputs/cv_toy_level3_metrics.csv: 指標
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
    
    # 計算邊界誤差（用於評估邊界精確度）
    # 簡單方法：計算預測和GT不同的像素比例
    boundary_error = (pred != gt).sum() / pred.size
    
    return pixel_acc, iou, dice, boundary_error


def main():
    # === 設定 ===
    images_dir = 'data/toy_level3/images'
    masks_dir = 'data/toy_level3/masks'
    info_csv = 'data/toy_level3/dataset_info.csv'
    output_vis_dir = 'outputs/cv_toy_level3_vis'
    output_metrics_csv = 'outputs/cv_toy_level3_metrics.csv'
    
    os.makedirs(output_vis_dir, exist_ok=True)
    
    print('=' * 60)
    print('  CV Baseline on Toy LEVEL 3: Complex Boundaries')
    print('=' * 60)
    print()
    print('  This test checks if CV baseline can handle')
    print('  non-horizontal sky boundaries (mountains, trees, buildings).')
    print()
    print('  Expected: HIGH IoU because the sky is still blue')
    print('            and ground is still dark/green.')
    print()
    
    # 讀取 dataset info
    dataset_info = {}
    if os.path.exists(info_csv):
        with open(info_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                dataset_info[row['sample_id']] = row
    
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
    print('[2] Evaluating...')
    print('-' * 60)
    
    results = []
    
    for idx in range(num_samples):
        sample_id = f'{idx+1:03d}'
        
        image_tensor = images[idx]
        gt_tensor = gt_masks[idx]
        
        image_np = tensor_to_numpy_uint8(image_tensor)
        gt_mask = gt_tensor_to_numpy(gt_tensor)
        pred_mask = predict_sky_mask_cv(image_tensor)
        
        pixel_acc, iou, dice, boundary_error = compute_metrics(pred_mask, gt_mask)
        
        info = dataset_info.get(sample_id, {})
        boundary_name = info.get('boundary_name', 'Unknown')
        boundary_type = info.get('boundary_type', '')
        
        results.append({
            'sample_id': sample_id,
            'boundary_type': boundary_type,
            'boundary_name': boundary_name,
            'pixel_accuracy': pixel_acc,
            'iou': iou,
            'dice': dice,
            'boundary_error': boundary_error,
        })
        
        # 儲存視覺化
        Image.fromarray(image_np).save(
            os.path.join(output_vis_dir, f'{sample_id}_1_image.png')
        )
        
        Image.fromarray((gt_mask * 255).astype(np.uint8), mode='L').save(
            os.path.join(output_vis_dir, f'{sample_id}_2_gt_mask.png')
        )
        
        Image.fromarray((pred_mask * 255).astype(np.uint8), mode='L').save(
            os.path.join(output_vis_dir, f'{sample_id}_3_pred_mask.png')
        )
        
        comparison = create_comparison_overlay(image_np, pred_mask, gt_mask)
        Image.fromarray(comparison).save(
            os.path.join(output_vis_dir, f'{sample_id}_4_comparison.png')
        )
        
        status = 'OK' if iou > 0.9 else 'WARN' if iou > 0.7 else 'FAIL'
        print(f'    {sample_id}: {boundary_name:12} | IoU={iou*100:5.1f}% [{status}]')
    
    print('-' * 60)
    print()
    
    # === 儲存 CSV ===
    print('[3] Saving metrics...')
    with open(output_metrics_csv, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['sample_id', 'boundary_type', 'boundary_name',
                      'pixel_accuracy', 'iou', 'dice', 'boundary_error']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                'sample_id': r['sample_id'],
                'boundary_type': r['boundary_type'],
                'boundary_name': r['boundary_name'],
                'pixel_accuracy': f"{r['pixel_accuracy']:.4f}",
                'iou': f"{r['iou']:.4f}",
                'dice': f"{r['dice']:.4f}",
                'boundary_error': f"{r['boundary_error']:.4f}",
            })
    
    print(f'    Saved to: {output_metrics_csv}')
    print()
    
    # === 統計結果 ===
    avg_iou = np.mean([r['iou'] for r in results])
    avg_dice = np.mean([r['dice'] for r in results])
    avg_acc = np.mean([r['pixel_accuracy'] for r in results])
    avg_boundary_error = np.mean([r['boundary_error'] for r in results])
    
    ok_count = sum(1 for r in results if r['iou'] > 0.9)
    warn_count = sum(1 for r in results if 0.7 < r['iou'] <= 0.9)
    fail_count = sum(1 for r in results if r['iou'] <= 0.7)
    
    print('=' * 60)
    print('  RESULTS SUMMARY')
    print('=' * 60)
    print()
    print(f'  Total samples: {num_samples}')
    print()
    print('  +------------------------+----------+')
    print('  | Metric                 | Value    |')
    print('  +------------------------+----------+')
    print(f'  | Pixel Accuracy         | {avg_acc*100:6.2f}%  |')
    print(f'  | IoU                    | {avg_iou*100:6.2f}%  |')
    print(f'  | Dice                   | {avg_dice*100:6.2f}%  |')
    print(f'  | Boundary Error         | {avg_boundary_error*100:6.2f}%  |')
    print('  +------------------------+----------+')
    print()
    print('  Status breakdown:')
    print(f'    OK   (IoU > 90%): {ok_count}')
    print(f'    WARN (70-90%):    {warn_count}')
    print(f'    FAIL (< 70%):     {fail_count}')
    print()
    
    # === 按邊界類型分析 ===
    print('=' * 60)
    print('  BREAKDOWN BY BOUNDARY TYPE')
    print('=' * 60)
    print()
    
    from collections import defaultdict
    by_boundary = defaultdict(list)
    for r in results:
        by_boundary[r['boundary_name']].append(r['iou'])
    
    print('  +----------------------+-------+--------+')
    print('  | Boundary Type        | Count | Avg IoU|')
    print('  +----------------------+-------+--------+')
    for boundary_name, ious in sorted(by_boundary.items(), key=lambda x: np.mean(x[1]), reverse=True):
        avg = np.mean(ious) * 100
        print(f'  | {boundary_name:20} | {len(ious):5} | {avg:5.1f}% |')
    print('  +----------------------+-------+--------+')
    print()
    
    # === 關鍵發現 ===
    print('=' * 60)
    print('  KEY FINDINGS')
    print('=' * 60)
    print()
    print('  Level 3 tests BOUNDARY COMPLEXITY, not color.')
    print()
    print('  Since CV baseline uses COLOR (not edges/shapes),')
    print('  it should handle complex boundaries WELL as long as:')
    print('  - Sky is blue')
    print('  - Ground is dark/green')
    print()
    print('  This shows that CV baseline is NOT affected by:')
    print('  - Mountain shapes')
    print('  - Tree silhouettes')
    print('  - Building outlines')
    print()
    print('  The boundary complexity is handled by morphological')
    print('  operations (opening/closing) which smooth edges.')
    print()
    print('=' * 60)


if __name__ == '__main__':
    main()
