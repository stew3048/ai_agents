"""
CV Baseline on Toy Level 4 Dataset
測試 CV baseline 對「天空與地面顏色相近」的分辨能力

輸出：
- outputs/cv_toy_level4_vis/: 視覺化結果
- outputs/cv_toy_level4_metrics.csv: 指標
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
    images_dir = 'data/toy_level4/images'
    masks_dir = 'data/toy_level4/masks'
    info_csv = 'data/toy_level4/dataset_info.csv'
    output_vis_dir = 'outputs/cv_toy_level4_vis'
    output_metrics_csv = 'outputs/cv_toy_level4_metrics.csv'
    
    os.makedirs(output_vis_dir, exist_ok=True)
    
    print('=' * 60)
    print('  CV Baseline on Toy LEVEL 4: Multi-terrain Ground')
    print('=' * 60)
    print()
    print('  This test checks if CV baseline can distinguish')
    print('  sky from similar-colored ground (snow, ocean, desert).')
    print()
    print('  Expected issues:')
    print('    - Blue sky + Blue ocean = confusion')
    print('    - White sky + White snow = confusion')
    print('    - Orange sky + Yellow desert = confusion')
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
        
        pixel_acc, iou, dice, fpr, fnr = compute_metrics(pred_mask, gt_mask)
        
        info = dataset_info.get(sample_id, {})
        sky_name = info.get('sky_name', 'Unknown')
        ground_name = info.get('ground_name', 'Unknown')
        color_distance = float(info.get('color_distance', 0))
        is_confusing = info.get('is_confusing', 'False') == 'True'
        
        results.append({
            'sample_id': sample_id,
            'sky_name': sky_name,
            'ground_name': ground_name,
            'color_distance': color_distance,
            'is_confusing': is_confusing,
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
        
        # 狀態判斷
        if iou < 0.5:
            status = 'FAIL'
        elif iou < 0.8:
            status = 'WARN'
        else:
            status = 'OK'
        
        confuse_mark = '*' if is_confusing else ' '
        print(f'    {sample_id}: {sky_name:10} + {ground_name:10} | IoU={iou*100:5.1f}% [{status}] {confuse_mark}')
    
    print('-' * 60)
    print('  (* = sky-ground color confusion pair)')
    print()
    
    # === 儲存 CSV ===
    print('[3] Saving metrics...')
    with open(output_metrics_csv, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['sample_id', 'sky_name', 'ground_name', 'color_distance', 'is_confusing',
                      'pixel_accuracy', 'iou', 'dice', 'fpr', 'fnr']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                'sample_id': r['sample_id'],
                'sky_name': r['sky_name'],
                'ground_name': r['ground_name'],
                'color_distance': f"{r['color_distance']:.1f}",
                'is_confusing': r['is_confusing'],
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
    avg_fpr = np.mean([r['fpr'] for r in results])
    avg_fnr = np.mean([r['fnr'] for r in results])
    
    ok_count = sum(1 for r in results if r['iou'] >= 0.8)
    warn_count = sum(1 for r in results if 0.5 <= r['iou'] < 0.8)
    fail_count = sum(1 for r in results if r['iou'] < 0.5)
    
    print('=' * 60)
    print('  RESULTS SUMMARY')
    print('=' * 60)
    print()
    print(f'  Total samples: {num_samples}')
    print()
    print('  +------------------------+----------+')
    print('  | Metric                 | Value    |')
    print('  +------------------------+----------+')
    print(f'  | IoU                    | {avg_iou*100:6.2f}%  |')
    print(f'  | False Positive Rate    | {avg_fpr*100:6.2f}%  |')
    print(f'  | False Negative Rate    | {avg_fnr*100:6.2f}%  |')
    print('  +------------------------+----------+')
    print()
    print('  Status breakdown:')
    print(f'    OK   (IoU >= 80%): {ok_count}')
    print(f'    WARN (50-80%):     {warn_count}')
    print(f'    FAIL (< 50%):      {fail_count}')
    print()
    
    # === 混淆對 vs 非混淆對分析 ===
    confusing_results = [r for r in results if r['is_confusing']]
    non_confusing_results = [r for r in results if not r['is_confusing']]
    
    print('=' * 60)
    print('  CONFUSING vs NON-CONFUSING PAIRS')
    print('=' * 60)
    print()
    
    if confusing_results:
        avg_iou_confusing = np.mean([r['iou'] for r in confusing_results])
        avg_fpr_confusing = np.mean([r['fpr'] for r in confusing_results])
        fail_confusing = sum(1 for r in confusing_results if r['iou'] < 0.5)
        print(f'  Confusing pairs ({len(confusing_results)}):')
        print(f'    Avg IoU: {avg_iou_confusing*100:.1f}%')
        print(f'    Avg FPR: {avg_fpr_confusing*100:.1f}%')
        print(f'    FAIL count: {fail_confusing}')
        print()
    
    if non_confusing_results:
        avg_iou_non = np.mean([r['iou'] for r in non_confusing_results])
        avg_fpr_non = np.mean([r['fpr'] for r in non_confusing_results])
        fail_non = sum(1 for r in non_confusing_results if r['iou'] < 0.5)
        print(f'  Non-confusing pairs ({len(non_confusing_results)}):')
        print(f'    Avg IoU: {avg_iou_non*100:.1f}%')
        print(f'    Avg FPR: {avg_fpr_non*100:.1f}%')
        print(f'    FAIL count: {fail_non}')
        print()
    
    # === 按地面類型分析 ===
    print('=' * 60)
    print('  BREAKDOWN BY GROUND TYPE')
    print('=' * 60)
    print()
    
    from collections import defaultdict
    by_ground = defaultdict(list)
    for r in results:
        by_ground[r['ground_name']].append(r)
    
    print('  +------------------+-------+--------+--------+')
    print('  | Ground Type      | Count | Avg IoU| Avg FPR|')
    print('  +------------------+-------+--------+--------+')
    for ground_name, group in sorted(by_ground.items(), key=lambda x: np.mean([r['iou'] for r in x[1]])):
        avg_iou_g = np.mean([r['iou'] for r in group]) * 100
        avg_fpr_g = np.mean([r['fpr'] for r in group]) * 100
        status = 'FAIL' if avg_iou_g < 50 else 'WARN' if avg_iou_g < 80 else ''
        print(f'  | {ground_name:16} | {len(group):5} | {avg_iou_g:5.1f}% | {avg_fpr_g:5.1f}% | {status}')
    print('  +------------------+-------+--------+--------+')
    print()
    
    # === 關鍵發現 ===
    print('=' * 60)
    print('  KEY FINDINGS')
    print('=' * 60)
    print()
    print('  Level 4 tests COLOR CONFUSION between sky and ground.')
    print()
    print('  CV baseline struggles when:')
    print('    - Blue sky meets blue ocean/lake/glacier')
    print('    - White sky meets white snow')
    print('    - Non-blue sky is not detected at all')
    print()
    print('  This demonstrates the FUNDAMENTAL LIMITATION:')
    print('    "Color is not semantics"')
    print()
    print('=' * 60)


if __name__ == '__main__':
    main()
