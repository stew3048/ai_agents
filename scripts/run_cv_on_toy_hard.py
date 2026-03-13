"""
CV Baseline on Toy Hard Dataset
測試 CV baseline 在「難搞」天空上的表現

輸出：
- outputs/cv_toy_hard_vis/: 視覺化結果
- outputs/cv_toy_hard_metrics.csv: 指標
"""

import os
import sys
import csv
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.dataset import get_dataloader
from src.models.cv_baseline import predict_sky_mask_cv


def create_overlay(image_np, mask_np, color=(255, 50, 50), alpha=0.5):
    """建立 overlay"""
    image_float = image_np.astype(np.float32)
    overlay = np.zeros_like(image_float)
    overlay[:, :, 0] = color[0]
    overlay[:, :, 1] = color[1]
    overlay[:, :, 2] = color[2]
    mask_3ch = np.stack([mask_np] * 3, axis=-1).astype(np.float32)
    result = image_float * (1 - mask_3ch * alpha) + overlay * (mask_3ch * alpha)
    return result.clip(0, 255).astype(np.uint8)


def create_comparison_overlay(image_np, pred_mask, gt_mask):
    """
    建立比較 overlay：
    - 綠色：正確預測為天空 (TP)
    - 紅色：錯誤預測為天空 (FP)
    - 藍色：漏掉的天空 (FN)
    """
    result = image_np.astype(np.float32).copy()
    
    tp = (pred_mask == 1) & (gt_mask == 1)  # True Positive
    fp = (pred_mask == 1) & (gt_mask == 0)  # False Positive
    fn = (pred_mask == 0) & (gt_mask == 1)  # False Negative
    
    alpha = 0.5
    
    # 綠色 = 正確
    result[tp] = result[tp] * (1 - alpha) + np.array([0, 255, 0]) * alpha
    # 紅色 = 誤判（把非天空當成天空）
    result[fp] = result[fp] * (1 - alpha) + np.array([255, 0, 0]) * alpha
    # 藍色 = 漏掉（沒抓到的天空）
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
    
    # Pixel Accuracy
    pixel_acc = (pred == gt).sum() / pred.size
    
    # IoU
    intersection = ((pred == 1) & (gt == 1)).sum()
    union = ((pred == 1) | (gt == 1)).sum()
    iou = (intersection + smooth) / (union + smooth)
    
    # Dice
    pred_area = (pred == 1).sum()
    gt_area = (gt == 1).sum()
    dice = (2 * intersection + smooth) / (pred_area + gt_area + smooth)
    
    return pixel_acc, iou, dice


def main():
    # === 設定 ===
    images_dir = 'data/toy_hard/images'
    masks_dir = 'data/toy_hard/masks'
    info_csv = 'data/toy_hard/dataset_info.csv'
    output_vis_dir = 'outputs/cv_toy_hard_vis'
    output_metrics_csv = 'outputs/cv_toy_hard_metrics.csv'
    
    os.makedirs(output_vis_dir, exist_ok=True)
    
    print('=' * 60)
    print('  CV Baseline on Toy HARD Dataset')
    print('=' * 60)
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
    
    # === 評估並產生視覺化 ===
    print('[2] Evaluating and generating visualizations...')
    print('-' * 60)
    
    results = []
    
    for idx in range(num_samples):
        sample_id = f'{idx+1:03d}'
        
        image_tensor = images[idx]
        gt_tensor = gt_masks[idx]
        
        image_np = tensor_to_numpy_uint8(image_tensor)
        gt_mask = gt_tensor_to_numpy(gt_tensor)
        pred_mask = predict_sky_mask_cv(image_tensor)
        
        # 計算指標
        pixel_acc, iou, dice = compute_metrics(pred_mask, gt_mask)
        
        # 取得 dataset info
        info = dataset_info.get(sample_id, {})
        sky_name = info.get('sky_name', 'Unknown')
        ground_name = info.get('ground_name', 'Unknown')
        
        results.append({
            'sample_id': sample_id,
            'sky_type': info.get('sky_type', ''),
            'sky_name': sky_name,
            'ground_type': info.get('ground_type', ''),
            'ground_name': ground_name,
            'pixel_accuracy': pixel_acc,
            'iou': iou,
            'dice': dice
        })
        
        # === 儲存視覺化 ===
        # 1. 原始圖片
        Image.fromarray(image_np).save(
            os.path.join(output_vis_dir, f'{sample_id}_1_image.png')
        )
        
        # 2. GT mask
        Image.fromarray((gt_mask * 255).astype(np.uint8), mode='L').save(
            os.path.join(output_vis_dir, f'{sample_id}_2_gt_mask.png')
        )
        
        # 3. Pred mask
        Image.fromarray((pred_mask * 255).astype(np.uint8), mode='L').save(
            os.path.join(output_vis_dir, f'{sample_id}_3_pred_mask.png')
        )
        
        # 4. 比較 overlay
        comparison = create_comparison_overlay(image_np, pred_mask, gt_mask)
        Image.fromarray(comparison).save(
            os.path.join(output_vis_dir, f'{sample_id}_4_comparison.png')
        )
        
        # 顯示進度
        status = 'OK' if iou > 0.7 else 'FAIL' if iou < 0.3 else 'WARN'
        print(f'    {sample_id}: {sky_name:12} | IoU={iou*100:5.1f}% [{status}]')
    
    print('-' * 60)
    print()
    
    # === 儲存 metrics CSV ===
    print('[3] Saving metrics...')
    with open(output_metrics_csv, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['sample_id', 'sky_type', 'sky_name', 'ground_type', 'ground_name', 
                      'pixel_accuracy', 'iou', 'dice']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                'sample_id': r['sample_id'],
                'sky_type': r['sky_type'],
                'sky_name': r['sky_name'],
                'ground_type': r['ground_type'],
                'ground_name': r['ground_name'],
                'pixel_accuracy': f"{r['pixel_accuracy']:.4f}",
                'iou': f"{r['iou']:.4f}",
                'dice': f"{r['dice']:.4f}"
            })
    
    print(f'    Saved to: {output_metrics_csv}')
    print()
    
    # === 統計結果 ===
    avg_iou = np.mean([r['iou'] for r in results])
    avg_dice = np.mean([r['dice'] for r in results])
    avg_acc = np.mean([r['pixel_accuracy'] for r in results])
    
    fail_count = sum(1 for r in results if r['iou'] < 0.3)
    warn_count = sum(1 for r in results if 0.3 <= r['iou'] < 0.7)
    ok_count = sum(1 for r in results if r['iou'] >= 0.7)
    
    print('=' * 60)
    print('  RESULTS SUMMARY')
    print('=' * 60)
    print()
    print(f'  Total samples: {num_samples}')
    print()
    print('  +------------------+----------+')
    print('  | Metric           | Value    |')
    print('  +------------------+----------+')
    print(f'  | Pixel Accuracy   | {avg_acc*100:6.2f}%  |')
    print(f'  | IoU              | {avg_iou*100:6.2f}%  |')
    print(f'  | Dice             | {avg_dice*100:6.2f}%  |')
    print('  +------------------+----------+')
    print()
    print(f'  Status breakdown:')
    print(f'    OK   (IoU >= 70%): {ok_count}')
    print(f'    WARN (30-70%):     {warn_count}')
    print(f'    FAIL (IoU < 30%):  {fail_count}')
    print()
    
    # === 按天空類型分析 ===
    print('=' * 60)
    print('  BREAKDOWN BY SKY TYPE')
    print('=' * 60)
    print()
    
    from collections import defaultdict
    by_sky = defaultdict(list)
    for r in results:
        by_sky[r['sky_name']].append(r['iou'])
    
    print('  +----------------------+-------+--------+')
    print('  | Sky Type             | Count | Avg IoU|')
    print('  +----------------------+-------+--------+')
    for sky_name, ious in sorted(by_sky.items(), key=lambda x: np.mean(x[1])):
        avg = np.mean(ious) * 100
        status = 'FAIL' if avg < 30 else 'WARN' if avg < 70 else ''
        print(f'  | {sky_name:20} | {len(ious):5} | {avg:5.1f}% | {status}')
    print('  +----------------------+-------+--------+')
    print()
    
    # === Overlay 說明 ===
    print('=' * 60)
    print('  COMPARISON OVERLAY LEGEND')
    print('=' * 60)
    print()
    print('  In XXX_4_comparison.png:')
    print('    GREEN = Correct prediction (True Positive)')
    print('    RED   = Wrong prediction (False Positive, predicted sky but is ground)')
    print('    BLUE  = Missed sky (False Negative, is sky but not predicted)')
    print()
    print('=' * 60)


if __name__ == '__main__':
    main()
