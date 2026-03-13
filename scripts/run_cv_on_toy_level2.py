"""
CV Baseline on Toy Level 2 Dataset
測試 CV baseline 對地面藍色物體的誤判

輸出：
- outputs/cv_toy_level2_vis/: 視覺化結果
- outputs/cv_toy_level2_metrics.csv: 指標
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
    """
    建立比較 overlay：
    - 綠色：正確預測為天空 (TP)
    - 紅色：錯誤預測為天空 (FP) - 這是我們要觀察的重點！
    - 藍色：漏掉的天空 (FN)
    """
    result = image_np.astype(np.float32).copy()
    
    tp = (pred_mask == 1) & (gt_mask == 1)
    fp = (pred_mask == 1) & (gt_mask == 0)  # 把地面的藍色物體當成天空
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
    
    # 額外計算 False Positive Rate（誤判率）
    fp = ((pred == 1) & (gt == 0)).sum()  # 把地面當成天空
    tn = ((pred == 0) & (gt == 0)).sum()  # 正確識別為地面
    fpr = fp / (fp + tn + smooth)  # False Positive Rate
    
    return pixel_acc, iou, dice, fpr, fp


def main():
    # === 設定 ===
    images_dir = 'data/toy_level2/images'
    masks_dir = 'data/toy_level2/masks'
    info_csv = 'data/toy_level2/dataset_info.csv'
    output_vis_dir = 'outputs/cv_toy_level2_vis'
    output_metrics_csv = 'outputs/cv_toy_level2_metrics.csv'
    
    os.makedirs(output_vis_dir, exist_ok=True)
    
    print('=' * 60)
    print('  CV Baseline on Toy LEVEL 2: Blue Objects on Ground')
    print('=' * 60)
    print()
    print('  This test checks if CV baseline incorrectly')
    print('  identifies blue objects on ground as "sky".')
    print()
    print('  Expected: HIGH False Positive Rate (FPR)')
    print('            because blue objects will be mistaken for sky.')
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
        
        pixel_acc, iou, dice, fpr, fp_pixels = compute_metrics(pred_mask, gt_mask)
        
        info = dataset_info.get(sample_id, {})
        num_objects = info.get('num_blue_objects', '?')
        obj_types = info.get('blue_object_types', '')
        
        results.append({
            'sample_id': sample_id,
            'num_blue_objects': num_objects,
            'blue_object_types': obj_types,
            'pixel_accuracy': pixel_acc,
            'iou': iou,
            'dice': dice,
            'false_positive_rate': fpr,
            'false_positive_pixels': fp_pixels,
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
        if fpr > 0.1:
            status = 'FP_HIGH'  # 高誤判率
        elif fpr > 0.02:
            status = 'FP_MED'
        else:
            status = 'OK'
        
        print(f'    {sample_id}: FPR={fpr*100:5.1f}% | IoU={iou*100:5.1f}% [{status}]')
    
    print('-' * 60)
    print()
    
    # === 儲存 CSV ===
    print('[3] Saving metrics...')
    with open(output_metrics_csv, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['sample_id', 'num_blue_objects', 'blue_object_types',
                      'pixel_accuracy', 'iou', 'dice', 'false_positive_rate', 'false_positive_pixels']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                'sample_id': r['sample_id'],
                'num_blue_objects': r['num_blue_objects'],
                'blue_object_types': r['blue_object_types'],
                'pixel_accuracy': f"{r['pixel_accuracy']:.4f}",
                'iou': f"{r['iou']:.4f}",
                'dice': f"{r['dice']:.4f}",
                'false_positive_rate': f"{r['false_positive_rate']:.4f}",
                'false_positive_pixels': r['false_positive_pixels'],
            })
    
    print(f'    Saved to: {output_metrics_csv}')
    print()
    
    # === 統計結果 ===
    avg_iou = np.mean([r['iou'] for r in results])
    avg_fpr = np.mean([r['false_positive_rate'] for r in results])
    avg_fp_pixels = np.mean([r['false_positive_pixels'] for r in results])
    avg_acc = np.mean([r['pixel_accuracy'] for r in results])
    
    high_fpr_count = sum(1 for r in results if r['false_positive_rate'] > 0.1)
    med_fpr_count = sum(1 for r in results if 0.02 < r['false_positive_rate'] <= 0.1)
    
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
    print(f'  | False Positive Rate    | {avg_fpr*100:6.2f}%  |')
    print(f'  | Avg FP Pixels/Image    | {avg_fp_pixels:6.0f}   |')
    print('  +------------------------+----------+')
    print()
    print('  False Positive breakdown:')
    print(f'    HIGH FPR (>10%): {high_fpr_count}')
    print(f'    MED FPR (2-10%): {med_fpr_count}')
    print(f'    LOW FPR (<2%):   {num_samples - high_fpr_count - med_fpr_count}')
    print()
    
    # === 重要發現 ===
    print('=' * 60)
    print('  KEY FINDINGS')
    print('=' * 60)
    print()
    print('  In Level 2, the sky is NORMAL BLUE, so CV baseline')
    print('  should detect it correctly.')
    print()
    print('  BUT: Blue objects on the ground are ALSO detected')
    print('  as "sky" (False Positives).')
    print()
    print('  This shows the fundamental flaw of color-based detection:')
    print('  "Blue color" != "Sky"')
    print()
    
    # === Overlay 說明 ===
    print('=' * 60)
    print('  COMPARISON OVERLAY LEGEND')
    print('=' * 60)
    print()
    print('  In XXX_4_comparison.png:')
    print('    GREEN = Correct sky prediction (TP)')
    print('    RED   = Ground mistaken as sky (FP) <-- KEY ERROR!')
    print('    BLUE  = Missed sky (FN)')
    print()
    print('  Look for RED areas on blue objects (buildings, pools, cars)')
    print('  These are the False Positives caused by color matching.')
    print()
    print('=' * 60)


if __name__ == '__main__':
    main()
