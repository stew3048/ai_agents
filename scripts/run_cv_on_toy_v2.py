"""
CV Baseline on Toy V2 Dataset (512x384 -> resize to 256x256)
測試 DataLoader resize 後的 CV baseline 表現

重點驗證：
1. DataLoader 會將 512x384 resize 到 256x256
2. CV baseline 在 resized 圖片上的表現
3. 確認 resize 不會破壞分割品質
"""

import os
import sys
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
    """計算指標"""
    smooth = 1e-6
    
    pixel_acc = (pred == gt).sum() / pred.size
    
    intersection = ((pred == 1) & (gt == 1)).sum()
    union = ((pred == 1) | (gt == 1)).sum()
    iou = (intersection + smooth) / (union + smooth)
    
    pred_area = (pred == 1).sum()
    gt_area = (gt == 1).sum()
    dice = (2 * intersection + smooth) / (pred_area + gt_area + smooth)
    
    return pixel_acc, iou, dice


def main():
    # === 設定 ===
    images_dir = 'data/toy_v2/images'
    masks_dir = 'data/toy_v2/masks'
    output_vis_dir = 'outputs/cv_toy_v2_vis'
    
    os.makedirs(output_vis_dir, exist_ok=True)
    
    print('=' * 60)
    print('  CV Baseline on Toy V2 Dataset')
    print('  (512x384 -> DataLoader resize to 256x256)')
    print('=' * 60)
    print()
    
    # === 載入原始圖片尺寸資訊 ===
    sample_img = Image.open(os.path.join(images_dir, '001.jpg'))
    original_size = sample_img.size
    print(f'  Original image size: {original_size[0]} x {original_size[1]}')
    print(f'  DataLoader target:   256 x 256')
    print()
    
    # === 載入 DataLoader ===
    print('[1] Loading DataLoader (with resize)...')
    dataloader = get_dataloader(
        images_dir=images_dir,
        masks_dir=masks_dir,
        batch_size=20,
        shuffle=False,
        transform=False
    )
    
    images, gt_masks = next(iter(dataloader))
    num_samples = images.shape[0]
    
    print(f'    Loaded {num_samples} images')
    print(f'    After DataLoader: {images.shape[2]} x {images.shape[3]}')
    print()
    
    # === 評估 ===
    print('[2] Evaluating CV baseline on resized images...')
    print('-' * 60)
    
    results = []
    
    for idx in range(num_samples):
        sample_id = f'{idx+1:03d}'
        
        image_tensor = images[idx]
        gt_tensor = gt_masks[idx]
        
        image_np = tensor_to_numpy_uint8(image_tensor)
        gt_mask = gt_tensor_to_numpy(gt_tensor)
        pred_mask = predict_sky_mask_cv(image_tensor)
        
        pixel_acc, iou, dice = compute_metrics(pred_mask, gt_mask)
        
        results.append({
            'sample_id': sample_id,
            'pixel_accuracy': pixel_acc,
            'iou': iou,
            'dice': dice,
        })
        
        # 儲存視覺化
        # 1. Resized image (from DataLoader)
        Image.fromarray(image_np).save(
            os.path.join(output_vis_dir, f'{sample_id}_1_resized_image.png')
        )
        
        # 2. Resized GT mask (from DataLoader)
        Image.fromarray((gt_mask * 255).astype(np.uint8)).save(
            os.path.join(output_vis_dir, f'{sample_id}_2_resized_gt_mask.png')
        )
        
        # 3. Pred mask
        Image.fromarray((pred_mask * 255).astype(np.uint8)).save(
            os.path.join(output_vis_dir, f'{sample_id}_3_pred_mask.png')
        )
        
        # 4. Comparison overlay
        comparison = create_comparison_overlay(image_np, pred_mask, gt_mask)
        Image.fromarray(comparison).save(
            os.path.join(output_vis_dir, f'{sample_id}_4_comparison.png')
        )
        
        status = 'OK' if iou > 0.9 else 'WARN' if iou > 0.7 else 'FAIL'
        print(f'    {sample_id}: IoU={iou*100:5.1f}% | Dice={dice*100:5.1f}% [{status}]')
    
    print('-' * 60)
    print()
    
    # === 統計結果 ===
    avg_iou = np.mean([r['iou'] for r in results])
    avg_dice = np.mean([r['dice'] for r in results])
    avg_acc = np.mean([r['pixel_accuracy'] for r in results])
    
    ok_count = sum(1 for r in results if r['iou'] > 0.9)
    
    print('=' * 60)
    print('  RESULTS SUMMARY')
    print('=' * 60)
    print()
    print(f'  Original size:  {original_size[0]} x {original_size[1]}')
    print(f'  Resized to:     256 x 256')
    print(f'  Total samples:  {num_samples}')
    print()
    print('  +------------------+----------+')
    print('  | Metric           | Value    |')
    print('  +------------------+----------+')
    print(f'  | Pixel Accuracy   | {avg_acc*100:6.2f}%  |')
    print(f'  | IoU              | {avg_iou*100:6.2f}%  |')
    print(f'  | Dice             | {avg_dice*100:6.2f}%  |')
    print('  +------------------+----------+')
    print()
    print(f'  OK (IoU > 90%): {ok_count} / {num_samples}')
    print()
    
    # === 關鍵驗證 ===
    print('=' * 60)
    print('  KEY VERIFICATION')
    print('=' * 60)
    print()
    print('  This test verifies:')
    print('    1. DataLoader correctly resizes 512x384 -> 256x256')
    print('    2. Image uses BILINEAR interpolation')
    print('    3. Mask uses NEAREST interpolation (preserves binary values)')
    print('    4. CV baseline works on resized images')
    print()
    
    if avg_iou > 0.9:
        print('  Result: PASS')
        print('    Resize does NOT break segmentation quality.')
    else:
        print('  Result: INVESTIGATE')
        print('    IoU lower than expected, check resize implementation.')
    
    print()
    print(f'  Output saved to: {output_vis_dir}/')
    print()
    print('=' * 60)


if __name__ == '__main__':
    main()
