"""
跨 Camera 泛化測試（Domain Shift Evaluation）

用 Camera 10066 訓練的 U-Net，在 Camera 10870 上評估。
這可以測試模型的泛化能力：是否學到了「天空的通用特徵」，
還是只是過度擬合了 10066 的特定場景。

輸出：
- outputs/unet_cross_camera_metrics.csv：每張圖的指標
- outputs/unet_cross_camera_vis/：隨機抽樣 20 張的 overlay 視覺化
"""

import os
import sys
import csv
import random
import torch
import numpy as np
from PIL import Image

# 加入專案路徑
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.unet import UNetSmall


# ============================================================
# 設定
# ============================================================
CONFIG = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    
    # 訓練來源（10066）
    'source_camera': '10066',
    'model_path': 'outputs/unet_10066_best.pt',
    
    # 測試目標（10870）
    'target_camera': '10870',
    'target_images_dir': 'data/skyfinder_10870/images',
    'target_masks_dir': 'data/skyfinder_10870/masks',
    
    # 輸出
    'output_csv': 'outputs/unet_cross_camera_metrics.csv',
    'output_vis_dir': 'outputs/unet_cross_camera_vis',
    
    # 設定
    'image_size': (256, 256),
    'num_vis_samples': 20,  # 隨機抽樣視覺化的數量
}

# Overlay 顏色設定
OVERLAY_ALPHA = 0.5
COLOR_TP = np.array([0, 255, 0])    # 綠色：正確預測天空
COLOR_FP = np.array([255, 0, 0])    # 紅色：誤判為天空
COLOR_FN = np.array([0, 0, 255])    # 藍色：漏掉的天空


# ============================================================
# Helper functions
# ============================================================
def create_overlay(image_np, pred_mask, gt_mask):
    """建立 overlay"""
    result = image_np.astype(np.float32).copy()
    
    tp = (pred_mask == 1) & (gt_mask == 1)
    fp = (pred_mask == 1) & (gt_mask == 0)
    fn = (pred_mask == 0) & (gt_mask == 1)
    
    result[tp] = result[tp] * (1 - OVERLAY_ALPHA) + COLOR_TP * OVERLAY_ALPHA
    result[fp] = result[fp] * (1 - OVERLAY_ALPHA) + COLOR_FP * OVERLAY_ALPHA
    result[fn] = result[fn] * (1 - OVERLAY_ALPHA) + COLOR_FN * OVERLAY_ALPHA
    
    return result.clip(0, 255).astype(np.uint8)


def compute_metrics(pred_mask, gt_mask):
    """計算完整的評估指標"""
    smooth = 1e-6
    
    pred = pred_mask.flatten().astype(bool)
    gt = gt_mask.flatten().astype(bool)
    
    tp = (pred & gt).sum()
    fp = (pred & ~gt).sum()
    fn = (~pred & gt).sum()
    tn = (~pred & ~gt).sum()
    
    total = pred.size
    total_positive = gt.sum()
    total_negative = (~gt).sum()
    
    intersection = tp
    union = tp + fp + fn
    iou = (intersection + smooth) / (union + smooth)
    
    dice = (2 * tp + smooth) / (2 * tp + fp + fn + smooth)
    pixel_acc = (tp + tn) / total
    fp_rate = fp / (total_negative + smooth)
    fn_rate = fn / (total_positive + smooth)
    
    return {
        'iou': float(iou),
        'dice': float(dice),
        'pixel_acc': float(pixel_acc),
        'fp_rate': float(fp_rate),
        'fn_rate': float(fn_rate),
    }


def load_image_and_mask(image_path, mask_path, size):
    """載入並預處理圖片和 mask"""
    # 允許讀取截斷的圖片
    from PIL import ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    
    try:
        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        
        image_resized = image.resize(size, Image.BILINEAR)
        mask_resized = mask.resize(size, Image.NEAREST)
        
        image_np = np.array(image_resized)
        mask_np = np.array(mask_resized)
        mask_np = (mask_np > 127).astype(np.uint8)
        
        return image_np, mask_np
    except Exception as e:
        print(f"    Warning: Failed to load {image_path}: {e}")
        return None, None


def predict_unet(model, image_np, device):
    """用 U-Net 預測"""
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
    image_tensor = image_tensor.unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = model(image_tensor)
        probs = torch.sigmoid(logits)
        pred = (probs > 0.5).float()
    
    return pred.squeeze().cpu().numpy().astype(np.uint8)


def get_all_image_files(images_dir):
    """取得所有圖片檔案"""
    files = []
    for f in sorted(os.listdir(images_dir)):
        if f.endswith(('.jpg', '.png', '.jpeg')):
            files.append(f)
    return files


# ============================================================
# 主程式
# ============================================================
def main():
    print('=' * 70)
    print('  Cross-Camera Generalization Test (Domain Shift)')
    print('=' * 70)
    print()
    print(f'  Source Camera (trained on): {CONFIG["source_camera"]}')
    print(f'  Target Camera (test on):    {CONFIG["target_camera"]}')
    print()
    print('  This test measures how well the model generalizes to')
    print('  a completely different scene that it has never seen.')
    print()
    
    device = CONFIG['device']
    print(f'  Device: {device}')
    print()
    
    # 建立輸出目錄
    os.makedirs(CONFIG['output_vis_dir'], exist_ok=True)
    os.makedirs(os.path.dirname(CONFIG['output_csv']), exist_ok=True)
    
    # ========================================
    # 1. 載入模型
    # ========================================
    print('[1] Loading model trained on camera 10066...')
    checkpoint = torch.load(CONFIG['model_path'], map_location=device, weights_only=False)
    
    model = UNetSmall(in_channels=3, out_channels=1).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f'    Model: {CONFIG["model_path"]}')
    print(f'    Trained on: Camera {CONFIG["source_camera"]}')
    print(f'    Best epoch: {checkpoint["epoch"]}, Val IoU: {checkpoint["val_iou"]*100:.2f}%')
    print()
    
    # ========================================
    # 2. 載入目標 Camera 資料
    # ========================================
    print(f'[2] Loading target camera {CONFIG["target_camera"]} data...')
    image_files = get_all_image_files(CONFIG['target_images_dir'])
    print(f'    Found {len(image_files)} images')
    print()
    
    # ========================================
    # 3. 評估
    # ========================================
    print('[3] Evaluating on target camera...')
    print('-' * 70)
    
    results = []
    
    skipped = 0
    for i, filename in enumerate(image_files):
        image_path = os.path.join(CONFIG['target_images_dir'], filename)
        mask_name = filename.replace('.jpg', '.png').replace('.jpeg', '.png')
        mask_path = os.path.join(CONFIG['target_masks_dir'], mask_name)
        
        # 載入
        image_np, gt_np = load_image_and_mask(image_path, mask_path, CONFIG['image_size'])
        
        # 跳過損壞的檔案
        if image_np is None:
            skipped += 1
            continue
        
        # 預測
        pred_np = predict_unet(model, image_np, device)
        
        # 計算 metrics
        metrics = compute_metrics(pred_np, gt_np)
        metrics['filename'] = filename
        metrics['image_np'] = image_np
        metrics['pred_np'] = pred_np
        metrics['gt_np'] = gt_np
        results.append(metrics)
        
        # 進度
        if (i + 1) % 20 == 0 or i == len(image_files) - 1:
            print(f'    Processed {i+1}/{len(image_files)} images (skipped {skipped})...')
    
    print('-' * 70)
    print()
    
    # ========================================
    # 4. 統計
    # ========================================
    ious = [r['iou'] for r in results]
    dices = [r['dice'] for r in results]
    pixel_accs = [r['pixel_acc'] for r in results]
    fp_rates = [r['fp_rate'] for r in results]
    fn_rates = [r['fn_rate'] for r in results]
    
    stats = {
        'iou': {'mean': np.mean(ious), 'std': np.std(ious)},
        'dice': {'mean': np.mean(dices), 'std': np.std(dices)},
        'pixel_acc': {'mean': np.mean(pixel_accs), 'std': np.std(pixel_accs)},
        'fp_rate': {'mean': np.mean(fp_rates), 'std': np.std(fp_rates)},
        'fn_rate': {'mean': np.mean(fn_rates), 'std': np.std(fn_rates)},
    }
    
    # ========================================
    # 5. 隨機抽樣視覺化
    # ========================================
    print(f'[4] Saving {CONFIG["num_vis_samples"]} random visualizations...')
    
    # 隨機抽樣
    random.seed(42)  # 固定 seed 以便重現
    vis_indices = random.sample(range(len(results)), min(CONFIG['num_vis_samples'], len(results)))
    
    for idx in vis_indices:
        r = results[idx]
        filename = r['filename']
        basename = filename.replace('.jpg', '').replace('.png', '')
        
        # 建立 overlay
        overlay = create_overlay(r['image_np'], r['pred_np'], r['gt_np'])
        
        # 儲存
        Image.fromarray(r['image_np']).save(
            os.path.join(CONFIG['output_vis_dir'], f'{basename}_1_image.png'))
        Image.fromarray((r['gt_np'] * 255).astype(np.uint8)).save(
            os.path.join(CONFIG['output_vis_dir'], f'{basename}_2_gt.png'))
        Image.fromarray((r['pred_np'] * 255).astype(np.uint8)).save(
            os.path.join(CONFIG['output_vis_dir'], f'{basename}_3_pred.png'))
        Image.fromarray(overlay).save(
            os.path.join(CONFIG['output_vis_dir'], f'{basename}_4_overlay.png'))
    
    print(f'    Saved to: {CONFIG["output_vis_dir"]}/')
    print()
    
    # ========================================
    # 6. 儲存 CSV
    # ========================================
    print('[5] Saving metrics to CSV...')
    
    with open(CONFIG['output_csv'], 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['filename', 'iou', 'dice', 'pixel_acc', 'fp_rate', 'fn_rate']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for r in results:
            writer.writerow({
                'filename': r['filename'],
                'iou': f"{r['iou']:.4f}",
                'dice': f"{r['dice']:.4f}",
                'pixel_acc': f"{r['pixel_acc']:.4f}",
                'fp_rate': f"{r['fp_rate']:.4f}",
                'fn_rate': f"{r['fn_rate']:.4f}",
            })
    
    print(f'    Saved to: {CONFIG["output_csv"]}')
    print()
    
    # ========================================
    # 7. 最終摘要
    # ========================================
    print('=' * 70)
    print('  CROSS-CAMERA GENERALIZATION RESULTS')
    print('=' * 70)
    print()
    print(f'  Source: Camera {CONFIG["source_camera"]} (trained on)')
    print(f'  Target: Camera {CONFIG["target_camera"]} (tested on)')
    print(f'  Samples: {len(results)}')
    print()
    print('  Metrics (Mean +/- Std):')
    print(f'    IoU:        {stats["iou"]["mean"]*100:>6.2f}% +/- {stats["iou"]["std"]*100:.2f}%')
    print(f'    Dice:       {stats["dice"]["mean"]*100:>6.2f}% +/- {stats["dice"]["std"]*100:.2f}%')
    print(f'    Pixel Acc:  {stats["pixel_acc"]["mean"]*100:>6.2f}% +/- {stats["pixel_acc"]["std"]*100:.2f}%')
    print(f'    FP Rate:    {stats["fp_rate"]["mean"]*100:>6.2f}% +/- {stats["fp_rate"]["std"]*100:.2f}%')
    print(f'    FN Rate:    {stats["fn_rate"]["mean"]*100:>6.2f}% +/- {stats["fn_rate"]["std"]*100:.2f}%')
    print()
    
    # 與原本的 val/test 比較
    source_val_iou = checkpoint['val_iou']
    cross_iou = stats['iou']['mean']
    drop = (source_val_iou - cross_iou) * 100
    
    print('  Comparison with source camera:')
    print(f'    Source Val IoU (10066):  {source_val_iou*100:.2f}%')
    print(f'    Cross-Camera IoU (10870): {cross_iou*100:.2f}%')
    print(f'    Performance Drop:         {drop:+.2f}%')
    print()
    
    # 評估泛化能力
    if cross_iou >= 0.90:
        generalization = "Excellent - Model generalizes very well to new scenes!"
    elif cross_iou >= 0.80:
        generalization = "Good - Model has learned general sky features."
    elif cross_iou >= 0.70:
        generalization = "Moderate - Some domain shift impact, but still usable."
    elif cross_iou >= 0.50:
        generalization = "Poor - Significant domain shift impact."
    else:
        generalization = "Very Poor - Model is overfitted to source camera."
    
    print(f'  Generalization Assessment: {generalization}')
    print()
    print('  Output files:')
    print(f'    - Metrics CSV: {CONFIG["output_csv"]}')
    print(f'    - Visualizations: {CONFIG["output_vis_dir"]}/')
    print()
    print('=' * 70)


if __name__ == '__main__':
    main()
