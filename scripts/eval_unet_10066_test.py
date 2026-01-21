"""
U-Net Test Set 評估腳本 - SkyFinder Camera 10066

重要：Test set 只跑一次，不能用來調參！
- Test set 用於最終評估模型的泛化能力
- 如果拿 test set 來調參，會造成資料洩漏，導致過度樂觀的評估

這個腳本會：
1. 載入 best checkpoint（val IoU 最高的模型）
2. 在 test split 上評估一次
3. 輸出每張圖的 metrics 到 CSV
4. 在終端印出摘要（平均與標準差）
"""

import os
import sys
import csv
import torch
import numpy as np
from PIL import Image

# 加入專案路徑
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.unet import UNetSmall
from src.datasets import get_skyfinder_dataloader


# ============================================================
# 設定
# ============================================================
CONFIG = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'splits_json': 'outputs/splits_10066.json',
    'best_model_path': 'outputs/unet_10066_best.pt',
    'output_csv': 'outputs/unet_10066_test_metrics.csv',
    'output_vis_dir': 'outputs/unet_10066_test_vis',
    'batch_size': 1,  # 逐張評估，方便記錄每張的 metrics
}


# ============================================================
# Metrics 計算（含 FP rate / FN rate）
# ============================================================
def compute_metrics(pred_mask, gt_mask):
    """
    計算完整的評估指標
    
    Args:
        pred_mask: 預測 mask (numpy array, 0/1)
        gt_mask: Ground truth mask (numpy array, 0/1)
    
    Returns:
        dict with iou, dice, pixel_acc, fp_rate, fn_rate
    """
    smooth = 1e-6
    
    pred = pred_mask.flatten().astype(bool)
    gt = gt_mask.flatten().astype(bool)
    
    # True Positive, False Positive, False Negative, True Negative
    tp = (pred & gt).sum()
    fp = (pred & ~gt).sum()
    fn = (~pred & gt).sum()
    tn = (~pred & ~gt).sum()
    
    # Total pixels
    total = pred.size
    total_positive = gt.sum()  # 實際為天空的像素
    total_negative = (~gt).sum()  # 實際為非天空的像素
    
    # IoU (Intersection over Union)
    intersection = tp
    union = tp + fp + fn
    iou = (intersection + smooth) / (union + smooth)
    
    # Dice Coefficient
    dice = (2 * tp + smooth) / (2 * tp + fp + fn + smooth)
    
    # Pixel Accuracy
    pixel_acc = (tp + tn) / total
    
    # FP Rate (False Positive Rate) = FP / (FP + TN) = FP / total_negative
    # 把非天空誤判為天空的比例
    fp_rate = fp / (total_negative + smooth)
    
    # FN Rate (False Negative Rate) = FN / (FN + TP) = FN / total_positive
    # 把天空漏掉的比例
    fn_rate = fn / (total_positive + smooth)
    
    return {
        'iou': float(iou),
        'dice': float(dice),
        'pixel_acc': float(pixel_acc),
        'fp_rate': float(fp_rate),
        'fn_rate': float(fn_rate),
        'tp': int(tp),
        'fp': int(fp),
        'fn': int(fn),
        'tn': int(tn),
    }


# ============================================================
# 視覺化（重用 train_unet_10066.py 的邏輯）
# ============================================================
def create_pred_overlay(image_np, pred_mask, gt_mask):
    """
    建立預測 overlay
    - 綠色：TP（正確預測天空）
    - 紅色：FP（錯誤預測為天空）
    - 藍色：FN（漏掉的天空）
    """
    result = image_np.astype(np.float32).copy()
    
    tp = (pred_mask == 1) & (gt_mask == 1)
    fp = (pred_mask == 1) & (gt_mask == 0)
    fn = (pred_mask == 0) & (gt_mask == 1)
    
    alpha = 0.5
    result[tp] = result[tp] * (1 - alpha) + np.array([0, 255, 0]) * alpha
    result[fp] = result[fp] * (1 - alpha) + np.array([255, 0, 0]) * alpha
    result[fn] = result[fn] * (1 - alpha) + np.array([0, 0, 255]) * alpha
    
    return result.clip(0, 255).astype(np.uint8)


# ============================================================
# 評估函數
# ============================================================
def evaluate_test_set(model, dataloader, device, output_vis_dir):
    """
    在 test set 上評估模型，並儲存視覺化結果
    
    Returns:
        list of dicts: 每張圖的 metrics
    """
    model.eval()
    results = []
    
    dataset = dataloader.dataset
    sample_idx = 0
    
    # 建立輸出目錄
    os.makedirs(output_vis_dir, exist_ok=True)
    
    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward
            logits = model(images)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            
            # 計算每個樣本的 metrics 並儲存視覺化
            for i in range(images.size(0)):
                # 轉換為 numpy
                img_np = (images[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                pred_np = preds[i].squeeze().cpu().numpy().astype(np.uint8)
                gt_np = masks[i].squeeze().cpu().numpy().astype(np.uint8)
                
                # 計算 metrics
                metrics = compute_metrics(pred_np, gt_np)
                filename = dataset.get_filename(sample_idx)
                metrics['filename'] = filename
                results.append(metrics)
                
                # 儲存視覺化
                basename = filename.replace('.jpg', '').replace('.png', '')
                
                # 1. 原圖
                Image.fromarray(img_np).save(
                    os.path.join(output_vis_dir, f'{basename}_1_image.png'))
                
                # 2. GT mask
                Image.fromarray((gt_np * 255).astype(np.uint8)).save(
                    os.path.join(output_vis_dir, f'{basename}_2_gt.png'))
                
                # 3. Pred mask
                Image.fromarray((pred_np * 255).astype(np.uint8)).save(
                    os.path.join(output_vis_dir, f'{basename}_3_pred.png'))
                
                # 4. Overlay
                overlay = create_pred_overlay(img_np, pred_np, gt_np)
                Image.fromarray(overlay).save(
                    os.path.join(output_vis_dir, f'{basename}_4_overlay.png'))
                
                sample_idx += 1
    
    return results


# ============================================================
# 主程式
# ============================================================
def main():
    print('=' * 60)
    print('  U-Net Test Set Evaluation - SkyFinder 10066')
    print('=' * 60)
    print()
    print('  [WARNING] Test set should only be evaluated ONCE!')
    print('      Do NOT use test results to tune hyperparameters.')
    print()
    
    device = CONFIG['device']
    print(f'  Device: {device}')
    print()
    
    # 檢查 checkpoint 是否存在
    if not os.path.exists(CONFIG['best_model_path']):
        print(f"  [ERROR] Best model not found at {CONFIG['best_model_path']}")
        print("     Please run train_unet_10066.py first.")
        return
    
    # ========================================
    # 1. 載入模型
    # ========================================
    print('[1] Loading best checkpoint...')
    checkpoint = torch.load(CONFIG['best_model_path'], map_location=device, weights_only=False)
    
    model = UNetSmall(in_channels=3, out_channels=1).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f'    Checkpoint epoch: {checkpoint["epoch"]}')
    print(f'    Val IoU at checkpoint: {checkpoint["val_iou"]*100:.2f}%')
    print(f'    Val Dice at checkpoint: {checkpoint["val_dice"]*100:.2f}%')
    print()
    
    # ========================================
    # 2. 載入 Test DataLoader
    # ========================================
    print('[2] Loading test data...')
    test_loader = get_skyfinder_dataloader(
        splits_json=CONFIG['splits_json'],
        split='test',
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        transform=False  # Test 不做 augmentation
    )
    
    test_dataset = test_loader.dataset
    print(f'    Test samples: {len(test_dataset)}')
    print(f'    Test files: {[test_dataset.get_filename(i) for i in range(len(test_dataset))]}')
    print()
    
    # ========================================
    # 3. 評估並儲存視覺化
    # ========================================
    print('[3] Evaluating on test set (ONE TIME ONLY)...')
    results = evaluate_test_set(model, test_loader, device, CONFIG['output_vis_dir'])
    print(f'    Evaluated {len(results)} samples.')
    print(f'    Visualizations saved to: {CONFIG["output_vis_dir"]}/')
    print()
    
    # ========================================
    # 4. 輸出每張的 metrics
    # ========================================
    print('[4] Per-image results:')
    print('-' * 90)
    print(f'  {"Filename":<12} | {"IoU":>8} | {"Dice":>8} | {"PixelAcc":>8} | {"FP Rate":>8} | {"FN Rate":>8}')
    print('-' * 90)
    
    for r in results:
        print(f'  {r["filename"]:<12} | {r["iou"]*100:>7.2f}% | {r["dice"]*100:>7.2f}% | {r["pixel_acc"]*100:>7.2f}% | {r["fp_rate"]*100:>7.2f}% | {r["fn_rate"]*100:>7.2f}%')
    
    print('-' * 90)
    print()
    
    # ========================================
    # 5. 計算統計量
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
    # 6. 儲存 CSV
    # ========================================
    print('[5] Saving metrics to CSV...')
    os.makedirs(os.path.dirname(CONFIG['output_csv']), exist_ok=True)
    
    with open(CONFIG['output_csv'], 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['filename', 'iou', 'dice', 'pixel_acc', 'fp_rate', 'fn_rate', 'tp', 'fp', 'fn', 'tn']
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
                'tp': r['tp'],
                'fp': r['fp'],
                'fn': r['fn'],
                'tn': r['tn'],
            })
    
    print(f'    Saved to: {CONFIG["output_csv"]}')
    print()
    
    # ========================================
    # 7. 最終摘要
    # ========================================
    print('=' * 60)
    print('  TEST SET SUMMARY')
    print('=' * 60)
    print()
    print(f'  Model: UNetSmall (best checkpoint from epoch {checkpoint["epoch"]})')
    print(f'  Test samples: {len(results)}')
    print()
    print('  Metrics (Mean ± Std):')
    print(f'    IoU:        {stats["iou"]["mean"]*100:>6.2f}% ± {stats["iou"]["std"]*100:.2f}%')
    print(f'    Dice:       {stats["dice"]["mean"]*100:>6.2f}% ± {stats["dice"]["std"]*100:.2f}%')
    print(f'    Pixel Acc:  {stats["pixel_acc"]["mean"]*100:>6.2f}% ± {stats["pixel_acc"]["std"]*100:.2f}%')
    print(f'    FP Rate:    {stats["fp_rate"]["mean"]*100:>6.2f}% ± {stats["fp_rate"]["std"]*100:.2f}%')
    print(f'    FN Rate:    {stats["fn_rate"]["mean"]*100:>6.2f}% ± {stats["fn_rate"]["std"]*100:.2f}%')
    print()
    
    # 與 Val 比較
    print('  Comparison with Validation:')
    print(f'    Val IoU (at best checkpoint):  {checkpoint["val_iou"]*100:.2f}%')
    print(f'    Test IoU:                      {stats["iou"]["mean"]*100:.2f}%')
    
    diff = stats["iou"]["mean"] - checkpoint["val_iou"]
    if abs(diff) < 0.02:
        print(f'    Difference:                    {diff*100:+.2f}% (similar performance)')
    elif diff < 0:
        print(f'    Difference:                    {diff*100:+.2f}% (slight degradation)')
    else:
        print(f'    Difference:                    {diff*100:+.2f}% (better on test)')
    
    print()
    print('  Output files:')
    print(f'    - Per-image metrics: {CONFIG["output_csv"]}')
    print(f'    - Visualizations: {CONFIG["output_vis_dir"]}/')
    print()
    print('=' * 60)


if __name__ == '__main__':
    main()
