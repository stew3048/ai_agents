"""
U-Net 訓練腳本 - SkyFinder Camera 10066

最小可跑版本：
- 用 train split 訓練
- 每個 epoch 計算 val metrics (IoU, Dice, PixelAcc) 和 val loss
- 輸出 val 的 pred overlay
- 保存最佳模型（以 val IoU 為準）
- 繪製 train/val loss 曲線
"""

import os
import sys
import csv
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# 加入專案路徑
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.unet import UNetSmall
from src.utils.losses import combined_loss
from src.datasets import get_skyfinder_dataloader


# ============================================================
# 設定
# ============================================================
CONFIG = {
    'num_epochs': 10,
    'batch_size': 2,  # GPU 記憶體有限，用小 batch
    'learning_rate': 1e-4,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'splits_json': 'outputs/splits_10066.json',
    'images_dir': 'data/skyfinder_sample/images',
    'masks_dir': 'data/skyfinder_sample/masks',
    'output_dir': 'outputs/unet_10066_vis',
    'metrics_csv': 'outputs/unet_10066_metrics.csv',
    'best_model_path': 'outputs/unet_10066_best.pt',
    'loss_plot_path': 'outputs/unet_10066_loss_curve.png',
    'num_vis_samples': 7,  # 輸出全部 val 樣本 (036-042)
}

# CV baseline 的 IoU（用於比較）
CV_BASELINE_IOU = 30.68


# ============================================================
# Metrics 計算
# ============================================================
def compute_metrics(pred_mask, gt_mask):
    """
    計算 IoU, Dice, Pixel Accuracy
    
    Args:
        pred_mask: 預測 mask (numpy array, 0/1)
        gt_mask: Ground truth mask (numpy array, 0/1)
    
    Returns:
        dict with iou, dice, pixel_acc
    """
    smooth = 1e-6
    
    pred = pred_mask.flatten()
    gt = gt_mask.flatten()
    
    # Intersection and Union
    intersection = (pred * gt).sum()
    union = pred.sum() + gt.sum() - intersection
    
    # IoU
    iou = (intersection + smooth) / (union + smooth)
    
    # Dice
    dice = (2 * intersection + smooth) / (pred.sum() + gt.sum() + smooth)
    
    # Pixel Accuracy
    pixel_acc = (pred == gt).sum() / pred.size
    
    return {
        'iou': iou,
        'dice': dice,
        'pixel_acc': pixel_acc
    }


# ============================================================
# 視覺化
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


def save_visualization(images, masks, preds, epoch, output_dir, dataset, num_samples=7):
    """儲存視覺化結果（全部 val 樣本）"""
    epoch_dir = os.path.join(output_dir, f'epoch_{epoch:02d}')
    os.makedirs(epoch_dir, exist_ok=True)
    
    for i in range(min(num_samples, len(images))):
        # 轉換為 numpy
        img_np = (images[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        gt_np = masks[i].squeeze().cpu().numpy().astype(np.uint8)
        pred_np = preds[i].squeeze().cpu().numpy().astype(np.uint8)
        
        # 取得檔名
        filename = dataset.get_filename(i).replace('.jpg', '').replace('.png', '')
        
        # 儲存原圖
        Image.fromarray(img_np).save(os.path.join(epoch_dir, f'{filename}_1_image.png'))
        
        # 儲存 GT mask
        Image.fromarray((gt_np * 255).astype(np.uint8)).save(
            os.path.join(epoch_dir, f'{filename}_2_gt.png'))
        
        # 儲存 pred mask
        Image.fromarray((pred_np * 255).astype(np.uint8)).save(
            os.path.join(epoch_dir, f'{filename}_3_pred.png'))
        
        # 儲存 overlay
        overlay = create_pred_overlay(img_np, pred_np, gt_np)
        Image.fromarray(overlay).save(os.path.join(epoch_dir, f'{filename}_4_overlay.png'))


def plot_loss_curve(history, best_epoch, output_path):
    """繪製 train/val loss 曲線"""
    epochs = [r['epoch'] for r in history]
    train_losses = [r['train_loss'] for r in history]
    val_losses = [r['val_loss'] for r in history]
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, 'b-o', label='Train Loss', linewidth=2, markersize=8)
    plt.plot(epochs, val_losses, 'r-s', label='Val Loss', linewidth=2, markersize=8)
    
    # 標註最佳 epoch
    best_idx = best_epoch - 1
    best_val_loss = val_losses[best_idx]
    plt.axvline(x=best_epoch, color='green', linestyle='--', linewidth=2, label=f'Best IoU (Epoch {best_epoch})')
    plt.scatter([best_epoch], [best_val_loss], color='green', s=200, zorder=5, marker='*')
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss Curve', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xticks(epochs)
    
    # 加入文字標註
    plt.annotate(f'Best IoU\nEpoch {best_epoch}', 
                 xy=(best_epoch, best_val_loss),
                 xytext=(best_epoch + 0.5, best_val_loss + 0.02),
                 fontsize=10,
                 arrowprops=dict(arrowstyle='->', color='green'))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f'    Loss curve saved to: {output_path}')


# ============================================================
# 訓練一個 Epoch
# ============================================================
def train_one_epoch(model, dataloader, optimizer, device):
    """訓練一個 epoch，回傳平均 loss"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    for images, masks in dataloader:
        images = images.to(device)
        masks = masks.to(device)
        
        # Forward
        optimizer.zero_grad()
        logits = model(images)
        loss = combined_loss(logits, masks)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


# ============================================================
# 驗證
# ============================================================
def validate(model, dataloader, device):
    """
    驗證模型，計算 metrics 和 val loss
    回傳平均 metrics、val loss 和用於視覺化的資料
    """
    model.eval()
    
    all_metrics = []
    total_loss = 0
    num_batches = 0
    vis_data = {'images': [], 'masks': [], 'preds': []}
    
    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward
            logits = model(images)
            loss = combined_loss(logits, masks)
            total_loss += loss.item()
            num_batches += 1
            
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            
            # 計算每個樣本的 metrics
            for i in range(images.size(0)):
                pred_np = preds[i].squeeze().cpu().numpy()
                gt_np = masks[i].squeeze().cpu().numpy()
                metrics = compute_metrics(pred_np, gt_np)
                all_metrics.append(metrics)
            
            # 保存視覺化資料（保存所有 batch）
            vis_data['images'].append(images.cpu())
            vis_data['masks'].append(masks.cpu())
            vis_data['preds'].append(preds.cpu())
    
    # 合併視覺化資料
    vis_data['images'] = torch.cat(vis_data['images'], dim=0)
    vis_data['masks'] = torch.cat(vis_data['masks'], dim=0)
    vis_data['preds'] = torch.cat(vis_data['preds'], dim=0)
    
    # 計算平均 metrics
    avg_metrics = {
        'iou': np.mean([m['iou'] for m in all_metrics]),
        'dice': np.mean([m['dice'] for m in all_metrics]),
        'pixel_acc': np.mean([m['pixel_acc'] for m in all_metrics])
    }
    
    avg_loss = total_loss / num_batches
    
    return avg_metrics, avg_loss, vis_data


# ============================================================
# 主訓練流程
# ============================================================
def main():
    print('=' * 60)
    print('  U-Net Training on SkyFinder 10066')
    print('=' * 60)
    print()
    
    # 設定
    device = CONFIG['device']
    print(f'  Device: {device}')
    print(f'  Epochs: {CONFIG["num_epochs"]}')
    print(f'  Batch size: {CONFIG["batch_size"]}')
    print(f'  Learning rate: {CONFIG["learning_rate"]}')
    print()
    
    # 顯示資料路徑
    print('  Data paths:')
    print(f'    Splits JSON: {CONFIG["splits_json"]}')
    print(f'    Images dir:  {CONFIG["images_dir"]}')
    print(f'    Masks dir:   {CONFIG["masks_dir"]}')
    print()
    
    # 建立輸出目錄
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    os.makedirs(os.path.dirname(CONFIG['metrics_csv']), exist_ok=True)
    
    # 載入資料
    print('[1] Loading data...')
    train_loader = get_skyfinder_dataloader(
        splits_json=CONFIG['splits_json'],
        split='train',
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        transform=True  # 訓練時用 augmentation
    )
    
    val_loader = get_skyfinder_dataloader(
        splits_json=CONFIG['splits_json'],
        split='val',
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        transform=False  # 驗證時不用 augmentation
    )
    
    # 取得 val dataset 用於檔名
    val_dataset = val_loader.dataset
    
    print(f'    Train samples: {len(train_loader.dataset)}')
    print(f'    Val samples: {len(val_loader.dataset)}')
    print(f'    Val files: {[val_dataset.get_filename(i) for i in range(len(val_dataset))]}')
    print()
    
    # 建立模型
    print('[2] Building model...')
    model = UNetSmall(in_channels=3, out_channels=1).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'    Model: UNetSmall')
    print(f'    Parameters: {num_params:,}')
    print()
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    
    # 訓練記錄
    history = []
    best_val_iou = 0
    best_epoch = 0
    
    # 訓練迴圈
    print('[3] Training...')
    print('-' * 75)
    print(f'  {"Epoch":>5} | {"Train Loss":>10} | {"Val Loss":>10} | {"Val IoU":>8} | {"Val Dice":>9} | {"Val Acc":>8}')
    print('-' * 75)
    
    for epoch in range(1, CONFIG['num_epochs'] + 1):
        # 訓練
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        
        # 驗證
        val_metrics, val_loss, vis_data = validate(model, val_loader, device)
        
        # 記錄
        record = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_iou': val_metrics['iou'],
            'val_dice': val_metrics['dice'],
            'val_pixel_acc': val_metrics['pixel_acc']
        }
        history.append(record)
        
        # 輸出
        print(f'  {epoch:>5} | {train_loss:>10.4f} | {val_loss:>10.4f} | {val_metrics["iou"]*100:>7.2f}% | {val_metrics["dice"]*100:>8.2f}% | {val_metrics["pixel_acc"]*100:>7.2f}%')
        
        # 儲存視覺化
        save_visualization(
            vis_data['images'], vis_data['masks'], vis_data['preds'],
            epoch, CONFIG['output_dir'], val_dataset, CONFIG['num_vis_samples']
        )
        
        # 儲存最佳模型
        if val_metrics['iou'] > best_val_iou:
            best_val_iou = val_metrics['iou']
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_iou': val_metrics['iou'],
                'val_dice': val_metrics['dice'],
                'val_loss': val_loss,
            }, CONFIG['best_model_path'])
    
    print('-' * 75)
    print()
    
    # 儲存 metrics CSV
    print('[4] Saving metrics...')
    with open(CONFIG['metrics_csv'], 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['epoch', 'train_loss', 'val_loss', 'val_iou', 'val_dice', 'val_pixel_acc']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in history:
            writer.writerow({
                'epoch': record['epoch'],
                'train_loss': f"{record['train_loss']:.4f}",
                'val_loss': f"{record['val_loss']:.4f}",
                'val_iou': f"{record['val_iou']:.4f}",
                'val_dice': f"{record['val_dice']:.4f}",
                'val_pixel_acc': f"{record['val_pixel_acc']:.4f}"
            })
    print(f'    Saved to: {CONFIG["metrics_csv"]}')
    print(f'    Best model: {CONFIG["best_model_path"]}')
    
    # 繪製 loss 曲線
    print()
    print('[5] Plotting loss curve...')
    plot_loss_curve(history, best_epoch, CONFIG['loss_plot_path'])
    print()
    
    # ========================================
    # 最終摘要
    # ========================================
    print('=' * 60)
    print('  TRAINING SUMMARY')
    print('=' * 60)
    print()
    print(f'  Best Epoch: {best_epoch}')
    print(f'  Best Val IoU: {best_val_iou*100:.2f}%')
    print()
    print('  Comparison with CV Baseline:')
    print(f'    CV Baseline IoU: {CV_BASELINE_IOU:.2f}%')
    print(f'    U-Net Best IoU:  {best_val_iou*100:.2f}%')
    print(f'    Improvement:     {best_val_iou*100 - CV_BASELINE_IOU:+.2f}%')
    print()
    
    if best_val_iou * 100 > CV_BASELINE_IOU:
        improvement_ratio = (best_val_iou * 100) / CV_BASELINE_IOU
        print(f'  Result: U-Net outperforms CV baseline by {improvement_ratio:.1f}x!')
    else:
        print('  Result: U-Net needs more training or tuning.')
    
    print()
    print('  Output files:')
    print(f'    - Metrics: {CONFIG["metrics_csv"]}')
    print(f'    - Best model: {CONFIG["best_model_path"]}')
    print(f'    - Loss curve: {CONFIG["loss_plot_path"]}')
    print(f'    - Visualizations: {CONFIG["output_dir"]}/epoch_*/')
    print()
    
    # Overfitting 檢查
    final_train_loss = history[-1]['train_loss']
    final_val_loss = history[-1]['val_loss']
    loss_gap = final_val_loss - final_train_loss
    print('  Overfitting check:')
    print(f'    Final Train Loss: {final_train_loss:.4f}')
    print(f'    Final Val Loss:   {final_val_loss:.4f}')
    print(f'    Gap (Val - Train): {loss_gap:.4f}')
    if loss_gap > 0.05:
        print('    Warning: Possible overfitting detected!')
    else:
        print('    Status: No significant overfitting.')
    print()
    print('=' * 60)


if __name__ == '__main__':
    main()
