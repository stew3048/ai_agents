"""
訓練腳本
用於訓練天空分割模型
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse
from datetime import datetime

from models import create_unet_model
from utils import get_dataloader, calculate_metrics


class DiceLoss(nn.Module):
    """
    Dice Loss
    用於二進制分割任務的損失函數
    
    Dice Loss 的優點：
    - 對類別不平衡問題較不敏感
    - 直接優化 Dice 分數（常用的評估指標）
    """
    
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        """
        計算 Dice Loss
        
        參數:
            pred: 預測 logits [B, 1, H, W]
            target: 真實 mask [B, 1, H, W]，值為 0 或 1
        
        返回:
            dice_loss: Dice Loss 值
        """
        # 將 logits 轉換為機率
        pred_probs = torch.sigmoid(pred)
        
        # 展平張量
        pred_flat = pred_probs.view(-1)
        target_flat = target.view(-1)
        
        # 計算交集
        intersection = (pred_flat * target_flat).sum()
        
        # 計算 Dice Loss
        dice = (2.0 * intersection + self.smooth) / (
            pred_flat.sum() + target_flat.sum() + self.smooth
        )
        
        # Dice Loss = 1 - Dice Score
        dice_loss = 1 - dice
        
        return dice_loss


class CombinedLoss(nn.Module):
    """
    組合損失函數
    結合 BCE Loss 和 Dice Loss
    
    優點：
    - BCE Loss 提供穩定的梯度
    - Dice Loss 直接優化評估指標
    - 兩者結合通常能獲得更好的結果
    """
    
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()
    
    def forward(self, pred, target):
        """
        計算組合損失
        
        參數:
            pred: 預測 logits [B, 1, H, W]
            target: 真實 mask [B, 1, H, W]
        
        返回:
            combined_loss: 組合損失值
        """
        bce = self.bce_loss(pred, target)
        dice = self.dice_loss(pred, target)
        
        combined_loss = self.bce_weight * bce + self.dice_weight * dice
        return combined_loss


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """
    訓練一個 epoch
    
    參數:
        model: 模型
        dataloader: 訓練資料載入器
        criterion: 損失函數
        optimizer: 優化器
        device: 計算設備（CPU 或 CUDA）
        epoch: 當前 epoch 編號
    
    返回:
        avg_loss: 平均損失
        metrics: 評估指標字典
    """
    model.train()  # 設置為訓練模式
    running_loss = 0.0
    total_iou = 0.0
    total_dice = 0.0
    total_pixel_acc = 0.0
    num_batches = 0
    
    # 使用 tqdm 顯示進度條
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Train]')
    
    for batch_idx, (images, masks) in enumerate(pbar):
        # 將資料移到指定設備
        images = images.to(device)
        masks = masks.to(device)
        
        # 前向傳播
        optimizer.zero_grad()  # 清零梯度
        outputs = model(images)  # 模型預測 [B, 1, H, W]
        
        # 計算損失
        loss = criterion(outputs, masks)
        
        # 反向傳播
        loss.backward()  # 計算梯度
        optimizer.step()  # 更新參數
        
        # 累積損失
        running_loss += loss.item()
        
        # 計算評估指標（用於監控訓練進度）
        with torch.no_grad():
            batch_metrics = calculate_metrics(outputs, masks)
            total_iou += batch_metrics['iou']
            total_dice += batch_metrics['dice']
            total_pixel_acc += batch_metrics['pixel_acc']
        
        num_batches += 1
        
        # 更新進度條
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'iou': f'{batch_metrics["iou"]:.4f}',
            'dice': f'{batch_metrics["dice"]:.4f}'
        })
    
    # 計算平均值
    avg_loss = running_loss / num_batches
    metrics = {
        'loss': avg_loss,
        'iou': total_iou / num_batches,
        'dice': total_dice / num_batches,
        'pixel_acc': total_pixel_acc / num_batches
    }
    
    return avg_loss, metrics


def validate(model, dataloader, criterion, device, epoch):
    """
    驗證模型
    
    參數:
        model: 模型
        dataloader: 驗證資料載入器
        criterion: 損失函數
        device: 計算設備
        epoch: 當前 epoch 編號
    
    返回:
        avg_loss: 平均損失
        metrics: 評估指標字典
    """
    model.eval()  # 設置為評估模式
    running_loss = 0.0
    total_iou = 0.0
    total_dice = 0.0
    total_pixel_acc = 0.0
    num_batches = 0
    
    # 不計算梯度，節省記憶體和計算
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Val]')
        
        for images, masks in pbar:
            # 將資料移到指定設備
            images = images.to(device)
            masks = masks.to(device)
            
            # 前向傳播
            outputs = model(images)
            
            # 計算損失
            loss = criterion(outputs, masks)
            
            # 累積損失
            running_loss += loss.item()
            
            # 計算評估指標
            batch_metrics = calculate_metrics(outputs, masks)
            total_iou += batch_metrics['iou']
            total_dice += batch_metrics['dice']
            total_pixel_acc += batch_metrics['pixel_acc']
            
            num_batches += 1
            
            # 更新進度條
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'iou': f'{batch_metrics["iou"]:.4f}',
                'dice': f'{batch_metrics["dice"]:.4f}'
            })
    
    # 計算平均值
    avg_loss = running_loss / num_batches
    metrics = {
        'loss': avg_loss,
        'iou': total_iou / num_batches,
        'dice': total_dice / num_batches,
        'pixel_acc': total_pixel_acc / num_batches
    }
    
    return avg_loss, metrics


def save_checkpoint(model, optimizer, epoch, loss, metrics, checkpoint_dir, is_best=False):
    """
    保存模型檢查點
    
    參數:
        model: 模型
        optimizer: 優化器
        epoch: epoch 編號
        loss: 損失值
        metrics: 評估指標
        checkpoint_dir: 檢查點保存目錄
        is_best: 是否為最佳模型
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'metrics': metrics
    }
    
    # 保存最新檢查點
    latest_path = os.path.join(checkpoint_dir, 'latest.pth')
    torch.save(checkpoint, latest_path)
    
    # 保存最佳模型
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'best.pth')
        torch.save(checkpoint, best_path)
        print(f"[BEST] 保存最佳模型 (IoU: {metrics['iou']:.4f}, Dice: {metrics['dice']:.4f})")
    
    # 保存每個 epoch 的檢查點（可選）
    epoch_path = os.path.join(checkpoint_dir, f'epoch_{epoch}.pth')
    torch.save(checkpoint, epoch_path)


def train(args):
    """
    主訓練函數
    
    參數:
        args: 命令行參數
    """
    # 設置設備
    use_cuda = torch.cuda.is_available() and not args.cpu_only
    
    # 檢查 GPU 記憶體，如果小於 4GB 且未強制使用 GPU，則建議使用 CPU
    if use_cuda:
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        gpu_name = torch.cuda.get_device_name(0)
        print(f"偵測到 GPU: {gpu_name} ({gpu_memory:.1f}GB)")
        if gpu_memory < 4.0 and not args.force_gpu:
            print(f"[WARN] GPU 記憶體只有 {gpu_memory:.1f}GB，建議使用 CPU 或減小 batch_size")
            print(f"[WARN] 自動切換到 CPU（使用 --force_gpu 強制使用 GPU）")
            use_cuda = False
    
    device = torch.device('cuda' if use_cuda else 'cpu')
    print(f"使用設備: {device}")
    
    # 創建輸出目錄
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join('outputs', f'train_{timestamp}')
    checkpoint_dir = os.path.join(run_dir, 'checkpoints')
    log_dir = os.path.join(run_dir, 'logs')
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # TensorBoard 記錄器
    writer = SummaryWriter(log_dir=log_dir)
    
    # 創建模型
    print("創建模型...")
    model = create_unet_model(n_channels=3, n_classes=1)
    model = model.to(device)
    
    # 計算模型參數數量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型參數總數: {total_params:,}")
    
    # 創建資料載入器
    print("載入資料集...")
    train_loader = get_dataloader(
        images_dir=args.train_images_dir,
        masks_dir=args.train_masks_dir,
        batch_size=args.batch_size,
        shuffle=True,
        transform=True,  # 訓練時開啟資料增強
        image_size=args.image_size,
        num_workers=args.num_workers
    )
    
    val_loader = get_dataloader(
        images_dir=args.val_images_dir,
        masks_dir=args.val_masks_dir,
        batch_size=args.batch_size,
        shuffle=False,
        transform=False,  # 驗證時關閉資料增強
        image_size=args.image_size,
        num_workers=args.num_workers
    )
    
    print(f"訓練樣本數: {len(train_loader.dataset)}")
    print(f"驗證樣本數: {len(val_loader.dataset)}")
    
    # 定義損失函數
    if args.loss == 'bce':
        criterion = nn.BCEWithLogitsLoss()
        print("使用損失函數: BCEWithLogitsLoss")
    elif args.loss == 'dice':
        criterion = DiceLoss()
        print("使用損失函數: Dice Loss")
    elif args.loss == 'combined':
        criterion = CombinedLoss(bce_weight=0.5, dice_weight=0.5)
        print("使用損失函數: Combined Loss (BCE + Dice)")
    else:
        raise ValueError(f"不支援的損失函數: {args.loss}")
    
    # 定義優化器
    if args.optimizer == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
        print(f"使用優化器: Adam (lr={args.learning_rate})")
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.learning_rate,
            momentum=0.9,
            weight_decay=args.weight_decay
        )
        print(f"使用優化器: SGD (lr={args.learning_rate})")
    else:
        raise ValueError(f"不支援的優化器: {args.optimizer}")
    
    # 學習率調度器（可選）
    scheduler = None
    if args.use_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        print("使用學習率調度器: ReduceLROnPlateau")
    
    # 訓練迴圈
    print("\n開始訓練...")
    print("=" * 60)
    
    best_val_iou = 0.0
    
    for epoch in range(1, args.epochs + 1):
        # 訓練
        train_loss, train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # 驗證
        val_loss, val_metrics = validate(
            model, val_loader, criterion, device, epoch
        )
        
        # 更新學習率（如果使用調度器）
        if scheduler is not None:
            scheduler.step(val_loss)
        
        # 記錄到 TensorBoard
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Val', val_loss, epoch)
        writer.add_scalar('IoU/Train', train_metrics['iou'], epoch)
        writer.add_scalar('IoU/Val', val_metrics['iou'], epoch)
        writer.add_scalar('Dice/Train', train_metrics['dice'], epoch)
        writer.add_scalar('Dice/Val', val_metrics['dice'], epoch)
        writer.add_scalar('PixelAcc/Train', train_metrics['pixel_acc'], epoch)
        writer.add_scalar('PixelAcc/Val', val_metrics['pixel_acc'], epoch)
        
        # 打印結果
        print(f"\nEpoch {epoch}/{args.epochs}:")
        print(f"  Train - Loss: {train_loss:.4f}, IoU: {train_metrics['iou']:.4f}, "
              f"Dice: {train_metrics['dice']:.4f}, Acc: {train_metrics['pixel_acc']:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, IoU: {val_metrics['iou']:.4f}, "
              f"Dice: {val_metrics['dice']:.4f}, Acc: {val_metrics['pixel_acc']:.4f}")
        
        # 保存檢查點
        is_best = val_metrics['iou'] > best_val_iou
        if is_best:
            best_val_iou = val_metrics['iou']
        
        save_checkpoint(
            model, optimizer, epoch, val_loss, val_metrics,
            checkpoint_dir, is_best=is_best
        )
        
        print("-" * 60)
    
    print("\n訓練完成！")
    print(f"最佳驗證 IoU: {best_val_iou:.4f}")
    print(f"模型保存在: {checkpoint_dir}")
    print(f"TensorBoard 日誌: {log_dir}")
    print(f"查看日誌: tensorboard --logdir {log_dir}")
    
    writer.close()


def main():
    """主函數"""
    parser = argparse.ArgumentParser(description='訓練天空分割模型')
    
    # 資料路徑
    parser.add_argument('--train_images_dir', type=str, required=True,
                       help='訓練圖片資料夾路徑')
    parser.add_argument('--train_masks_dir', type=str, required=True,
                       help='訓練 mask 資料夾路徑')
    parser.add_argument('--val_images_dir', type=str, required=True,
                       help='驗證圖片資料夾路徑')
    parser.add_argument('--val_masks_dir', type=str, required=True,
                       help='驗證 mask 資料夾路徑')
    
    # 訓練參數
    parser.add_argument('--epochs', type=int, default=50,
                       help='訓練輪數，預設 50')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='批次大小，預設 8')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='學習率，預設 1e-4')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='權重衰減（L2 正則化），預設 1e-5')
    
    # 模型參數
    parser.add_argument('--image_size', type=int, nargs=2, default=[256, 256],
                       help='圖片尺寸 [height, width]，預設 [256, 256]')
    
    # 損失函數和優化器
    parser.add_argument('--loss', type=str, default='combined',
                       choices=['bce', 'dice', 'combined'],
                       help='損失函數：bce/dice/combined，預設 combined')
    parser.add_argument('--optimizer', type=str, default='adam',
                       choices=['adam', 'sgd'],
                       help='優化器：adam/sgd，預設 adam')
    parser.add_argument('--use_scheduler', action='store_true',
                       help='是否使用學習率調度器')
    
    # 其他參數
    parser.add_argument('--num_workers', type=int, default=0,
                       help='資料載入進程數，Windows 建議設為 0')
    parser.add_argument('--cpu_only', action='store_true',
                       help='強制只使用 CPU')
    parser.add_argument('--force_gpu', action='store_true',
                       help='強制使用 GPU（即使記憶體不足）')
    
    args = parser.parse_args()
    
    # 開始訓練
    train(args)


if __name__ == '__main__':
    main()
