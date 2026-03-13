"""
訓練 DINO → DL 方案6：混合方案（DINO + 原圖細節）

訓練參數（所有 DINO+DL 方案相同）：
- Batch size: 2
- Epochs: 10
- Learning rate: 1e-4
- Loss: BCE + Dice
- Optimizer: Adam
- Image size: 256×256
- DINO: 凍結（frozen）

輸出：
- outputs/train_dino_hybrid_YYYYMMDD_HHMMSS/checkpoints/best.pth
- outputs/train_dino_hybrid_YYYYMMDD_HHMMSS/training_log.csv
"""

import os
import csv
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from datetime import datetime
import numpy as np
from torchvision import transforms

# 確保輸出立即刷新
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

from models.dino_hybrid_decoder import DINOWithHybridDecoder
from utils.dataset import get_dataloader
from utils.metrics import calculate_metrics
from scripts.train_in_domain import load_list_file


class DiceLoss(nn.Module):
    """Dice Loss（binary，sigmoid 後計算）"""
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred_probs = torch.sigmoid(pred)
        pred_flat = pred_probs.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (
            pred_flat.sum() + target_flat.sum() + self.smooth
        )
        return 1 - dice


class CombinedLoss(nn.Module):
    """BCE + Dice Loss（權重 1:1）"""
    def __init__(self, bce_weight=1.0, dice_weight=1.0):
        super(CombinedLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
    
    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(pred, target)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


class NormalizedDataset(torch.utils.data.Dataset):
    """
    Wrapper dataset 來處理 DINO 需要的 ImageNet normalization
    原始 dataset 輸出 [0, 1] 範圍的影像，需要轉換為 ImageNet normalized
    注意：Hybrid 方案需要原始影像（未 normalize）給淺層 CNN，所以這裡需要特殊處理
    """
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        image, mask = self.base_dataset[idx]
        # image 已經是 [0, 1] 範圍
        # Hybrid 方案需要原始影像（未 normalize）給淺層 CNN
        # 但 DINO 需要 normalize，所以模型內部會處理
        # 這裡返回原始影像（模型會自己 normalize）
        return image, mask


def train_epoch(model, dataloader, criterion, optimizer, device, use_amp=False):
    """訓練一個 epoch，返回 loss 和 metrics"""
    model.train()
    running_loss = 0.0
    total_iou = 0.0
    num_batches = 0
    
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    
    for images, masks in tqdm(dataloader, desc='  Train', leave=False):
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        
        if use_amp:
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
        
        running_loss += loss.item()
        
        # 計算 metrics
        with torch.no_grad():
            batch_metrics = calculate_metrics(outputs, masks)
            total_iou += batch_metrics['iou']
        
        num_batches += 1
        
        # 清理記憶體
        del images, masks, outputs, loss
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    avg_loss = running_loss / num_batches if num_batches > 0 else 0.0
    avg_iou = total_iou / num_batches if num_batches > 0 else 0.0
    return avg_loss, avg_iou


def evaluate(model, dataloader, criterion, device, use_amp=False):
    """評估模型，返回 loss 和 metrics"""
    model.eval()
    running_loss = 0.0
    total_iou = 0.0
    total_dice = 0.0
    total_pixel_acc = 0.0
    run_fp = 0.0
    run_fn = 0.0
    run_pos = 0.0
    run_neg = 0.0
    num_batches = 0
    smooth = 1e-6
    
    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc='  Val', leave=False):
            images = images.to(device)
            masks = masks.to(device)
            
            if use_amp:
                with torch.amp.autocast('cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, masks)
            else:
                outputs = model(images)
                loss = criterion(outputs, masks)
            
            running_loss += loss.item()
            
            # 計算 metrics
            batch_metrics = calculate_metrics(outputs, masks)
            total_iou += batch_metrics['iou']
            total_dice += batch_metrics['dice']
            total_pixel_acc += batch_metrics['pixel_acc']
            
            # FP/FN
            pred_b = (torch.sigmoid(outputs) > 0.5).float()
            gt_b = (masks > 0.5).float()
            run_fp += ((pred_b == 1) & (gt_b == 0)).sum().item()
            run_fn += ((pred_b == 0) & (gt_b == 1)).sum().item()
            run_pos += (gt_b == 1).sum().item()
            run_neg += (gt_b == 0).sum().item()
            
            num_batches += 1
            
            # 清理記憶體
            del images, masks, outputs, loss
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    
    if num_batches == 0:
        return 0.0, {'iou': 0.0, 'dice': 0.0, 'pixel_acc': 0.0, 'fp_rate': 0.0, 'fn_rate': 0.0}
    
    avg_loss = running_loss / num_batches
    fp_rate = run_fp / (run_neg + smooth)
    fn_rate = run_fn / (run_pos + smooth)
    metrics = {
        'iou': total_iou / num_batches,
        'dice': total_dice / num_batches,
        'pixel_acc': total_pixel_acc / num_batches,
        'fp_rate': fp_rate,
        'fn_rate': fn_rate
    }
    
    return avg_loss, metrics


def save_checkpoint(model, optimizer, epoch, val_iou, checkpoint_dir, is_best=False):
    """保存 checkpoint"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_iou': val_iou,
    }
    
    # 保存 latest checkpoint
    latest_path = os.path.join(checkpoint_dir, 'latest.pth')
    torch.save(checkpoint, latest_path)
    
    # 保存 best checkpoint
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'best.pth')
        torch.save(checkpoint, best_path)
        print(f"    ✓ 保存 best checkpoint (val IoU: {val_iou:.4f})")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='訓練 DINO → DL 方案6（混合方案）')
    parser.add_argument('--device', type=str, default=None,
                        help='設備（cpu/cuda），預設自動偵測；如果設為 cpu 則強制使用 CPU')
    args = parser.parse_args()
    
    print("=" * 60)
    print("  訓練 DINO → DL 方案6：混合方案（DINO + 原圖細節）")
    print("=" * 60)
    print()
    
    # 參數設定（與 plan 一致）
    batch_size = 2
    epochs = 10
    learning_rate = 1e-4
    seed = 42
    image_size = (256, 256)
    num_workers = 0  # Windows 建議設為 0
    
    # 設備選擇
    if args.device == 'cpu':
        device = torch.device('cpu')
        use_amp = False  # CPU 不使用混合精度
    elif args.device == 'cuda':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        use_amp = True if device.type == 'cuda' else False
    else:
        # 自動偵測
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        use_amp = True if device.type == 'cuda' else False
    
    # 如果環境變數設定強制使用 CPU，則覆蓋
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if cuda_visible == '' or cuda_visible is None:
        # 環境變數為空字串或 None，強制使用 CPU
        device = torch.device('cpu')
        use_amp = False
        print(f"[強制] 環境變數 CUDA_VISIBLE_DEVICES='{cuda_visible}'，強制使用 CPU")
    
    # 設定隨機種子
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == 'cuda' and torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    print(f"設備: {device}")
    print(f"混合精度訓練: {use_amp}")
    print()
    
    # 載入 split lists
    train_list_file = "outputs/train_list.txt"
    val_list_file = "outputs/val_list.txt"
    
    print(f"載入 train list: {train_list_file}...")
    train_split_list = load_list_file(train_list_file)
    print(f"  共 {len(train_split_list)} 張訓練影像")
    
    print(f"載入 val list: {val_list_file}...")
    val_split_list = load_list_file(val_list_file)
    print(f"  共 {len(val_split_list)} 張驗證影像")
    print()
    
    # 建立 base dataloaders（未 normalize）
    print("建立 dataloaders...")
    train_base_dataset = get_dataloader(
        batch_size=1,  # 先取得 dataset
        shuffle=False,
        transform=True,  # 訓練時使用 augmentation
        image_size=image_size,
        num_workers=0,
        split_list=train_split_list,
        base_data_dir="data"
    ).dataset
    
    val_base_dataset = get_dataloader(
        batch_size=1,
        shuffle=False,
        transform=False,  # 驗證時不使用 augmentation
        image_size=image_size,
        num_workers=0,
        split_list=val_split_list,
        base_data_dir="data"
    ).dataset
    
    # 包裝成 normalized dataset（Hybrid 方案需要原始影像）
    train_dataset = NormalizedDataset(train_base_dataset)
    val_dataset = NormalizedDataset(val_base_dataset)
    
    # 建立 dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print()
    
    # 建立模型
    print("建立模型...")
    print("  載入 DINOv2 ViT-B/14...")
    model = DINOWithHybridDecoder(device=device, target_size=256)
    
    # 確認 DINO 已凍結
    dino_params = sum(p.numel() for p in model.dino_model.parameters())
    decoder_params = sum(p.numel() for p in model.decoder.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"  DINO 參數（凍結）: {dino_params:,}")
    print(f"  Decoder 參數（可訓練）: {decoder_params:,}")
    print(f"  總可訓練參數: {trainable_params:,}")
    print()
    
    # Loss 和 Optimizer（只優化 decoder）
    criterion = CombinedLoss(bce_weight=1.0, dice_weight=1.0)
    optimizer = optim.Adam(model.decoder.parameters(), lr=learning_rate)
    
    # 建立輸出目錄
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"outputs/train_dino_hybrid_{timestamp}"
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    print(f"輸出目錄: {output_dir}")
    print()
    
    # 訓練記錄
    training_log = []
    best_val_iou = 0.0
    
    # 訓練循環
    print("開始訓練...")
    print()
    
    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}/{epochs}")
        
        # 訓練
        train_loss, train_iou = train_epoch(model, train_loader, criterion, optimizer, device, use_amp)
        
        # 驗證
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device, use_amp)
        val_iou = val_metrics['iou']
        
        # 記錄
        log_entry = {
            'epoch': epoch,
            'train_loss': train_loss,
            'train_iou': train_iou,
            'val_loss': val_loss,
            'val_iou': val_iou,
            'val_dice': val_metrics['dice'],
            'val_pixel_acc': val_metrics['pixel_acc'],
            'val_fp_rate': val_metrics['fp_rate'],
            'val_fn_rate': val_metrics['fn_rate']
        }
        training_log.append(log_entry)
        
        # 輸出結果
        print(f"  Train Loss: {train_loss:.4f}, Train IoU: {train_iou:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}, Val Dice: {val_metrics['dice']:.4f}")
        print(f"  Val FP Rate: {val_metrics['fp_rate']:.4f}, Val FN Rate: {val_metrics['fn_rate']:.4f}")
        
        # 保存 checkpoint
        is_best = val_iou > best_val_iou
        if is_best:
            best_val_iou = val_iou
        
        save_checkpoint(model, optimizer, epoch, val_iou, checkpoint_dir, is_best)
        
        print()
    
    # 保存訓練記錄
    log_file = os.path.join(output_dir, "training_log.csv")
    with open(log_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=training_log[0].keys())
        writer.writeheader()
        writer.writerows(training_log)
    
    print("=" * 60)
    print("  訓練完成")
    print("=" * 60)
    print(f"Best Val IoU: {best_val_iou:.4f}")
    print(f"輸出目錄: {output_dir}")
    print(f"Best checkpoint: {os.path.join(checkpoint_dir, 'best.pth')}")
    print()


if __name__ == "__main__":
    main()
