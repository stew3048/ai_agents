"""
使用 in-domain split（train_list.txt, val_list.txt）重新訓練 DL baseline (U-Net)

變更：
- 讀取 outputs/train_list.txt 與 outputs/val_list.txt（而非 multi_camera_splits.json）
- 使用 val_list.txt 做 early stopping（依 val IoU）
- 訓練參數：沿用現有參數（batch_size=2, epochs=10, learning_rate=1e-4, seed 等）
- 保存 best checkpoint

輸出：
- outputs/train_in_domain_YYYYMMDD_HHMMSS/checkpoints/best.pth
- outputs/train_in_domain_YYYYMMDD_HHMMSS/training_log.csv
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
from PIL import Image
from pathlib import Path

# 確保輸出立即刷新
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# 添加專案根目錄到 Python 路徑
_script_dir = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_script_dir)
sys.path.insert(0, _root)

from models import create_unet_model
from utils.dataset import get_dataloader
from utils.metrics import calculate_metrics


def load_list_file(list_file: str, base_data_dir: str = "data"):
    """
    從 list 檔案載入資料（每行一個相對路徑，如 skyfinder_10066/images/001.jpg）
    
    返回:
        List[Dict]: 每個 dict 包含 {'camera_id': str, 'image': str, 'mask': str}
    """
    split_list = []
    
    if not os.path.exists(list_file):
        raise FileNotFoundError(f"找不到 list 檔案：{list_file}")
    
    with open(list_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # 解析相對路徑：skyfinder_10066/images/001.jpg
            # 或：skyfinder_10066\images\001.jpg（Windows）
            parts = line.replace('\\', '/').split('/')
            
            if len(parts) < 3:
                print(f"  警告：無法解析路徑：{line}")
                continue
            
            # parts[0] = skyfinder_10066
            # parts[1] = images
            # parts[2:] = 001.jpg (可能包含子目錄)
            camera_folder = parts[0]
            if not camera_folder.startswith('skyfinder_'):
                print(f"  警告：無效的 camera 資料夾名稱：{camera_folder}")
                continue
            
            camera_id = camera_folder.replace('skyfinder_', '')
            image_filename = '/'.join(parts[2:])  # 處理可能的子目錄
            
            # 找到對應的 mask
            # 嘗試多種可能的 mask 檔名
            img_base = os.path.splitext(image_filename)[0]
            possible_mask_names = [
                f"{img_base}.png",
                f"{img_base}.pgm",
                f"{img_base}.jpg",
                image_filename,  # 相同檔名
            ]
            
            mask_filename = None
            for mask_name in possible_mask_names:
                mask_path = os.path.join(base_data_dir, camera_folder, "masks", mask_name)
                if os.path.exists(mask_path):
                    mask_filename = mask_name
                    break
            
            if mask_filename is None:
                print(f"  警告：找不到 mask 檔案：{line}")
                continue
            
            split_list.append({
                'camera_id': camera_id,
                'image': image_filename,
                'mask': mask_filename
            })
    
    return split_list


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
    
    parser = argparse.ArgumentParser(description='訓練 DL baseline（使用 in-domain split）')
    parser.add_argument('--device', type=str, default=None,
                        help='設備（cpu/cuda），預設自動偵測；如果設為 cpu 則強制使用 CPU')
    args = parser.parse_args()
    
    print("=" * 60)
    print("  使用 in-domain split 重新訓練 DL baseline (U-Net)")
    print("=" * 60)
    print()
    
    # 參數設定（與 train_multi_camera.py 一致）
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
    
    # 建立 dataloaders
    print("建立 dataloaders...")
    train_loader = get_dataloader(
        batch_size=batch_size,
        shuffle=True,
        transform=True,  # 訓練時使用 augmentation
        image_size=image_size,
        num_workers=num_workers,
        split_list=train_split_list,
        base_data_dir="data"
    )
    
    val_loader = get_dataloader(
        batch_size=batch_size,
        shuffle=False,
        transform=False,  # 驗證時不使用 augmentation
        image_size=image_size,
        num_workers=num_workers,
        split_list=val_split_list,
        base_data_dir="data"
    )
    
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print()
    
    # 建立模型
    print("建立模型...")
    model = create_unet_model().to(device)
    print(f"  模型參數數量: {sum(p.numel() for p in model.parameters()):,}")
    print()
    
    # Loss 和 Optimizer
    criterion = CombinedLoss(bce_weight=1.0, dice_weight=1.0)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 建立輸出目錄
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"outputs/train_in_domain_{timestamp}"
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
