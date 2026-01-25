"""
使用 multi_camera_splits.json 訓練 U-Net 模型

訓練設定：
- epochs = 10
- 每 epoch 輸出：train loss/IoU, sanity IoU, val IoU/Dice/PixelAcc
- 依 val IoU 儲存 best checkpoint
- 不使用 augmentation（transform=False）
- 記錄訓練過程到 CSV 檔案
- 訓練結束後生成 val overlay 視覺化（最好和最差各 5 張）
"""

import os
import json
import csv
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from datetime import datetime
import numpy as np
from PIL import Image

# 確保輸出立即刷新
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

from models import create_unet_model
from utils.dataset import get_dataloader, load_splits_from_json
from utils.metrics import calculate_metrics
from pathlib import Path


class DiceLoss(nn.Module):
    """Dice Loss"""
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
        if device.type == 'cpu':
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    avg_loss = running_loss / num_batches if num_batches > 0 else 0.0
    avg_iou = total_iou / num_batches if num_batches > 0 else 0.0
    return avg_loss, avg_iou


def evaluate(model, dataloader, criterion, device, desc='Eval', use_amp=False):
    """評估模型（計算 loss、metrics 與 FP/FN rate）"""
    model.eval()
    running_loss = 0.0
    total_iou = 0.0
    total_dice = 0.0
    total_pixel_acc = 0.0
    run_fp, run_fn, run_neg, run_pos = 0.0, 0.0, 0.0, 0.0
    num_batches = 0
    smooth = 1e-6

    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc=f'  {desc}', leave=True):
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
            batch_metrics = calculate_metrics(outputs, masks)
            total_iou += batch_metrics['iou']
            total_dice += batch_metrics['dice']
            total_pixel_acc += batch_metrics['pixel_acc']

            pred_b = (torch.sigmoid(outputs) > 0.5).float()
            gt_b = (masks > 0.5).float()
            run_fp += ((pred_b == 1) & (gt_b == 0)).sum().item()
            run_fn += ((pred_b == 0) & (gt_b == 1)).sum().item()
            run_neg += (gt_b == 0).sum().item()
            run_pos += (gt_b == 1).sum().item()

            num_batches += 1

            # 清理記憶體
            del images, masks, outputs, loss
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

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


def _source_id_from_path(path: str) -> str:
    """從 image 路徑產生可對照原檔的短 id：camera_id_filename（例 10870_xxx.jpg）"""
    fname = os.path.basename(path)
    parent2 = os.path.dirname(os.path.dirname(path))
    folder = os.path.basename(parent2)
    if folder.startswith('skyfinder_'):
        camera_id = folder[len('skyfinder_'):]
    else:
        camera_id = folder or 'unknown'
    return f"{camera_id}_{fname}"


def evaluate_with_predictions(model, dataloader, device, use_amp=False, split_label=None):
    """
    評估並返回所有預測結果（用於生成 overlay）。
    - split_label: 若提供（如 'val','test'），source_id 使用編碼 {split_label}_{index:04d}，方便對照
      該 split 清單的第 index 筆即為原圖（例 val_0042 → val 清單第 42 筆）。
    - 每筆含 source_id、orig_path（有 paths 時）。
    """
    model.eval()
    all_results = []
    ds = getattr(dataloader, 'dataset', None)
    paths = getattr(ds, 'image_paths', None) if ds else None
    bs = dataloader.batch_size

    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(tqdm(dataloader, desc='  Eval (with predictions)', leave=False)):
            images = images.to(device)
            masks = masks.to(device)

            if use_amp:
                with torch.amp.autocast('cuda'):
                    outputs = model(images)
            else:
                outputs = model(images)

            pred_probs = torch.sigmoid(outputs)
            pred_masks = (pred_probs > 0.5).float()

            for i in range(images.shape[0]):
                gidx = batch_idx * bs + i
                orig_path = paths[gidx] if paths and gidx < len(paths) else None
                if split_label is not None:
                    source_id = f"{split_label}_{gidx:04d}"
                elif paths and gidx < len(paths):
                    source_id = _source_id_from_path(paths[gidx])
                else:
                    source_id = f"idx{gidx}"

                batch_metrics = calculate_metrics(
                    outputs[i:i+1].cpu(),
                    masks[i:i+1].cpu()
                )

                all_results.append({
                    'image': images[i].cpu(),
                    'gt_mask': masks[i].cpu(),
                    'pred_mask': pred_masks[i].cpu(),
                    'iou': batch_metrics['iou'],
                    'source_id': source_id,
                    'orig_path': orig_path
                })

            del images, masks, outputs, pred_probs, pred_masks
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return all_results


def create_overlay(image_tensor, gt_mask_tensor, pred_mask_tensor, alpha=0.5):
    """
    創建 overlay 視覺化
    
    參數:
        image_tensor: [C, H, W], 值範圍 [0, 1]
        gt_mask_tensor: [1, H, W], 值範圍 [0, 1]
        pred_mask_tensor: [1, H, W], 值範圍 [0, 1]
        alpha: 透明度
    
    返回:
        PIL Image
    """
    # 轉換為 numpy
    image_np = image_tensor.permute(1, 2, 0).numpy()  # [H, W, C]
    image_np = (image_np * 255).clip(0, 255).astype(np.uint8)
    
    gt_mask_np = gt_mask_tensor.squeeze(0).numpy()  # [H, W]
    pred_mask_np = pred_mask_tensor.squeeze(0).numpy()  # [H, W]
    
    # 創建 overlay
    overlay = image_np.copy().astype(np.float32)
    
    # GT mask: 藍色 (0, 100, 255)
    gt_overlay = np.zeros_like(overlay)
    gt_overlay[:, :, 0] = 0
    gt_overlay[:, :, 1] = 100
    gt_overlay[:, :, 2] = 255
    gt_mask_3d = np.stack([gt_mask_np] * 3, axis=-1)
    overlay = overlay * (1 - gt_mask_3d * alpha) + gt_overlay * (gt_mask_3d * alpha)
    
    # Pred mask: 紅色 (255, 50, 50)
    pred_overlay = np.zeros_like(overlay)
    pred_overlay[:, :, 0] = 255
    pred_overlay[:, :, 1] = 50
    pred_overlay[:, :, 2] = 50
    pred_mask_3d = np.stack([pred_mask_np] * 3, axis=-1)
    overlay = overlay * (1 - pred_mask_3d * alpha * 0.7) + pred_overlay * (pred_mask_3d * alpha * 0.7)
    
    overlay = overlay.clip(0, 255).astype(np.uint8)
    return Image.fromarray(overlay)


def _safe_overlay_suffix(source_id: str) -> str:
    """從 source_id 產生可放進 overlay 檔名的安全後綴（不含副檔名，已清理特殊字元）。"""
    base = os.path.splitext(str(source_id))[0]
    safe = ''.join(c if (c.isalnum() or c in '_.-') else '_' for c in base)
    return safe[:80] if len(safe) > 80 else safe


def save_val_overlays(model, val_loader, device, output_dir, use_amp=False):
    """
    生成並儲存 val set 的 overlay（最好和最差各 5 張），以及 best/worst 5 的 IoU、FP、FN 等。
    - 檔名使用 split 編碼（val_0042），對照 val 清單第 42 筆即為原圖。
    """
    print("生成 Val Overlay 視覺化...")

    all_results = evaluate_with_predictions(model, val_loader, device, use_amp, split_label='val')
    _smooth = 1e-6
    for r in all_results:
        p = (r['pred_mask'] > 0.5).float()
        g = (r['gt_mask'] > 0.5).float()
        tn = (g == 0).sum().item()
        tp = (g == 1).sum().item()
        r['fp_rate'] = ((p == 1) & (g == 0)).sum().item() / (tn + _smooth)
        r['fn_rate'] = ((p == 0) & (g == 1)).sum().item() / (tp + _smooth)

    all_results.sort(key=lambda x: x['iou'], reverse=True)
    best_5 = all_results[:5]
    worst_5 = all_results[-5:]

    os.makedirs(output_dir, exist_ok=True)

    def _save(bunch, prefix):
        for i, r in enumerate(bunch):
            overlay = create_overlay(r['image'], r['gt_mask'], r['pred_mask'])
            suf = _safe_overlay_suffix(r.get('source_id', ''))
            name = f'{prefix}_{i+1}_iou_{r["iou"]:.4f}_{suf}.png' if suf else f'{prefix}_{i+1}_iou_{r["iou"]:.4f}.png'
            overlay.save(os.path.join(output_dir, name))

    print("  儲存最好的 5 張...")
    _save(best_5, 'best')
    print("  儲存最差的 5 張...")
    _save(worst_5, 'worst')

    def _to_records(bunch):
        return [
            {
                'rank': i + 1,
                'source_id': r.get('source_id'),
                'iou': round(float(r['iou']), 6),
                'fp_rate': round(float(r.get('fp_rate', 0)), 6),
                'fn_rate': round(float(r.get('fn_rate', 0)), 6),
                'orig_path': r.get('orig_path'),
            }
            for i, r in enumerate(bunch)
        ]

    metrics = {'best_5': _to_records(best_5), 'worst_5': _to_records(worst_5)}
    with open(os.path.join(output_dir, 'val_best_worst_metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"  已寫入 val_best_worst_metrics.json（best/worst 5 的 IoU、FP、FN、orig_path）")

    print(f"  Overlay 已儲存至: {output_dir}")


def save_checkpoint(model, optimizer, epoch, val_iou, checkpoint_dir, is_best=False):
    """儲存檢查點"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_iou': val_iou
    }
    
    # 儲存最新檢查點
    latest_path = os.path.join(checkpoint_dir, 'latest.pth')
    torch.save(checkpoint, latest_path)
    
    # 儲存最佳檢查點
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'best.pth')
        torch.save(checkpoint, best_path)
        print(f"    [保存最佳模型] val_iou={val_iou:.4f}")


def main():
    print("=" * 60)
    print("  Multi-Camera U-Net Training")
    print("=" * 60)
    print()
    
    # === 設定 ===
    json_file = 'outputs/multi_camera_splits.json'
    epochs = 6
    batch_size = 2
    learning_rate = 1e-4
    image_size = (160, 160)  # 改為 160
    
    # 檢查設備和 AMP（嘗試使用 GPU，如果記憶體不足則使用 CPU）
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        # 嘗試使用 GPU，但設置較小的記憶體分配
        torch.cuda.empty_cache()
        device = torch.device('cuda')
        use_amp = True
        print(f"使用 GPU: {torch.cuda.get_device_name(0)} ({gpu_memory:.1f}GB)")
    else:
        device = torch.device('cpu')
        use_amp = False
        print("使用 CPU")
    
    # 啟動資訊
    print("訓練設定：")
    print(f"  batch_size: {batch_size}")
    print(f"  img_size: {image_size}")
    print(f"  amp: {'on' if use_amp else 'off'}")
    print(f"  cuda available: {use_cuda}")
    if use_cuda:
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  GPU Memory: {gpu_memory:.1f}GB")
    print()
    
    # === 載入 splits ===
    print("載入資料集 splits...")
    train_list = load_splits_from_json(json_file, 'train')
    sanity_list = load_splits_from_json(json_file, 'sanity')
    val_list = load_splits_from_json(json_file, 'val')
    
    print(f"  Train:  {len(train_list)} samples")
    print(f"  Sanity: {len(sanity_list)} samples")
    print(f"  Val:    {len(val_list)} samples")
    print()
    
    # === 創建 DataLoaders ===
    print("創建 DataLoaders...")
    train_loader = get_dataloader(
        split_list=train_list,
        batch_size=batch_size,
        shuffle=True,
        transform=False,  # 不加 augmentation
        image_size=image_size,
        num_workers=0
    )
    
    sanity_loader = get_dataloader(
        split_list=sanity_list,
        batch_size=batch_size,
        shuffle=False,
        transform=False,  # 不加 augmentation
        image_size=image_size,
        num_workers=0
    )
    
    val_loader = get_dataloader(
        split_list=val_list,
        batch_size=batch_size,
        shuffle=False,
        transform=False,  # 不加 augmentation
        image_size=image_size,
        num_workers=0
    )
    
    print(f"  Train batches:  {len(train_loader)}")
    print(f"  Sanity batches: {len(sanity_loader)}")
    print(f"  Val batches:    {len(val_loader)}")
    print()
    
    # === 創建模型 ===
    print("創建模型...")
    model = create_unet_model(n_channels=3, n_classes=1)
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  模型參數總數: {total_params:,}")
    print()
    
    # === 定義損失函數和優化器 ===
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    print(f"損失函數: BCEWithLogitsLoss")
    print(f"優化器: Adam (lr={learning_rate})")
    print()
    
    # === 創建輸出目錄 ===
    # 檢查是否有未完成的訓練目錄
    outputs_dir = Path('outputs')
    existing_dirs = list(outputs_dir.glob('train_multi_camera_*'))
    use_existing = False
    run_dir = None
    
    if existing_dirs:
        # 找到最新的目錄
        latest_dir = max(existing_dirs, key=os.path.getmtime)
        log_file_path = latest_dir / 'training_log.csv'
        
        # 檢查是否已完成
        if log_file_path.exists():
            with open(log_file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                completed_epochs = len(lines) - 1  # 減去 header
            
            if completed_epochs < epochs:
                # 使用現有目錄繼續訓練
                run_dir = str(latest_dir)
                use_existing = True
                print(f"[繼續訓練] 使用現有目錄: {latest_dir.name}")
                print(f"  已完成: {completed_epochs}/{epochs} epochs")
                print()
    
    # 如果沒有未完成的訓練，創建新目錄
    if not use_existing:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join('outputs', f'train_multi_camera_{timestamp}')
        print(f"[新訓練] 創建新目錄: {os.path.basename(run_dir)}")
        print()
    
    checkpoint_dir = os.path.join(run_dir, 'checkpoints')
    log_file = os.path.join(run_dir, 'training_log.csv')
    overlay_dir = os.path.join(run_dir, 'val_overlays')
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(overlay_dir, exist_ok=True)
    
    print(f"輸出目錄: {run_dir}")
    print(f"  檢查點: {checkpoint_dir}")
    print(f"  訓練日誌: {log_file}")
    print(f"  Overlay: {overlay_dir}")
    print()
    
    # === 初始化訓練日誌 ===
    log_headers = ['epoch', 'train_loss', 'train_iou', 'sanity_iou', 'val_loss', 'val_iou', 'val_dice', 'val_pixel_acc', 'val_fp_rate', 'val_fn_rate']
    
    # 檢查是否繼續訓練
    start_epoch = 1
    best_val_iou = 0.0
    
    if use_existing and os.path.exists(log_file):
        # 讀取現有日誌，確定從哪裡繼續
        with open(log_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            if rows:
                start_epoch = len(rows) + 1
                # 找到最佳 val_iou
                best_val_iou = max([float(row['val_iou']) for row in rows])
                print(f"[繼續訓練] 從 Epoch {start_epoch} 開始")
                print(f"  目前最佳 Val IoU: {best_val_iou:.4f}")
                print()
                
                # 嘗試載入最新檢查點
                latest_checkpoint = os.path.join(checkpoint_dir, 'latest.pth')
                if os.path.exists(latest_checkpoint):
                    print(f"[載入檢查點] {latest_checkpoint}")
                    checkpoint = torch.load(latest_checkpoint, map_location=device)
                    model.load_state_dict(checkpoint['model_state_dict'])
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    print(f"  已載入 Epoch {checkpoint.get('epoch', 'N/A')} 的模型")
                    print()
            else:
                # 日誌文件存在但為空，重新開始
                with open(log_file, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(log_headers)
    else:
        # 新訓練，創建日誌文件
        with open(log_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(log_headers)
    
    # === 訓練迴圈 ===
    print("=" * 60)
    print("開始訓練...")
    print("=" * 60)
    print()

    legacy_log = False
    if os.path.exists(log_file):
        with open(log_file, 'r', encoding='utf-8') as f:
            legacy_log = 'val_fp_rate' not in (f.readline() or '')
    last_val_fp_rate, last_val_fn_rate = None, None
    for epoch in range(start_epoch, epochs + 1):
        print(f"Epoch {epoch}/{epochs}")
        print("-" * 60)
        
        # 訓練
        train_loss, train_iou = train_epoch(model, train_loader, criterion, optimizer, device, use_amp)
        
        # 評估 sanity set
        _, sanity_metrics = evaluate(model, sanity_loader, criterion, device, 'Sanity', use_amp)
        sanity_iou = sanity_metrics['iou']
        
        # 評估 val set
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device, 'Val', use_amp)
        last_val_fp_rate = val_metrics.get('fp_rate', 0.0)
        last_val_fn_rate = val_metrics.get('fn_rate', 0.0)

        # 輸出結果（立即刷新）
        print(f"  Train Loss:   {train_loss:.4f}", flush=True)
        print(f"  Train IoU:    {train_iou:.4f}", flush=True)
        print(f"  Sanity IoU:   {sanity_iou:.4f}", flush=True)
        print(f"  Val Loss:     {val_loss:.4f}", flush=True)
        print(f"  Val IoU:      {val_metrics['iou']:.4f}", flush=True)
        print(f"  Val Dice:     {val_metrics['dice']:.4f}", flush=True)
        print(f"  Val PixelAcc: {val_metrics['pixel_acc']:.4f}", flush=True)
        print(f"  Val FP Rate:  {last_val_fp_rate:.4f}  (非天空誤判為天空)", flush=True)
        print(f"  Val FN Rate:  {last_val_fn_rate:.4f}  (天空漏檢)", flush=True)

        # 記錄到 CSV
        with open(log_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            row = [
                epoch,
                f"{train_loss:.6f}",
                f"{train_iou:.6f}",
                f"{sanity_iou:.6f}",
                f"{val_loss:.6f}",
                f"{val_metrics['iou']:.6f}",
                f"{val_metrics['dice']:.6f}",
                f"{val_metrics['pixel_acc']:.6f}",
            ]
            if not legacy_log:
                row += [f"{last_val_fp_rate:.6f}", f"{last_val_fn_rate:.6f}"]
            writer.writerow(row)
        
        # 儲存檢查點
        is_best = val_metrics['iou'] > best_val_iou
        if is_best:
            best_val_iou = val_metrics['iou']
        
        save_checkpoint(model, optimizer, epoch, val_metrics['iou'], checkpoint_dir, is_best=is_best)
        
        print()
    
    print("=" * 60)
    print("訓練完成！")
    print("=" * 60)
    print(f"最佳 Val IoU:  {best_val_iou:.4f}")
    if last_val_fp_rate is not None and last_val_fn_rate is not None:
        print(f"最後 Epoch Val FP Rate: {last_val_fp_rate:.4f}  (非天空誤判為天空)")
        print(f"最後 Epoch Val FN Rate: {last_val_fn_rate:.4f}  (天空漏檢)")
    print(f"檢查點目錄: {checkpoint_dir}")
    print(f"訓練日誌: {log_file}")
    print()

    # === 生成 Val Overlay ===
    print("=" * 60)
    print("生成 Val Overlay 視覺化...")
    print("=" * 60)
    print()
    
    # 載入最佳模型
    best_checkpoint_path = os.path.join(checkpoint_dir, 'best.pth')
    if os.path.exists(best_checkpoint_path):
        print(f"載入最佳模型: {best_checkpoint_path}")
        checkpoint = torch.load(best_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  最佳模型 Val IoU: {checkpoint.get('val_iou', 'N/A'):.4f}")
        print()
    
    save_val_overlays(model, val_loader, device, overlay_dir, use_amp)
    
    print()
    print("=" * 60)
    print("完成！")
    print("=" * 60)
    print(f"訓練日誌: {log_file}")
    print(f"Val Overlays: {overlay_dir}")
    print()


if __name__ == '__main__':
    main()
