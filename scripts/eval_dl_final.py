"""
評估 DL baseline（使用固定 test_list.txt）

輸入：
- outputs/test_list.txt（固定，每行一個相對路徑）
- outputs/train_in_domain_*/checkpoints/best.pth

處理：
- 讀取 test_list.txt 的每張圖（相對路徑需加上 data/ 前綴載入）
- 用 DL checkpoint 推論
- 計算 IoU / FP / FN（與 GT mask 對齊）
- Per-camera 統計

輸出：
- 寫入 outputs/metrics_summary.csv（method='DL'）
- Per-camera metrics
"""

import os
import sys
import csv
import json
import argparse
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from PIL import Image

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

from models import create_unet_model
from utils.dataset import get_dataloader
from utils.metrics import calculate_metrics
from scripts.train_in_domain import load_list_file


def find_latest_dl_checkpoint():
    """找最新的 train_in_domain_* 下的 checkpoints/best.pth"""
    outputs_dir = Path('outputs')
    candidates = list(outputs_dir.glob('train_in_domain_*/checkpoints/best.pth'))
    if not candidates:
        return None
    return str(max(candidates, key=os.path.getmtime))


def create_overlay(image_path, pred_mask, gt_mask=None, alpha=0.5):
    """
    創建 overlay 視覺化
    
    參數:
        image_path: 原始影像路徑
        pred_mask: 預測 mask（numpy array，0-1 範圍）
        gt_mask: GT mask（numpy array，0-1 範圍），可選
        alpha: overlay 透明度（0-1）
    
    返回:
        PIL Image: overlay 影像
    """
    # 載入原始影像
    image = Image.open(image_path).convert('RGB')
    image_np = np.array(image, dtype=np.float32)
    
    # 確保 pred_mask 是 0-1 範圍
    if pred_mask.max() > 1.0:
        pred_mask = pred_mask.astype(np.float32) / 255.0
    else:
        pred_mask = pred_mask.astype(np.float32)
    
    # 調整 pred_mask 尺寸（如果需要）
    if pred_mask.shape[:2] != image_np.shape[:2]:
        pred_img = Image.fromarray((pred_mask * 255).astype(np.uint8))
        pred_img = pred_img.resize((image_np.shape[1], image_np.shape[0]), Image.NEAREST)
        pred_mask = np.array(pred_img, dtype=np.float32) / 255.0
    
    pred_b = (pred_mask > 0.5)
    overlay = image_np.copy()
    
    if gt_mask is not None:
        # 確保 gt_mask 是 0-1 範圍
        if gt_mask.max() > 1.0:
            gt_mask = gt_mask.astype(np.float32) / 255.0
        else:
            gt_mask = gt_mask.astype(np.float32)
        
        # 調整 gt_mask 尺寸（如果需要）
        if gt_mask.shape[:2] != image_np.shape[:2]:
            gt_img = Image.fromarray((gt_mask * 255).astype(np.uint8))
            gt_img = gt_img.resize((image_np.shape[1], image_np.shape[0]), Image.NEAREST)
            gt_mask = np.array(gt_img, dtype=np.float32) / 255.0
        
        gt_b = (gt_mask > 0.5)
        
        # TP: 綠色（預測正確的天空區域）
        # FP: 紅色（誤判為天空）
        # FN: 藍色（漏判的天空區域）
        fn_mask = gt_b & (~pred_b)  # False Negative
        fp_mask = pred_b & (~gt_b)  # False Positive
        tp_mask = gt_b & pred_b      # True Positive
        
        blue = np.array([0, 100, 255], dtype=np.float32)   # FN: 藍色
        red = np.array([255, 50, 50], dtype=np.float32)   # FP: 紅色
        green = np.array([50, 255, 50], dtype=np.float32) # TP: 綠色
        
        fn_3d = np.stack([fn_mask] * 3, axis=-1)
        fp_3d = np.stack([fp_mask] * 3, axis=-1)
        tp_3d = np.stack([tp_mask] * 3, axis=-1)
        
        # 先畫 TP（綠色，較淡）
        overlay = np.where(tp_3d, overlay * (1 - alpha * 0.3) + green * (alpha * 0.3), overlay)
        # 再畫 FN（藍色）
        overlay = np.where(fn_3d, overlay * (1 - alpha) + blue * alpha, overlay)
        # 最後畫 FP（紅色）
        overlay = np.where(fp_3d, overlay * (1 - alpha * 0.9) + red * (alpha * 0.9), overlay)
    else:
        # 沒有 GT：只顯示預測（綠色）
        pred_3d = np.stack([pred_b] * 3, axis=-1)
        green = np.array([50, 255, 50], dtype=np.float32)
        overlay = np.where(pred_3d, overlay * (1 - alpha * 0.5) + green * (alpha * 0.5), overlay)
    
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    overlay_img = Image.fromarray(overlay)
    
    return overlay_img


def evaluate_per_image(model, dataloader, device, use_amp=False, overlay_dir=None, image_size=(256, 256)):
    """
    評估每張影像，返回詳細結果
    
    返回:
        List[Dict]: 每筆包含 image_path, camera_id, iou, fp_rate, fn_rate, has_gt, overlay_path
    """
    model.eval()
    results = []
    smooth = 1e-6
    
    dataset = dataloader.dataset
    image_paths = getattr(dataset, 'image_paths', None)
    
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(tqdm(dataloader, desc='  Evaluating', leave=False)):
            images = images.to(device)
            masks = masks.to(device)
            
            if use_amp:
                with torch.amp.autocast('cuda'):
                    outputs = model(images)
            else:
                outputs = model(images)
            
            pred_probs = torch.sigmoid(outputs)
            pred_b = (pred_probs > 0.5).float()
            gt_b = (masks > 0.5).float()
            
            for i in range(images.shape[0]):
                global_idx = batch_idx * dataloader.batch_size + i
                
                # 取得 image_path 和 camera_id
                if image_paths and global_idx < len(image_paths):
                    image_path = image_paths[global_idx]
                    # 從路徑提取 camera_id
                    parts = image_path.replace('\\', '/').split('/')
                    camera_id = None
                    for part in parts:
                        if part.startswith('skyfinder_'):
                            camera_id = part.replace('skyfinder_', '')
                            break
                else:
                    image_path = f"unknown_{global_idx}"
                    camera_id = "unknown"
                
                # 計算 metrics
                batch_metrics = calculate_metrics(
                    outputs[i:i+1].cpu(),
                    masks[i:i+1].cpu()
                )
                
                # FP/FN rate
                p = pred_b[i]
                g = gt_b[i]
                fp = ((p == 1) & (g == 0)).sum().item()
                fn = ((p == 0) & (g == 1)).sum().item()
                tn = (g == 0).sum().item()
                tp = (g == 1).sum().item()
                
                fp_rate = fp / (tn + smooth) if tn > 0 else 0.0
                fn_rate = fn / (tp + smooth) if tp > 0 else 0.0
                
                # 檢查是否有 GT（是否有天空）
                has_gt = (g.sum().item() > 0)
                
                # 生成 overlay（如果指定）
                overlay_path = None
                if overlay_dir:
                    try:
                        # 取得 pred_mask 和 gt_mask（需要 resize 回原圖尺寸）
                        pred_mask_np = pred_probs[i].cpu().numpy().squeeze()
                        gt_mask_np = masks[i].cpu().numpy().squeeze()
                        
                        # Resize 回原圖尺寸
                        orig_image = Image.open(image_path).convert('RGB')
                        orig_h, orig_w = orig_image.size[1], orig_image.size[0]
                        
                        pred_img = Image.fromarray((pred_mask_np * 255).astype(np.uint8))
                        pred_img = pred_img.resize((orig_w, orig_h), Image.NEAREST)
                        pred_mask_resized = np.array(pred_img, dtype=np.float32) / 255.0
                        
                        gt_img = Image.fromarray((gt_mask_np * 255).astype(np.uint8))
                        gt_img = gt_img.resize((orig_w, orig_h), Image.NEAREST)
                        gt_mask_resized = np.array(gt_img, dtype=np.float32) / 255.0
                        
                        overlay_img = create_overlay(image_path, pred_mask_resized, gt_mask_resized)
                        
                        # 生成 overlay 檔名
                        image_filename = os.path.basename(image_path)
                        image_base = os.path.splitext(image_filename)[0]
                        overlay_filename = f"{camera_id}_{image_base}_dl_overlay.png"
                        overlay_path = os.path.join(overlay_dir, overlay_filename)
                        
                        os.makedirs(overlay_dir, exist_ok=True)
                        overlay_img.save(overlay_path)
                    except Exception as e:
                        print(f"  警告：無法生成 overlay {image_path}: {e}")
                
                results.append({
                    'image_path': image_path,
                    'camera_id': camera_id,
                    'iou': batch_metrics['iou'],
                    'fp_rate': fp_rate,
                    'fn_rate': fn_rate,
                    'has_gt': has_gt,
                    'overlay_path': overlay_path
                })
            
            del images, masks, outputs, pred_probs, pred_b, gt_b
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    return results


def aggregate_per_camera(results):
    """
    依 camera_id 聚合 metrics
    
    返回:
        Dict[camera_id, Dict]: 每個 camera 的統計資訊
    """
    camera_stats = defaultdict(lambda: {
        'images': [],
        'num_images': 0,
        'iou_sum': 0.0,
        'fp_rate_sum': 0.0,
        'fn_rate_sum': 0.0,
        'num_with_sky': 0,  # 有 GT sky 的影像數
        'num_no_sky': 0,    # 沒有 GT sky 的影像數
        'fp_rate_nosky_sum': 0.0  # no-sky 影像的 FP rate 總和
    })
    
    for r in results:
        camera_id = r['camera_id']
        stats = camera_stats[camera_id]
        
        stats['images'].append(r)
        stats['num_images'] += 1
        stats['fp_rate_sum'] += r['fp_rate']
        
        if r['has_gt']:
            # 有 GT sky
            stats['num_with_sky'] += 1
            stats['iou_sum'] += r['iou']
            stats['fn_rate_sum'] += r['fn_rate']
        else:
            # 沒有 GT sky（no-sky camera）
            stats['num_no_sky'] += 1
            stats['fp_rate_nosky_sum'] += r['fp_rate']
    
    # 計算平均值
    aggregated = {}
    for camera_id, stats in camera_stats.items():
        num_images = stats['num_images']
        num_with_sky = stats['num_with_sky']
        num_no_sky = stats['num_no_sky']
        
        aggregated[camera_id] = {
            'num_images': num_images,
            'iou_mean': stats['iou_sum'] / num_with_sky if num_with_sky > 0 else None,
            'fp_mean': stats['fp_rate_sum'] / num_images,
            'fn_mean': stats['fn_rate_sum'] / num_with_sky if num_with_sky > 0 else None,
            'fp_rate_nosky': stats['fp_rate_nosky_sum'] / num_no_sky if num_no_sky > 0 else None
        }
    
    return aggregated


def main():
    parser = argparse.ArgumentParser(description='評估 DL baseline（使用固定 test_list.txt）')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='best.pth 路徑；未指定則用最新的 train_in_domain_*/checkpoints/best.pth')
    parser.add_argument('--test_list', type=str, default='outputs/test_list.txt',
                        help='test_list.txt 路徑')
    parser.add_argument('--image_size', type=int, nargs=2, default=[256, 256],
                        help='與訓練時一致的 image_size，預設 256 256')
    parser.add_argument('--output_csv', type=str, default='outputs/metrics_summary.csv',
                        help='輸出 CSV 路徑')
    parser.add_argument('--overlay_dir', type=str, default='outputs/dl_overlays',
                        help='overlay 輸出目錄（預設：outputs/dl_overlays）')
    args = parser.parse_args()
    
    image_size = tuple(args.image_size)
    test_list_file = args.test_list
    
    print("=" * 60)
    print("  評估 DL baseline（使用固定 test_list.txt）")
    print("=" * 60)
    print()
    
    # === 解析 checkpoint ===
    if args.checkpoint:
        best_path = args.checkpoint
        if not os.path.isfile(best_path):
            print(f"[錯誤] 找不到 checkpoint: {best_path}")
            sys.exit(1)
    else:
        best_path = find_latest_dl_checkpoint()
        if not best_path:
            print("[錯誤] 找不到任何 outputs/train_in_domain_*/checkpoints/best.pth")
            print("       請先完成訓練，或使用 --checkpoint 指定路徑")
            sys.exit(1)
    
    print(f"  Checkpoint:   {best_path}")
    print(f"  Test list:    {test_list_file}")
    print(f"  Image size:   {image_size}")
    print()
    
    # === 設備 ===
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    use_amp = use_cuda
    print(f"  裝置: {'GPU' if use_cuda else 'CPU'}, AMP: {'on' if use_amp else 'off'}")
    print()
    
    # === 載入 test list ===
    print(f"載入 test list: {test_list_file}...")
    if not os.path.exists(test_list_file):
        print(f"[錯誤] 找不到 test_list.txt: {test_list_file}")
        sys.exit(1)
    
    test_split_list = load_list_file(test_list_file)
    print(f"  共 {len(test_split_list)} 張測試影像")
    print()
    
    # === DataLoader ===
    print("建立 dataloader...")
    test_loader = get_dataloader(
        batch_size=4,
        shuffle=False,
        transform=False,
        image_size=image_size,
        num_workers=0,
        split_list=test_split_list,
        base_data_dir="data"
    )
    print(f"  Test batches: {len(test_loader)}")
    print()
    
    # === 模型與 checkpoint ===
    print("載入模型與 checkpoint...")
    model = create_unet_model(n_channels=3, n_classes=1)
    ck = torch.load(best_path, map_location=device)
    model.load_state_dict(ck['model_state_dict'])
    model = model.to(device)
    model.eval()
    print(f"  Epoch: {ck.get('epoch', 'N/A')}, Val IoU: {ck.get('val_iou', 'N/A'):.4f}" if isinstance(ck.get('val_iou'), (int, float)) else f"  Epoch: {ck.get('epoch', 'N/A')}")
    print()
    
    # === 評估 ===
    print("開始評估...")
    # === 建立 overlay 目錄 ===
    if args.overlay_dir:
        os.makedirs(args.overlay_dir, exist_ok=True)
    
    results = evaluate_per_image(model, test_loader, device, use_amp, 
                                  overlay_dir=args.overlay_dir, image_size=image_size)
    print(f"  完成，共評估 {len(results)} 張影像")
    if args.overlay_dir:
        overlay_count = sum(1 for r in results if r.get('overlay_path'))
        print(f"  生成 {overlay_count} 張 overlay")
    print()
    
    # === 聚合 per-camera ===
    print("聚合 per-camera metrics...")
    camera_metrics = aggregate_per_camera(results)
    print(f"  找到 {len(camera_metrics)} 個 camera")
    print()
    
    # === 輸出結果 ===
    print("輸出結果...")
    output_file = args.output_csv
    
    # 讀取現有的 CSV（如果存在）
    existing_rows = []
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            existing_rows = [row for row in reader if row.get('method') != 'DL']
    
    # 準備新的 rows
    new_rows = []
    for camera_id in sorted(camera_metrics.keys()):
        metrics = camera_metrics[camera_id]
        row = {
            'method': 'DL',
            'camera_id': camera_id,
            'num_images': str(metrics['num_images']),
            'IoU_mean': f"{metrics['iou_mean']:.6f}" if metrics['iou_mean'] is not None else '',
            'FP_mean': f"{metrics['fp_mean']:.6f}",
            'FN_mean': f"{metrics['fn_mean']:.6f}" if metrics['fn_mean'] is not None else '',
            'FP_rate_nosky': f"{metrics['fp_rate_nosky']:.6f}" if metrics['fp_rate_nosky'] is not None else ''
        }
        new_rows.append(row)
    
    # 合併並寫入
    all_rows = existing_rows + new_rows
    fieldnames = ['method', 'camera_id', 'num_images', 'IoU_mean', 'FP_mean', 'FN_mean', 'FP_rate_nosky']
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    
    print(f"  已寫入 {output_file}")
    print()
    
    # === 顯示摘要 ===
    print("=" * 60)
    print("  DL Baseline 評估結果摘要")
    print("=" * 60)
    print()
    for camera_id in sorted(camera_metrics.keys()):
        metrics = camera_metrics[camera_id]
        print(f"Camera {camera_id}:")
        print(f"  影像數: {metrics['num_images']}")
        if metrics['iou_mean'] is not None:
            print(f"  IoU: {metrics['iou_mean']:.4f}")
            print(f"  FN Rate: {metrics['fn_mean']:.4f}")
        print(f"  FP Rate: {metrics['fp_mean']:.4f}")
        if metrics['fp_rate_nosky'] is not None:
            print(f"  FP Rate (no-sky): {metrics['fp_rate_nosky']:.4f}")
        print()
    
    print("=" * 60)
    print("  評估完成")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
