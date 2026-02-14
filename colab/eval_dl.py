"""
DL baseline 評估腳本（Colab 專屬版本，參數化）

使用固定 test_list.txt 評估 DL baseline
所有路徑和參數都可通過 CLI 指定
支援 overlay 輸出

輸出：
- {output_csv}：metrics 摘要 CSV
- {overlay_dir}/*.png：overlay 視覺化影像（如果指定）
"""

import os
import sys
import csv
import argparse
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from PIL import Image

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# 添加專案根目錄到 Python 路徑
_script_dir = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_script_dir)
sys.path.insert(0, _root)
sys.path.insert(0, _script_dir)

from models import create_unet_model
from utils.dataset import get_dataloader
from utils.metrics import calculate_metrics
from utils_colab import setup_device, detect_environment
from eval_utils import create_overlay, save_overlay


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
            parts = line.replace('\\', '/').split('/')
            
            if len(parts) < 3:
                print(f"  警告：無法解析路徑：{line}")
                continue
            
            camera_folder = parts[0]
            if not camera_folder.startswith('skyfinder_'):
                print(f"  警告：無效的 camera 資料夾名稱：{camera_folder}")
                continue
            
            camera_id = camera_folder.replace('skyfinder_', '')
            image_filename = '/'.join(parts[2:])
            
            # 找到對應的 mask
            img_base = os.path.splitext(image_filename)[0]
            possible_mask_names = [
                f"{img_base}.png",
                f"{img_base}.pgm",
                f"{img_base}.jpg",
                image_filename,
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
                'mask': mask_filename,
                'image_path': line  # 相對路徑
            })
    
    return split_list


def find_latest_dl_checkpoint(outputs_dir: str):
    """找最新的 train_dl_* 下的 checkpoints/best.pth"""
    outputs_path = Path(outputs_dir)
    candidates = list(outputs_path.glob('train_dl_*/checkpoints/best.pth'))
    if not candidates:
        # 也嘗試舊的命名方式
        candidates = list(outputs_path.glob('train_in_domain_*/checkpoints/best.pth'))
    if not candidates:
        return None
    return str(max(candidates, key=os.path.getmtime))


def evaluate_per_image(model, dataloader, device, use_amp=False, overlay_dir=None, data_dir="data", split_list=None):
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
                if split_list and global_idx < len(split_list):
                    item = split_list[global_idx]
                    image_path = item['image_path']
                    camera_id = item['camera_id']
                    image_filename = item['image']
                    mask_filename = item['mask']
                elif image_paths and global_idx < len(image_paths):
                    image_path = image_paths[global_idx]
                    parts = image_path.replace('\\', '/').split('/')
                    camera_id = None
                    for part in parts:
                        if part.startswith('skyfinder_'):
                            camera_id = part.replace('skyfinder_', '')
                            break
                    if camera_id is None:
                        camera_id = "unknown"
                    image_filename = os.path.basename(image_path)
                    mask_filename = None
                else:
                    image_path = f"unknown_{global_idx}"
                    camera_id = "unknown"
                    image_filename = None
                    mask_filename = None
                
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
                
                # 轉換為 numpy 用於 overlay
                pred_np = pred_b[i].cpu().numpy().astype(np.float32)
                gt_np = gt_b[i].cpu().numpy().astype(np.float32)
                
                # 生成 overlay（如果指定）
                overlay_path = None
                if overlay_dir and image_filename:
                    # 構建完整影像路徑
                    parts = image_path.replace('\\', '/').split('/')
                    camera_folder = parts[0]
                    full_image_path = os.path.join(data_dir, camera_folder, "images", image_filename)
                    
                    if os.path.exists(full_image_path):
                        # 載入 GT mask（如果需要）
                        gt_mask_for_overlay = None
                        if mask_filename:
                            mask_path = os.path.join(data_dir, camera_folder, "masks", mask_filename)
                            if os.path.exists(mask_path):
                                gt_mask_img = Image.open(mask_path)
                                gt_mask_for_overlay = np.array(gt_mask_img)
                                if len(gt_mask_for_overlay.shape) == 3:
                                    gt_mask_for_overlay = gt_mask_for_overlay[:, :, 0]
                                gt_mask_for_overlay = (gt_mask_for_overlay > 128).astype(np.float32)
                        
                        # 生成 overlay
                        overlay_img = create_overlay(full_image_path, pred_np, gt_mask_for_overlay)
                        
                        # 保存 overlay
                        overlay_filename = f"{camera_id}_{os.path.splitext(image_filename)[0]}_overlay.png"
                        overlay_path = os.path.join(overlay_dir, overlay_filename)
                        save_overlay(overlay_img, overlay_path)
                
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
        'num_with_sky': 0,
        'num_no_sky': 0,
        'fp_rate_nosky_sum': 0.0
    })
    
    for r in results:
        camera_id = r['camera_id']
        stats = camera_stats[camera_id]
        
        stats['images'].append(r)
        stats['num_images'] += 1
        stats['fp_rate_sum'] += r['fp_rate']
        
        if r['has_gt']:
            stats['num_with_sky'] += 1
            stats['iou_sum'] += r['iou']
            stats['fn_rate_sum'] += r['fn_rate']
        else:
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
    parser = argparse.ArgumentParser(description='評估 DL baseline（Colab 專屬版本，參數化）')
    
    # 路徑參數
    parser.add_argument('--data_dir', type=str, default='data',
                        help='數據根目錄（預設：data）')
    parser.add_argument('--outputs_dir', type=str, default='outputs',
                        help='輸出目錄（預設：outputs）')
    parser.add_argument('--test_list', type=str, default=None,
                        help='test_list.txt 路徑（預設：{outputs_dir}/test_list.txt）')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='checkpoint 路徑；未指定則自動尋找最新的 train_dl_*/checkpoints/best.pth')
    parser.add_argument('--output_csv', type=str, default=None,
                        help='輸出 CSV 路徑（預設：{outputs_dir}/metrics_summary.csv）')
    parser.add_argument('--overlay_dir', type=str, default=None,
                        help='overlay 輸出目錄（預設：不輸出 overlay；指定則輸出到該目錄）')
    
    # 評估參數
    parser.add_argument('--image_size', type=int, nargs=2, default=[256, 256],
                        help='與訓練時一致的 image_size（預設：256 256）')
    parser.add_argument('--device', type=str, default=None,
                        help='設備（cpu/cuda/auto，預設：自動偵測環境）')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='批次大小（預設：4）')
    
    args = parser.parse_args()
    
    # 環境偵測
    env = detect_environment()
    
    # 設定預設路徑
    if args.test_list is None:
        args.test_list = os.path.join(args.outputs_dir, 'test_list.txt')
    if args.output_csv is None:
        args.output_csv = os.path.join(args.outputs_dir, 'metrics_summary.csv')
    
    # 設定 device
    device, use_amp, env = setup_device(args.device, env)
    
    # 訓練參數
    image_size = tuple(args.image_size)
    
    print("=" * 60)
    print("  評估 DL baseline（Colab 專屬版本，參數化）")
    print("=" * 60)
    print()
    print(f"環境: {env}")
    print(f"設備: {device}")
    print(f"影像尺寸: {image_size}")
    if args.overlay_dir:
        print(f"Overlay 輸出目錄: {args.overlay_dir}")
    print()
    
    # === 解析 checkpoint ===
    if args.checkpoint:
        best_path = args.checkpoint
        if not os.path.isfile(best_path):
            print(f"[錯誤] 找不到 checkpoint: {best_path}")
            sys.exit(1)
    else:
        best_path = find_latest_dl_checkpoint(args.outputs_dir)
        if not best_path:
            print(f"[錯誤] 找不到任何 {args.outputs_dir}/train_dl_*/checkpoints/best.pth")
            print("       請先完成訓練，或使用 --checkpoint 指定路徑")
            sys.exit(1)
    
    print(f"Checkpoint: {best_path}")
    print(f"Test list: {args.test_list}")
    print()
    
    # === 載入 test list ===
    print(f"載入 test list: {args.test_list}...")
    if not os.path.exists(args.test_list):
        print(f"[錯誤] 找不到 test_list.txt: {args.test_list}")
        sys.exit(1)
    
    test_split_list = load_list_file(args.test_list, base_data_dir=args.data_dir)
    print(f"  共 {len(test_split_list)} 張測試影像")
    print()
    
    # === DataLoader ===
    print("建立 dataloader...")
    test_loader = get_dataloader(
        batch_size=args.batch_size,
        shuffle=False,
        transform=False,
        image_size=image_size,
        num_workers=0,
        split_list=test_split_list,
        base_data_dir=args.data_dir
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
    
    # === 建立 overlay 目錄 ===
    if args.overlay_dir:
        os.makedirs(args.overlay_dir, exist_ok=True)
    
    # === 評估 ===
    print("開始評估...")
    results = evaluate_per_image(
        model, test_loader, device, use_amp,
        overlay_dir=args.overlay_dir,
        data_dir=args.data_dir,
        split_list=test_split_list
    )
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
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
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
