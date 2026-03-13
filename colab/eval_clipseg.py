"""
CLIPSeg 評估腳本（Colab 專屬版本，參數化）

使用固定 test_list.txt 評估 CLIPSeg
所有路徑和參數都可通過 CLI 指定
支援 overlay 輸出

CLIPSeg 是 text-conditioned segmentation model，使用文字 prompt "sky" 進行分割
Zero-shot 方法，不需要訓練

輸出：
- {output_csv}：metrics 摘要 CSV
- {overlay_dir}/*.png：overlay 視覺化影像（如果指定）
"""

import os
import sys
import csv
import argparse
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

from utils_colab import setup_device, detect_environment
from eval_utils import create_overlay, save_overlay

try:
    import torch
    from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
    CLIPSEG_AVAILABLE = True
except ImportError as e:
    CLIPSEG_AVAILABLE = False
    print(f"[WARNING] CLIPSeg 未安裝: {e}")
    print("請執行: pip install transformers timm")


def load_list_file(list_file: str, base_data_dir: str = "data"):
    """從 list 檔案載入資料"""
    split_list = []
    
    if not os.path.exists(list_file):
        raise FileNotFoundError(f"找不到 list 檔案：{list_file}")
    
    with open(list_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
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
                'image_path': line
            })
    
    return split_list


def load_gt_mask(image_path, mask_path):
    """
    載入 GT mask
    
    返回:
        gt_mask: numpy array (H, W) 0-1 範圍，或 None
        has_gt: bool，是否有 GT
    """
    if not os.path.exists(mask_path):
        return None, False
    
    try:
        mask_img = Image.open(mask_path).convert('L')  # 轉為灰階
        mask_array = np.array(mask_img)
        
        # 轉換為二值化 mask
        if len(mask_array.shape) == 3:
            mask_array = mask_array[:, :, 0]
        
        # 處理不同格式的 mask：
        # 1. 布林值（True/False）：True 代表天空
        # 2. 數值（0-255）：需要正規化到 0-1，> 0.5 視為天空
        if mask_array.dtype == bool:
            # 布林值：True = 天空
            gt_mask = mask_array.astype(np.float32)
        else:
            # 數值：正規化到 0-1，> 0.5 視為天空
            if mask_array.max() > 1.0:
                mask_array = mask_array.astype(np.float32) / 255.0
            else:
                mask_array = mask_array.astype(np.float32)
            gt_mask = (mask_array > 0.5).astype(np.float32)
        
        # 檢查是否有天空
        has_gt = (gt_mask.sum() > 0)
        
        return gt_mask, has_gt
    except Exception as e:
        print(f"  警告：無法載入 GT mask {mask_path}: {e}")
        import traceback
        traceback.print_exc()
        return None, False


def calculate_metrics_per_image(pred_mask, gt_mask=None, has_gt=True):
    """
    計算單張影像的 metrics
    
    參數:
        pred_mask: numpy array (H, W) 0-1 範圍
        gt_mask: numpy array (H, W) 0-1 範圍，或 None
        has_gt: bool，是否有 GT
    
    返回:
        Dict: {'iou': float, 'fp_rate': float, 'fn_rate': float}
    """
    smooth = 1e-6
    
    if not has_gt or gt_mask is None:
        # No-sky case：只計算 FP rate
        fp = pred_mask.sum()
        total = pred_mask.size
        fp_rate = fp / (total + smooth)
        return {
            'iou': None,
            'fp_rate': fp_rate,
            'fn_rate': None
        }
    
    # 確保尺寸一致
    if pred_mask.shape != gt_mask.shape:
        pred_img = Image.fromarray((pred_mask * 255).astype(np.uint8))
        pred_img = pred_img.resize((gt_mask.shape[1], gt_mask.shape[0]), Image.NEAREST)
        pred_mask = np.array(pred_img, dtype=np.float32) / 255.0
    
    # 二值化
    pred_b = (pred_mask > 0.5).astype(np.float32)
    gt_b = gt_mask.astype(np.float32)
    
    # 計算 metrics
    intersection = (pred_b * gt_b).sum()
    union = pred_b.sum() + gt_b.sum() - intersection
    iou = intersection / (union + smooth)
    
    # FP/FN rate
    fp = ((pred_b == 1) & (gt_b == 0)).sum()
    fn = ((pred_b == 0) & (gt_b == 1)).sum()
    tn = ((pred_b == 0) & (gt_b == 0)).sum()
    tp = ((pred_b == 1) & (gt_b == 1)).sum()
    
    fp_rate = fp / (tn + smooth) if tn > 0 else 0.0
    fn_rate = fn / (tp + smooth) if tp > 0 else 0.0
    
    return {
        'iou': float(iou),
        'fp_rate': float(fp_rate),
        'fn_rate': float(fn_rate)
    }


def load_clipseg_model(model_id='CIDAS/clipseg-rd64-refined', device='cpu'):
    """載入 CLIPSeg 模型與 processor"""
    processor = CLIPSegProcessor.from_pretrained(model_id)
    model = CLIPSegForImageSegmentation.from_pretrained(model_id)
    model.to(device)
    model.eval()
    return processor, model


def predict_clipseg(processor, model, image_path, text_prompts=None, device='cpu', threshold=0.5):
    """
    單張圖用 CLIPSeg 預測 sky mask
    
    參數:
        processor: CLIPSegProcessor
        model: CLIPSegForImageSegmentation
        image_path: 影像路徑
        text_prompts: 文字提示列表（預設：['sky']）
        device: 設備
        threshold: 二值化閾值
    
    返回:
        pred_mask: numpy array (H, W) 0-1 範圍
    """
    if text_prompts is None:
        text_prompts = ['sky']
    
    image = Image.open(image_path).convert('RGB')
    image_np = np.array(image)
    orig_h, orig_w = image_np.shape[:2]

    inputs = processor(
        text=text_prompts,
        images=image,
        return_tensors='pt',
        padding=True,
    )
    inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    # logits: (batch, num_prompts, H, W) 或 (batch, H, W)
    logits = outputs.logits
    if logits.dim() == 4:
        logits = logits.mean(dim=1)
    logits = logits.cpu().float().numpy()
    logits = np.squeeze(logits)
    if logits.ndim != 2:
        logits = logits[0]

    # sigmoid
    pred = 1.0 / (1.0 + np.exp(-logits))
    pred = (pred > threshold).astype(np.float32)

    # resize 回原圖尺寸
    if pred.shape[0] != orig_h or pred.shape[1] != orig_w:
        pil = Image.fromarray((pred * 255).astype(np.uint8))
        pil = pil.resize((orig_w, orig_h), Image.NEAREST)
        pred = np.array(pil, dtype=np.float32) / 255.0

    return pred


def evaluate_per_image_clipseg(processor, model, test_list, device='cpu', overlay_dir=None, data_dir="data"):
    """
    使用 CLIPSeg 評估每張影像
    
    返回:
        List[Dict]: 每筆包含 image_path, camera_id, iou, fp_rate, fn_rate, has_gt, overlay_path
    """
    results = []
    
    for item in tqdm(test_list, desc='  Evaluating CLIPSeg'):
        image_path_rel = item['image_path']
        camera_id = item['camera_id']
        image_filename = item['image']
        mask_filename = item['mask']
        
        # 構建完整路徑
        camera_folder = f"skyfinder_{camera_id}"
        full_image_path = os.path.join(data_dir, camera_folder, "images", image_filename)
        full_mask_path = os.path.join(data_dir, camera_folder, "masks", mask_filename)
        
        if not os.path.exists(full_image_path):
            print(f"  警告：找不到影像：{full_image_path}")
            continue
        
        # CLIPSeg 推論
        try:
            pred_mask = predict_clipseg(processor, model, full_image_path, text_prompts=['sky'], device=device)
        except Exception as e:
            print(f"  錯誤：CLIPSeg 推論失敗 {full_image_path}: {e}")
            continue
        
        # 載入 GT mask
        gt_mask, has_gt = load_gt_mask(full_image_path, full_mask_path)
        
        # 計算 metrics
        if has_gt and gt_mask is not None:
            metrics = calculate_metrics_per_image(pred_mask, gt_mask, has_gt=True)
            iou = metrics.get('iou')
            fp_rate = metrics.get('fp_rate', 0.0)
            fn_rate = metrics.get('fn_rate')
        else:
            # No-sky case
            metrics = calculate_metrics_per_image(pred_mask, None, has_gt=False)
            iou = None
            fp_rate = metrics.get('fp_rate', 0.0)
            fn_rate = None
        
        # 生成 overlay（如果指定）
        overlay_path = None
        if overlay_dir:
            try:
                overlay_img = create_overlay(full_image_path, pred_mask, gt_mask)
                overlay_filename = f"{camera_id}_{os.path.splitext(image_filename)[0]}_clipseg_overlay.png"
                overlay_path = os.path.join(overlay_dir, overlay_filename)
                save_overlay(overlay_img, overlay_path)
            except Exception as e:
                print(f"  警告：無法生成 overlay {full_image_path}: {e}")
        
        results.append({
            'image_path': image_path_rel,
            'camera_id': camera_id,
            'iou': iou,
            'fp_rate': fp_rate,
            'fn_rate': fn_rate,
            'has_gt': has_gt,
            'overlay_path': overlay_path
        })
    
    return results


def aggregate_per_camera(results):
    """依 camera_id 聚合 metrics"""
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
            if r['iou'] is not None:
                stats['iou_sum'] += r['iou']
            if r['fn_rate'] is not None:
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
    parser = argparse.ArgumentParser(description='評估 CLIPSeg（Colab 專屬版本，參數化）')
    
    # 路徑參數
    parser.add_argument('--data_dir', type=str, default='data',
                        help='數據根目錄（預設：data）')
    parser.add_argument('--outputs_dir', type=str, default='outputs',
                        help='輸出目錄（預設：outputs）')
    parser.add_argument('--test_list', type=str, default=None,
                        help='test_list.txt 路徑（預設：{outputs_dir}/test_list.txt）')
    parser.add_argument('--output_csv', type=str, default=None,
                        help='輸出 CSV 路徑（預設：{outputs_dir}/metrics_summary.csv）')
    parser.add_argument('--overlay_dir', type=str, default=None,
                        help='overlay 輸出目錄（預設：不輸出 overlay；指定則輸出到該目錄）')
    
    # 模型參數
    parser.add_argument('--model_id', type=str, default='CIDAS/clipseg-rd64-refined',
                        help='CLIPSeg 模型 ID（預設：CIDAS/clipseg-rd64-refined）')
    parser.add_argument('--text_prompts', type=str, nargs='+', default=['sky'],
                        help='文字提示（預設：sky）')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='二值化閾值（預設：0.5）')
    
    # 評估參數
    parser.add_argument('--device', type=str, default=None,
                        help='設備（cpu/cuda/auto，預設：自動偵測環境）')
    
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
    
    print("=" * 60)
    print("  評估 CLIPSeg（Colab 專屬版本，參數化）")
    print("=" * 60)
    print()
    print(f"環境: {env}")
    print(f"設備: {device}")
    print(f"模型 ID: {args.model_id}")
    print(f"文字提示: {args.text_prompts}")
    if args.overlay_dir:
        print(f"Overlay 輸出目錄: {args.overlay_dir}")
    print()
    
    # === 檢查 CLIPSeg 可用性 ===
    if not CLIPSEG_AVAILABLE:
        print("[錯誤] CLIPSeg 未安裝或不可用")
        print("請執行: pip install transformers timm")
        sys.exit(1)
    
    # === 載入 test list ===
    print(f"載入 test list: {args.test_list}...")
    if not os.path.exists(args.test_list):
        print(f"[錯誤] 找不到 test_list.txt: {args.test_list}")
        sys.exit(1)
    
    test_split_list = load_list_file(args.test_list, base_data_dir=args.data_dir)
    print(f"  共 {len(test_split_list)} 張測試影像")
    print()
    
    # === 載入 CLIPSeg 模型 ===
    print("載入 CLIPSeg 模型...")
    try:
        processor, model = load_clipseg_model(model_id=args.model_id, device=device)
        print(f"  CLIPSeg 模型載入成功")
    except Exception as e:
        print(f"[錯誤] 無法載入 CLIPSeg 模型: {e}")
        sys.exit(1)
    print()
    
    # === 建立 overlay 目錄 ===
    if args.overlay_dir:
        os.makedirs(args.overlay_dir, exist_ok=True)
    
    # === 評估 ===
    print("開始評估...")
    results = evaluate_per_image_clipseg(
        processor, model, test_split_list,
        device=device,
        overlay_dir=args.overlay_dir,
        data_dir=args.data_dir
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
            existing_rows = [row for row in reader if row.get('method') != 'CLIPSeg']
    
    # 準備新的 rows
    new_rows = []
    for camera_id in sorted(camera_metrics.keys()):
        metrics = camera_metrics[camera_id]
        row = {
            'method': 'CLIPSeg',
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
    print("  CLIPSeg 評估結果摘要")
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
