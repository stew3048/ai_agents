"""
評估 VLM → SAM pipeline（使用固定 test_list.txt）

輸入：
- outputs/test_list.txt（固定，每行一個相對路徑）
- SAM 2.0 模型（sam2_hiera_small）

處理：
- 讀取 test_list.txt 的每張圖（相對路徑需加上 data/ 前綴載入）
- 依情境（derived_subset_name）給 prompt（沿用 get_prompt_strategy）
- 計算 IoU / FP / FN（與 GT mask 對齊）
- Per-camera 統計

輸出：
- 追加到 outputs/metrics_summary.csv（method='SAM'）
"""

import os
import sys
import csv
import argparse
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

_script_dir = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_script_dir)
sys.path.insert(0, _root)
sys.path.insert(0, _script_dir)
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

from eval_sam2_failure_cases import (
    SAM2_AVAILABLE,
    load_sam2_model,
    get_prompt_strategy,
    predict_with_sam2,
    load_gt_mask,
    calculate_metrics_per_image,
)
from scripts.train_in_domain import load_list_file


def derive_subset_name_from_image(image_path, metadata=None):
    """
    從影像路徑推導 subset_name（簡化版，實際可能需要 metadata）
    這裡使用簡單的啟發式方法，或可以從 metadata 讀取
    """
    # 簡化：預設使用 default，實際應該根據影像內容判斷
    # 如果需要更準確，可以讀取 metadata.csv
    return 'default'


def resolve_image_path(relative_path, base_data_dir="data"):
    """將相對路徑轉換為絕對路徑"""
    full_path = os.path.join(base_data_dir, relative_path)
    if os.path.exists(full_path):
        return os.path.abspath(full_path)
    return None


def evaluate_per_image_sam(model, test_list, device='cpu', model_type='sam2_hiera_small'):
    """
    使用 SAM 評估每張影像
    
    返回:
        List[Dict]: 每筆包含 image_path, camera_id, iou, fp_rate, fn_rate, has_gt
    """
    results = []
    
    for item in tqdm(test_list, desc='  Evaluating SAM'):
        relative_path = item['image_path'] if isinstance(item, dict) else item
        camera_id = item.get('camera_id', 'unknown') if isinstance(item, dict) else 'unknown'
        
        # 解析相對路徑
        if isinstance(item, dict):
            # 從 item 構建完整路徑
            image_filename = item['image']
            image_path = os.path.join("data", f"skyfinder_{camera_id}", "images", image_filename)
        else:
            image_path = resolve_image_path(relative_path)
        
        if not image_path or not os.path.exists(image_path):
            print(f"  警告：找不到影像：{relative_path}")
            continue
        
        # 推導 subset_name（簡化版）
        subset_name = derive_subset_name_from_image(image_path)
        
        # 取得 prompt strategy
        prompt_strategy = get_prompt_strategy(subset_name)
        
        # SAM 推論
        try:
            pred_mask = predict_with_sam2(
                model, image_path, prompt_strategy, 
                model_type=model_type, device=device
            )
        except Exception as e:
            print(f"  錯誤：SAM 推論失敗 {image_path}: {e}")
            continue
        
        # 載入 GT mask
        gt_mask, has_gt = load_gt_mask(image_path)
        
        # 計算 metrics
        if has_gt and gt_mask is not None:
            metrics = calculate_metrics_per_image(pred_mask, gt_mask, has_gt=True)
            iou = metrics.get('iou', 0.0)
            fp_rate = metrics.get('fp_rate', 0.0)
            fn_rate = metrics.get('fn_rate', 0.0)
        else:
            # No-sky case
            metrics = calculate_metrics_per_image(pred_mask, None, has_gt=False)
            iou = None
            fp_rate = metrics.get('fp_rate', 0.0)
            fn_rate = None
        
        results.append({
            'image_path': relative_path,
            'camera_id': camera_id,
            'iou': iou,
            'fp_rate': fp_rate,
            'fn_rate': fn_rate,
            'has_gt': has_gt
        })
    
    return results


def aggregate_per_camera(results):
    """
    依 camera_id 聚合 metrics（與 eval_dl_final.py 相同邏輯）
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
            if r['iou'] is not None:
                stats['iou_sum'] += r['iou']
            if r['fn_rate'] is not None:
                stats['fn_rate_sum'] += r['fn_rate']
        else:
            stats['num_no_sky'] += 1
            stats['fp_rate_nosky_sum'] += r['fp_rate']
    
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
    parser = argparse.ArgumentParser(description='評估 SAM（使用固定 test_list.txt）')
    parser.add_argument('--test_list', type=str, default='outputs/test_list.txt',
                        help='test_list.txt 路徑')
    parser.add_argument('--model_type', type=str, default='sam2_hiera_small',
                        help='SAM 模型類型')
    parser.add_argument('--device', type=str, default='cpu',
                        help='設備（cpu/cuda）')
    parser.add_argument('--output_csv', type=str, default='outputs/metrics_summary.csv',
                        help='輸出 CSV 路徑')
    args = parser.parse_args()
    
    test_list_file = args.test_list
    
    print("=" * 60)
    print("  評估 VLM → SAM pipeline（使用固定 test_list.txt）")
    print("=" * 60)
    print()
    
    # === 檢查 SAM 可用性 ===
    if not SAM2_AVAILABLE:
        print("[錯誤] SAM2 未安裝或不可用")
        print("請確認已安裝 sam2 套件")
        sys.exit(1)
    
    print(f"  Test list:    {test_list_file}")
    print(f"  Model type:   {args.model_type}")
    print(f"  Device:       {args.device}")
    print()
    
    # === 載入 test list ===
    print(f"載入 test list: {test_list_file}...")
    if not os.path.exists(test_list_file):
        print(f"[錯誤] 找不到 test_list.txt: {test_list_file}")
        sys.exit(1)
    
    test_split_list = load_list_file(test_list_file)
    print(f"  共 {len(test_split_list)} 張測試影像")
    print()
    
    # === 載入 SAM 模型 ===
    print("載入 SAM 模型...")
    try:
        sam_model = load_sam2_model(model_type=args.model_type, device=args.device)
        print(f"  SAM 模型載入成功")
    except Exception as e:
        print(f"[錯誤] 無法載入 SAM 模型: {e}")
        sys.exit(1)
    print()
    
    # === 評估 ===
    print("開始評估...")
    results = evaluate_per_image_sam(
        sam_model, test_split_list, 
        device=args.device, model_type=args.model_type
    )
    print(f"  完成，共評估 {len(results)} 張影像")
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
            existing_rows = [row for row in reader if row.get('method') != 'SAM']
    
    # 準備新的 rows
    new_rows = []
    for camera_id in sorted(camera_metrics.keys()):
        metrics = camera_metrics[camera_id]
        row = {
            'method': 'SAM',
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
    print("  SAM 評估結果摘要")
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
