"""
合併所有方法的評估結果，產出統一 metrics_summary.csv

此腳本主要用於驗證和格式化最終的 metrics_summary.csv
各個評估腳本（eval_dl_final.py, eval_sam_final.py, eval_clipseg_final.py, eval_dino_dl_final.py）
已經會自動追加結果到 metrics_summary.csv

此腳本的功能：
- 驗證 CSV 格式
- 顯示摘要統計
- 可選：重新排序和格式化
"""

import os
import sys
import csv
import argparse
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')


def load_metrics_csv(csv_path):
    """載入 metrics_summary.csv"""
    if not os.path.exists(csv_path):
        return []
    
    rows = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    
    return rows


def aggregate_by_method(rows):
    """依 method 聚合統計"""
    method_stats = defaultdict(lambda: {
        'cameras': set(),
        'total_images': 0,
        'iou_sum': 0.0,
        'iou_count': 0,
        'fp_sum': 0.0,
        'fn_sum': 0.0,
        'fn_count': 0,
        'fp_nosky_sum': 0.0,
        'fp_nosky_count': 0
    })
    
    for row in rows:
        method = row.get('method', '')
        camera_id = row.get('camera_id', '')
        num_images = int(row.get('num_images', 0))
        
        stats = method_stats[method]
        stats['cameras'].add(camera_id)
        stats['total_images'] += num_images
        
        # IoU
        iou_str = row.get('IoU_mean', '')
        if iou_str:
            try:
                stats['iou_sum'] += float(iou_str)
                stats['iou_count'] += 1
            except ValueError:
                pass
        
        # FP
        fp_str = row.get('FP_mean', '')
        if fp_str:
            try:
                stats['fp_sum'] += float(fp_str) * num_images
            except ValueError:
                pass
        
        # FN
        fn_str = row.get('FN_mean', '')
        if fn_str:
            try:
                stats['fn_sum'] += float(fn_str) * num_images
                stats['fn_count'] += num_images
            except ValueError:
                pass
        
        # FP (no-sky)
        fp_nosky_str = row.get('FP_rate_nosky', '')
        if fp_nosky_str:
            try:
                stats['fp_nosky_sum'] += float(fp_nosky_str) * num_images
                stats['fp_nosky_count'] += num_images
            except ValueError:
                pass
    
    # 計算平均值
    aggregated = {}
    for method, stats in method_stats.items():
        aggregated[method] = {
            'num_cameras': len(stats['cameras']),
            'total_images': stats['total_images'],
            'avg_iou': stats['iou_sum'] / stats['iou_count'] if stats['iou_count'] > 0 else None,
            'avg_fp': stats['fp_sum'] / stats['total_images'] if stats['total_images'] > 0 else None,
            'avg_fn': stats['fn_sum'] / stats['fn_count'] if stats['fn_count'] > 0 else None,
            'avg_fp_nosky': stats['fp_nosky_sum'] / stats['fp_nosky_count'] if stats['fp_nosky_count'] > 0 else None
        }
    
    return aggregated


def main():
    parser = argparse.ArgumentParser(description='合併所有方法的評估結果')
    parser.add_argument('--input_csv', type=str, default='outputs/metrics_summary.csv',
                        help='輸入 CSV 路徑')
    parser.add_argument('--output_csv', type=str, default=None,
                        help='輸出 CSV 路徑（如果指定，會重新排序和格式化）')
    args = parser.parse_args()
    
    input_csv = args.input_csv
    
    print("=" * 60)
    print("  合併所有方法的評估結果")
    print("=" * 60)
    print()
    
    # === 載入 CSV ===
    print(f"載入 metrics: {input_csv}...")
    if not os.path.exists(input_csv):
        print(f"[錯誤] 找不到 CSV 檔案: {input_csv}")
        print("請先執行各個評估腳本")
        sys.exit(1)
    
    rows = load_metrics_csv(input_csv)
    print(f"  共 {len(rows)} 筆記錄")
    print()
    
    # === 驗證格式 ===
    print("驗證 CSV 格式...")
    fieldnames = ['method', 'camera_id', 'num_images', 'IoU_mean', 'FP_mean', 'FN_mean', 'FP_rate_nosky']
    methods = set()
    cameras = set()
    
    for row in rows:
        methods.add(row.get('method', ''))
        cameras.add(row.get('camera_id', ''))
    
    print(f"  方法: {sorted(methods)}")
    print(f"  Camera 數: {len(cameras)}")
    print()
    
    # === 聚合統計 ===
    print("聚合統計...")
    method_stats = aggregate_by_method(rows)
    
    print("=" * 60)
    print("  整體統計摘要")
    print("=" * 60)
    print()
    
    for method in sorted(method_stats.keys()):
        stats = method_stats[method]
        print(f"{method}:")
        print(f"  Camera 數: {stats['num_cameras']}")
        print(f"  總影像數: {stats['total_images']}")
        if stats['avg_iou'] is not None:
            print(f"  平均 IoU: {stats['avg_iou']:.4f}")
        if stats['avg_fp'] is not None:
            print(f"  平均 FP Rate: {stats['avg_fp']:.4f}")
        if stats['avg_fn'] is not None:
            print(f"  平均 FN Rate: {stats['avg_fn']:.4f}")
        if stats['avg_fp_nosky'] is not None:
            print(f"  平均 FP Rate (no-sky): {stats['avg_fp_nosky']:.4f}")
        print()
    
    # === 重新排序和格式化（如果需要） ===
    if args.output_csv:
        print(f"重新排序和格式化...")
        
        # 定義方法順序
        method_order = ['DL', 'SAM', 'CLIPSeg', 'DINO-MLP', 'DINO-CNN', 'DINO-FPN', 'DINO-Hybrid']
        
        # 排序：先依 method，再依 camera_id
        def sort_key(row):
            method = row.get('method', '')
            camera_id = row.get('camera_id', '')
            method_idx = method_order.index(method) if method in method_order else len(method_order)
            return (method_idx, camera_id)
        
        sorted_rows = sorted(rows, key=sort_key)
        
        # 寫入
        with open(args.output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(sorted_rows)
        
        print(f"  已寫入 {args.output_csv}")
        print()
    
    # === Per-camera 摘要 ===
    print("=" * 60)
    print("  Per-Camera 摘要（前 5 個 camera）")
    print("=" * 60)
    print()
    
    camera_methods = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        camera_id = row.get('camera_id', '')
        method = row.get('method', '')
        camera_methods[camera_id][method] = row
    
    for camera_id in sorted(camera_methods.keys())[:5]:
        print(f"Camera {camera_id}:")
        for method in sorted(camera_methods[camera_id].keys()):
            row = camera_methods[camera_id][method]
            print(f"  {method}:")
            print(f"    影像數: {row.get('num_images', 'N/A')}")
            if row.get('IoU_mean'):
                print(f"    IoU: {row.get('IoU_mean', 'N/A')}")
            print(f"    FP Rate: {row.get('FP_mean', 'N/A')}")
            if row.get('FN_mean'):
                print(f"    FN Rate: {row.get('FN_mean', 'N/A')}")
            if row.get('FP_rate_nosky'):
                print(f"    FP Rate (no-sky): {row.get('FP_rate_nosky', 'N/A')}")
        print()
    
    print("=" * 60)
    print("  完成")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
