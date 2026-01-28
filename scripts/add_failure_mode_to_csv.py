"""
為 diagnostic_with_dl_metrics.csv 添加 failure_mode 欄位

為每張影像標註：
- failure_mode: FP_dominant / FN_dominant / balanced / good
- failure_reason: 具體的失敗原因描述

用法:
  python scripts/add_failure_mode_to_csv.py
"""

import os
import sys
import argparse
import csv
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')


def parse_float(value):
    """解析浮點數，處理空值"""
    if value == '' or value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def classify_failure_mode(row):
    """
    分類每張影像的 failure mode
    
    返回：
    - failure_mode: FP_dominant / FN_dominant / balanced / good / no_gt
    - failure_reason: 具體原因描述
    """
    fp_rate = row.get('dl_fp_rate')
    fn_rate = row.get('dl_fn_rate')
    iou = row.get('dl_iou')
    has_sky = row.get('has_sky', '').upper() == 'TRUE'
    scene = row.get('scene(sea|urban|forest|other)', '')
    occlusion = row.get('occlusion(none|partial|heavy)', '')
    weather = row.get('weather(clear|cloudy|rain|fog|snow|unknown)', '')
    light = row.get('light (day|dusk|night)', '')
    
    # 沒有 GT 的情況
    if not has_sky:
        pred_ratio = row.get('dl_pred_positive_ratio')
        if pred_ratio and pred_ratio > 0.2:
            return 'FP_dominant', 'no_sky_but_predicted_sky'
        else:
            return 'good', 'no_sky_correctly_predicted'
    
    # 沒有指標值
    if fp_rate is None or fn_rate is None:
        return 'unknown', 'missing_metrics'
    
    # 定義閾值
    fp_high = fp_rate > 0.15
    fn_high = fn_rate > 0.05
    iou_low = iou is not None and iou < 0.6
    
    # 分類
    if fp_high and not fn_high:
        # FP 主導
        reason = generate_fp_reason(scene, occlusion, weather, light, fp_rate, iou)
        return 'FP_dominant', reason
    elif fn_high and not fp_high:
        # FN 主導
        reason = generate_fn_reason(scene, occlusion, weather, light, fn_rate, iou)
        return 'FN_dominant', reason
    elif fp_high and fn_high:
        # 兩者都高
        return 'balanced', f'both_high_fp_{fp_rate:.3f}_fn_{fn_rate:.3f}'
    elif iou_low:
        # IoU 低但 FP/FN 都不高（可能是邊界問題）
        return 'boundary_issue', f'low_iou_{iou:.3f}_boundary_problem'
    else:
        # 表現良好
        return 'good', 'good_performance'


def generate_fp_reason(scene, occlusion, weather, light, fp_rate, iou):
    """生成 FP 失敗原因"""
    reasons = []
    
    if scene == 'urban':
        if fp_rate > 0.25:
            reasons.append('建築物頂部/玻璃反光被誤判')
        else:
            reasons.append('建築邊緣誤判')
    elif scene == 'forest':
        reasons.append('樹葉間隙/亮葉被誤判')
    elif scene == 'sea':
        if fp_rate > 0.3:
            reasons.append('海面反光/波浪被誤判（海天混淆）')
        else:
            reasons.append('水面顏色相似誤判')
    else:
        reasons.append('非天空區域被誤判')
    
    # 添加情境資訊
    if occlusion == 'heavy':
        reasons.append('嚴重遮擋場景')
    if weather == 'fog':
        reasons.append('霧天')
    if light == 'night':
        reasons.append('夜間')
    
    if iou and iou < 0.6:
        reasons.append('整體IoU低')
    
    return ' | '.join(reasons)


def generate_fn_reason(scene, occlusion, weather, light, fn_rate, iou):
    """生成 FN 失敗原因"""
    reasons = []
    
    if scene == 'urban':
        if fn_rate > 0.2:
            reasons.append('漏檢狹長天空區域')
        else:
            reasons.append('部分天空漏檢')
    elif scene == 'forest':
        reasons.append('樹葉/樹枝遮擋漏檢')
    elif scene == 'sea':
        reasons.append('天空區域漏檢')
    else:
        reasons.append('天空區域漏檢')
    
    # 添加情境資訊
    if occlusion == 'heavy':
        reasons.append('嚴重遮擋導致')
    if weather == 'fog':
        reasons.append('霧天低能見度')
    if light == 'night':
        reasons.append('夜間低對比度')
    
    if iou and iou < 0.6:
        reasons.append('整體IoU低')
    
    return ' | '.join(reasons)


def add_failure_mode_to_csv(input_csv, output_csv):
    """為 CSV 添加 failure_mode 欄位"""
    print(f"讀取: {input_csv}")
    
    rows = []
    with open(input_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)
        
        for row in reader:
            # 處理數值欄位
            for col in ['dl_iou', 'dl_fp_rate', 'dl_fn_rate', 'dl_pred_positive_ratio']:
                if col in row:
                    row[col] = parse_float(row[col])
            
            # 分類 failure mode
            failure_mode, failure_reason = classify_failure_mode(row)
            row['failure_mode'] = failure_mode
            row['failure_reason'] = failure_reason
            
            rows.append(row)
    
    print(f"  處理了 {len(rows)} 筆資料")
    
    # 統計
    mode_counts = defaultdict(int)
    for row in rows:
        mode_counts[row['failure_mode']] += 1
    
    print("\nFailure Mode 分布：")
    for mode, count in sorted(mode_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {mode}: {count} 張")
    
    # 寫入輸出 CSV
    print(f"\n寫入: {output_csv}")
    output_fieldnames = fieldnames + ['failure_mode', 'failure_reason']
    
    with open(output_csv, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=output_fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"  完成！已添加 failure_mode 和 failure_reason 欄位")


def main():
    parser = argparse.ArgumentParser(description='為 diagnostic CSV 添加 failure_mode 欄位')
    parser.add_argument('--input_csv', type=str, default='outputs/diagnostic_with_dl_metrics.csv',
                       help='輸入 CSV 路徑')
    parser.add_argument('--output_csv', type=str, default='outputs/diagnostic_with_failure_modes.csv',
                       help='輸出 CSV 路徑')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("  Add Failure Mode to CSV")
    print("=" * 60)
    print()
    
    add_failure_mode_to_csv(args.input_csv, args.output_csv)
    
    print("\n" + "=" * 60)
    print("  完成")
    print("=" * 60)


if __name__ == '__main__':
    main()
