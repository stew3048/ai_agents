"""
對 diagnostic_with_dl_metrics.csv 進行情境切片分析，找出每個關鍵情境子集的 top-5 失敗案例

核心流程：
1. 切片篩選（5 個關鍵子集）
2. Top-5 失敗案例提取
3. 失敗原因分析（一句話問題）

用法:
  python scripts/analyze_diagnostic_failures.py
  python scripts/analyze_diagnostic_failures.py --top_k 5
"""

import os
import sys
import argparse
import csv
from pathlib import Path
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')


# 子集配置
SUBSET_CONFIGS = [
    {
        'name': 'no-sky',
        'condition': lambda row: row.get('has_sky', '').upper() == 'FALSE',
        'sort_by': 'dl_pred_positive_ratio',
        'focus_metrics': ['dl_fp_rate', 'dl_pred_positive_ratio'],
        'failure_question': '為什麼會誤判為天空？'
    },
    {
        'name': 'sea-sky-confusable',
        'condition': lambda row: row.get('sea_sky_confusable', '') == '1',
        'sort_by': 'dl_fp_rate',
        'focus_metrics': ['dl_fp_rate'],
        'failure_question': '為什麼海天混淆導致誤判？'
    },
    {
        'name': 'heavy-occlusion',
        'condition': lambda row: row.get('occlusion(none|partial|heavy)', '') == 'heavy',
        'sort_by': 'dl_fn_rate',
        'focus_metrics': ['dl_fn_rate'],
        'failure_question': '為什麼漏檢天空？'
    },
    {
        'name': 'night',
        'condition': lambda row: (row.get('light (day|dusk|night)', '') == 'night') and (row.get('has_sky', '').upper() == 'TRUE'),
        'sort_by': ['dl_fn_rate', 'dl_iou'],  # 優先 FN，其次 IoU
        'focus_metrics': ['dl_fn_rate', 'dl_iou'],
        'failure_question': '為什麼夜間表現差？'
    },
    {
        'name': 'urban',
        'condition': lambda row: (row.get('scene(sea|urban|forest|other)', '') == 'urban') and (row.get('has_sky', '').upper() == 'TRUE'),
        'sort_by': ['dl_fp_rate', 'dl_iou'],  # 優先 FP，其次 IoU
        'focus_metrics': ['dl_fp_rate', 'dl_iou'],
        'failure_question': '為什麼城市場景表現差？'
    }
]


def parse_float(value):
    """解析浮點數，處理空值"""
    if value == '' or value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def load_diagnostic_data(csv_path):
    """讀取 diagnostic CSV"""
    print(f"讀取資料: {csv_path}")
    rows = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # 處理數值欄位
            for col in ['dl_iou', 'dl_fp_rate', 'dl_fn_rate', 'dl_pred_positive_ratio']:
                if col in row:
                    row[col] = parse_float(row[col])
            rows.append(row)
    
    print(f"  總樣本數: {len(rows)}")
    return rows


def slice_subset(rows, condition_func):
    """根據條件篩選子集"""
    return [row for row in rows if condition_func(row)]


def extract_top_failures(subset_rows, sort_by, top_k=5):
    """
    根據排序規則提取 top-k 失敗案例
    
    參數:
        subset_rows: 子集資料（字典列表）
        sort_by: 排序欄位（字串或列表）
        top_k: 提取前 k 個
    """
    if len(subset_rows) == 0:
        return []
    
    # 處理排序
    def get_sort_key(row):
        if isinstance(sort_by, list):
            # 多欄位排序：返回元組
            key_parts = []
            for col in sort_by:
                val = row.get(col)
                # 處理 None：放在最後（用很大的負數）
                if val is None:
                    key_parts.append(-999999)
                else:
                    key_parts.append(val)
            return tuple(key_parts)
        else:
            # 單欄位排序
            val = row.get(sort_by)
            if val is None:
                return -999999
            return val
    
    # 排序（降序：值越大越失敗）
    sorted_rows = sorted(subset_rows, key=get_sort_key, reverse=True)
    
    # 提取前 top_k 個
    top_failures = sorted_rows[:top_k]
    
    # 添加 rank
    for i, row in enumerate(top_failures, start=1):
        row['rank'] = i
    
    return top_failures


def analyze_failure_pattern(failure_cases, subset_name):
    """分析失敗案例的共同模式，生成失敗原因描述"""
    if len(failure_cases) == 0:
        return "無失敗案例"
    
    # 計算指標平均值
    metric_summary = {}
    focus_metrics_map = {
        'no-sky': ['dl_fp_rate', 'dl_pred_positive_ratio'],
        'sea-sky-confusable': ['dl_fp_rate'],
        'heavy-occlusion': ['dl_fn_rate'],
        'night': ['dl_fn_rate', 'dl_iou'],
        'urban': ['dl_fp_rate', 'dl_iou']
    }
    
    focus_metrics = focus_metrics_map.get(subset_name, [])
    for metric in focus_metrics:
        values = [row.get(metric) for row in failure_cases if row.get(metric) is not None]
        if values:
            metric_summary[metric] = {
                'mean': sum(values) / len(values),
                'max': max(values),
                'min': min(values)
            }
    
    # 根據子集類型生成失敗原因
    return generate_failure_reason(subset_name, metric_summary)


def generate_failure_reason(subset_name, metric_summary):
    """根據子集類型和指標生成失敗原因描述"""
    
    if subset_name == 'no-sky':
        pred_ratio = metric_summary.get('dl_pred_positive_ratio', {}).get('mean', 0)
        if pred_ratio > 0.3:
            return "亮牆/雪地/高亮度區域被當成天空，模型只靠顏色亮度判斷"
        else:
            return "部分高亮度區域被誤判為天空"
    
    elif subset_name == 'sea-sky-confusable':
        fp_rate = metric_summary.get('dl_fp_rate', {}).get('mean', 0)
        if fp_rate > 0.3:
            return "海面反光與天空顏色相似，模型無法區分海天邊界"
        else:
            return "海天顏色相似導致部分誤判"
    
    elif subset_name == 'heavy-occlusion':
        fn_rate = metric_summary.get('dl_fn_rate', {}).get('mean', 0)
        if fn_rate > 0.1:
            return "建築物/樹木嚴重遮擋天空邊界，模型漏檢被遮擋的天空區域"
        else:
            return "嚴重遮擋導致部分天空區域漏檢"
    
    elif subset_name == 'night':
        fn_rate = metric_summary.get('dl_fn_rate', {}).get('mean', 0)
        iou = metric_summary.get('dl_iou', {}).get('mean', 0)
        if fn_rate > 0.05:
            return "夜間低對比度，模型無法識別天空邊界，導致漏檢"
        elif iou < 0.6:
            return "夜間低亮度導致整體分割精度下降"
        else:
            return "夜間場景表現較差"
    
    elif subset_name == 'urban':
        fp_rate = metric_summary.get('dl_fp_rate', {}).get('mean', 0)
        iou = metric_summary.get('dl_iou', {}).get('mean', 0)
        if fp_rate > 0.2:
            return "建築物頂部/玻璃反光與天空混淆，模型誤判非天空為天空"
        elif iou < 0.7:
            return "城市場景複雜，建築物邊界與天空混淆"
        else:
            return "城市場景複雜度導致表現下降"
    
    else:
        return "需要進一步分析"


def generate_failure_report(all_failures_dict, output_md_path):
    """生成 Markdown 報告"""
    print(f"\n生成報告: {output_md_path}")
    
    with open(output_md_path, 'w', encoding='utf-8') as f:
        f.write("# Diagnostic Failure Analysis Report\n\n")
        f.write("> 聚焦於關鍵情境子集的 top-5 失敗案例分析\n\n")
        
        # 統計摘要
        total_cases = sum(len(cases) for cases in all_failures_dict.values())
        f.write("## 統計摘要\n\n")
        f.write(f"- 總分析子集數：{len(SUBSET_CONFIGS)}\n")
        f.write(f"- 總失敗案例數：{total_cases}\n")
        f.write(f"- 每個子集提取：Top-5 失敗案例\n\n")
        
        # 每個子集的分析
        for config in SUBSET_CONFIGS:
            subset_name = config['name']
            subset_failures = all_failures_dict.get(subset_name, [])
            
            f.write(f"## {subset_name.upper()} 子集分析\n\n")
            
            # 條件描述
            f.write("### 條件\n")
            if subset_name == 'no-sky':
                f.write("- `has_sky == FALSE`\n")
            elif subset_name == 'sea-sky-confusable':
                f.write("- `sea_sky_confusable == 1`\n")
            elif subset_name == 'heavy-occlusion':
                f.write("- `occlusion == heavy`\n")
            elif subset_name == 'night':
                f.write("- `light == night AND has_sky == TRUE`\n")
            elif subset_name == 'urban':
                f.write("- `scene == urban AND has_sky == TRUE`\n")
            
            f.write(f"- 樣本數：{len(subset_failures)}\n\n")
            
            if len(subset_failures) == 0:
                f.write("**無失敗案例**\n\n")
                continue
            
            # Top-5 失敗案例表格
            f.write("### Top-5 失敗案例\n\n")
            f.write("| Rank | Camera | Image | ")
            
            focus_metrics = config.get('focus_metrics', [])
            metric_labels = {
                'dl_iou': 'IoU',
                'dl_fp_rate': 'FP Rate',
                'dl_fn_rate': 'FN Rate',
                'dl_pred_positive_ratio': 'Pred Positive Ratio'
            }
            
            for metric in focus_metrics:
                if metric in metric_labels:
                    f.write(f"{metric_labels[metric]} | ")
            
            f.write("Overlay |\n")
            f.write("|------|--------|-------|")
            for _ in focus_metrics:
                f.write("--------|")
            f.write("--------|\n")
            
            for row in subset_failures:
                f.write(f"| {row.get('rank', '?')} | {row.get('camera_id', '?')} | {row.get('image_id', '?')} | ")
                
                for metric in focus_metrics:
                    val = row.get(metric)
                    if val is None:
                        f.write("- | ")
                    else:
                        f.write(f"{val:.4f} | ")
                
                overlay_path = row.get('overlay_path', '')
                if overlay_path:
                    f.write(f"[查看]({overlay_path}) |\n")
                else:
                    f.write("- |\n")
            
            f.write("\n")
            
            # 失敗原因分析
            f.write("### 失敗原因分析\n\n")
            failure_reason = row.get('failure_reason', '需要進一步分析')
            f.write(f"**一句話問題**：{failure_reason}\n\n")
            
            # 共同特徵
            weather_counts = defaultdict(int)
            light_counts = defaultdict(int)
            for row in subset_failures:
                weather = row.get('weather(clear|cloudy|rain|fog|snow|unknown)', '')
                if weather:
                    weather_counts[weather] += 1
                light = row.get('light (day|dusk|night)', '')
                if light:
                    light_counts[light] += 1
            
            if weather_counts:
                f.write("**Weather 分布**：\n")
                for weather, count in sorted(weather_counts.items(), key=lambda x: x[1], reverse=True):
                    f.write(f"- {weather}: {count} 張\n")
                f.write("\n")
            
            if light_counts:
                f.write("**Light 分布**：\n")
                for light, count in sorted(light_counts.items(), key=lambda x: x[1], reverse=True):
                    f.write(f"- {light}: {count} 張\n")
                f.write("\n")
            
            f.write("---\n\n")
        
        # 關鍵發現總結
        f.write("## 關鍵發現總結\n\n")
        f.write("### 失敗模式總結\n\n")
        
        for config in SUBSET_CONFIGS:
            subset_name = config['name']
            subset_failures = all_failures_dict.get(subset_name, [])
            if len(subset_failures) > 0:
                failure_reason = subset_failures[0].get('failure_reason', '')
                f.write(f"- **{subset_name}**：{failure_reason}\n")
        
        f.write("\n### 建議\n\n")
        f.write("- 針對不同情境子集的失敗模式，可以考慮：\n")
        f.write("  1. 使用不同的模型或方法（例如：VLM/SAM 用於海天混淆場景）\n")
        f.write("  2. 針對特定失敗模式進行資料增強或後處理\n")
        f.write("  3. 建立選模策略：根據情境自動選擇最適合的方法\n")
    
    print(f"  報告已生成")


def export_failure_cases(all_failures_dict, output_csv_path):
    """導出所有失敗案例到 CSV"""
    print(f"\n導出 CSV: {output_csv_path}")
    
    # 收集所有失敗案例
    all_rows = []
    for subset_name, failures in all_failures_dict.items():
        for row in failures:
            all_rows.append(row)
    
    if not all_rows:
        print("  無失敗案例可導出")
        return
    
    # 導出欄位
    fieldnames = [
        'subset_name', 'rank', 'camera_id', 'image_id',
        'dl_iou', 'dl_fp_rate', 'dl_fn_rate', 'dl_pred_positive_ratio',
        'overlay_path', 'failure_reason'
    ]
    
    with open(output_csv_path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for row in all_rows:
            export_row = {}
            for field in fieldnames:
                export_row[field] = row.get(field, '')
            writer.writerow(export_row)
    
    print(f"  已導出 {len(all_rows)} 筆失敗案例")


def main():
    parser = argparse.ArgumentParser(description='診斷失敗案例分析')
    parser.add_argument('--input_csv', type=str, default='outputs/diagnostic_with_dl_metrics.csv',
                       help='輸入 CSV 路徑')
    parser.add_argument('--output_csv', type=str, default='outputs/failure_cases_analysis.csv',
                       help='輸出 CSV 路徑')
    parser.add_argument('--output_report', type=str, default='outputs/failure_analysis_report.md',
                       help='輸出 Markdown 報告路徑')
    parser.add_argument('--top_k', type=int, default=5,
                       help='每個子集提取的失敗案例數（預設: 5）')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("  Diagnostic Failure Analysis")
    print("=" * 60)
    print()
    print(f"  輸入 CSV:     {args.input_csv}")
    print(f"  輸出 CSV:     {args.output_csv}")
    print(f"  輸出報告:     {args.output_report}")
    print(f"  Top-K:        {args.top_k}")
    print()
    
    # 讀取資料
    rows = load_diagnostic_data(args.input_csv)
    
    # 處理每個子集
    all_failures_dict = {}
    
    for config in SUBSET_CONFIGS:
        subset_name = config['name']
        print(f"\n處理子集: {subset_name}")
        print("-" * 60)
        
        # 切片
        subset_rows = slice_subset(rows, config['condition'])
        print(f"  樣本數: {len(subset_rows)}")
        
        if len(subset_rows) == 0:
            print(f"  跳過（無樣本）")
            continue
        
        # 提取 top failures
        sort_by = config['sort_by']
        top_failures = extract_top_failures(subset_rows, sort_by, top_k=args.top_k)
        
        if len(top_failures) == 0:
            print(f"  跳過（無法排序）")
            continue
        
        print(f"  提取 Top-{len(top_failures)} 失敗案例")
        
        # 分析失敗模式
        failure_reason = analyze_failure_pattern(top_failures, subset_name)
        print(f"  失敗原因: {failure_reason}")
        
        # 添加子集名稱和失敗原因
        for row in top_failures:
            row['subset_name'] = subset_name
            row['failure_reason'] = failure_reason
        
        all_failures_dict[subset_name] = top_failures
    
    # 導出結果
    if all_failures_dict:
        # 導出 CSV
        export_failure_cases(all_failures_dict, args.output_csv)
        
        # 生成報告
        generate_failure_report(all_failures_dict, args.output_report)
        
        total_cases = sum(len(cases) for cases in all_failures_dict.values())
        print("\n" + "=" * 60)
        print("  完成")
        print("=" * 60)
        print(f"  總失敗案例數: {total_cases}")
        print(f"  CSV: {args.output_csv}")
        print(f"  報告: {args.output_report}")
    else:
        print("\n[警告] 沒有找到任何失敗案例")


if __name__ == '__main__':
    main()
