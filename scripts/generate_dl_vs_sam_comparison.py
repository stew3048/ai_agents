"""
生成 DL vs SAM 2.0 對比報告

功能：
- 讀取 DL 結果（failure_cases_analysis.csv）
- 讀取 SAM 2.0 結果（sam2_failure_cases_metrics.csv）
- 生成對比 CSV 和報告
- 分析各情境的改善/惡化情況

用法:
  python scripts/generate_dl_vs_sam_comparison.py
"""

import os
import sys
import argparse
import csv
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')


def parse_float(value):
    """解析浮點數"""
    if value == '' or value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def load_csv(csv_path):
    """讀取 CSV"""
    rows = []
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # 處理數值欄位
            for col in row.keys():
                if 'iou' in col.lower() or 'rate' in col.lower() or 'ratio' in col.lower():
                    row[col] = parse_float(row[col])
            rows.append(row)
    return rows


def calculate_improvement(dl_value, sam_value):
    """計算改善率"""
    if dl_value is None or sam_value is None:
        return None
    if dl_value == 0:
        return None
    return (sam_value - dl_value) / dl_value


def generate_comparison_csv(dl_rows, sam_rows, output_csv):
    """生成對比 CSV"""
    print(f"\n生成對比 CSV: {output_csv}")
    
    # 建立 SAM 結果索引
    sam_dict = {}
    for row in sam_rows:
        key = (row.get('camera_id', ''), row.get('image_id', ''))
        sam_dict[key] = row
    
    # 合併結果
    comparison_rows = []
    
    for dl_row in dl_rows:
        camera_id = dl_row.get('camera_id', '')
        image_id = dl_row.get('image_id', '')
        key = (camera_id, image_id)
        
        if key not in sam_dict:
            print(f"[WARNING] 找不到 SAM 結果: {camera_id}/{image_id}")
            continue
        
        sam_row = sam_dict[key]
        
        # 提取指標
        dl_iou = dl_row.get('dl_iou')
        dl_fp_rate = dl_row.get('dl_fp_rate')
        dl_fn_rate = dl_row.get('dl_fn_rate')
        dl_pred_ratio = dl_row.get('dl_pred_positive_ratio')
        
        sam_iou = sam_row.get('sam2_iou')
        sam_fp_rate = sam_row.get('sam2_fp_rate')
        sam_fn_rate = sam_row.get('sam2_fn_rate')
        sam_pred_ratio = sam_row.get('sam2_pred_positive_ratio')
        
        # 計算改善率
        iou_improvement = calculate_improvement(dl_iou, sam_iou)
        fp_improvement = calculate_improvement(dl_fp_rate, sam_fp_rate)
        fn_improvement = calculate_improvement(dl_fn_rate, sam_fn_rate)
        
        # 計算絕對差異
        iou_diff = sam_iou - dl_iou if (dl_iou is not None and sam_iou is not None) else None
        fp_diff = sam_fp_rate - dl_fp_rate if (dl_fp_rate is not None and sam_fp_rate is not None) else None
        fn_diff = sam_fn_rate - dl_fn_rate if (dl_fn_rate is not None and sam_fn_rate is not None) else None
        
        # 組合結果
        comp_row = {
            'subset_name': dl_row.get('subset_name', ''),
            'rank': dl_row.get('rank', ''),
            'camera_id': camera_id,
            'image_id': image_id,
            'failure_reason': dl_row.get('failure_reason', ''),
            # DL 指標
            'dl_iou': dl_iou,
            'dl_fp_rate': dl_fp_rate,
            'dl_fn_rate': dl_fn_rate,
            'dl_pred_positive_ratio': dl_pred_ratio,
            'dl_overlay_path': dl_row.get('overlay_path', ''),
            # SAM 2.0 指標
            'sam2_iou': sam_iou,
            'sam2_fp_rate': sam_fp_rate,
            'sam2_fn_rate': sam_fn_rate,
            'sam2_pred_positive_ratio': sam_pred_ratio,
            'sam2_overlay_path': sam_row.get('sam2_overlay_path', ''),
            # 對比
            'iou_diff': iou_diff,
            'iou_improvement': iou_improvement,
            'fp_rate_diff': fp_diff,
            'fp_rate_improvement': fp_improvement,
            'fn_rate_diff': fn_diff,
            'fn_rate_improvement': fn_improvement,
        }
        
        comparison_rows.append(comp_row)
    
    # 寫入 CSV
    if comparison_rows:
        fieldnames = list(comparison_rows[0].keys())
        with open(output_csv, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(comparison_rows)
        print(f"  完成！寫入了 {len(comparison_rows)} 筆對比結果")
    else:
        print("  沒有結果可寫入")


def generate_comparison_report(comparison_csv, output_report):
    """生成對比報告"""
    print(f"\n生成對比報告: {output_report}")
    
    rows = load_csv(comparison_csv)
    
    # 按情境分組
    scenario_data = defaultdict(list)
    for row in rows:
        scenario = row.get('subset_name', '')
        scenario_data[scenario].append(row)
    
    with open(output_report, 'w', encoding='utf-8') as f:
        f.write("# DL vs SAM 2.0 對比分析報告\n\n")
        f.write("> 基於 failure_cases_analysis.csv 中的 25 個 top-5 worst 樣本\n\n")
        
        # 整體統計
        f.write("## 整體統計\n\n")
        f.write(f"- **總樣本數**：{len(rows)}\n")
        
        # 計算平均指標
        dl_ious = [r.get('dl_iou') for r in rows if r.get('dl_iou') is not None]
        sam_ious = [r.get('sam2_iou') for r in rows if r.get('sam2_iou') is not None]
        
        if dl_ious and sam_ious:
            avg_dl_iou = sum(dl_ious) / len(dl_ious)
            avg_sam_iou = sum(sam_ious) / len(sam_ious)
            f.write(f"- **DL 平均 IoU**：{avg_dl_iou:.4f}\n")
            f.write(f"- **SAM 2.0 平均 IoU**：{avg_sam_iou:.4f}\n")
            f.write(f"- **平均 IoU 改善**：{avg_sam_iou - avg_dl_iou:.4f} ({((avg_sam_iou - avg_dl_iou) / avg_dl_iou * 100):.1f}%)\n")
        
        f.write("\n")
        
        # 各情境分析
        for scenario_name in ['no-sky', 'sea-sky-confusable', 'heavy-occlusion', 'night', 'urban']:
            if scenario_name not in scenario_data:
                continue
            
            scenario_rows = scenario_data[scenario_name]
            
            f.write(f"## {scenario_name.upper()} 情境\n\n")
            f.write(f"### 樣本數：{len(scenario_rows)}\n\n")
            
            # 計算該情境的平均指標
            scenario_dl_ious = [r.get('dl_iou') for r in scenario_rows if r.get('dl_iou') is not None]
            scenario_sam_ious = [r.get('sam2_iou') for r in scenario_rows if r.get('sam2_iou') is not None]
            scenario_dl_fp = [r.get('dl_fp_rate') for r in scenario_rows if r.get('dl_fp_rate') is not None]
            scenario_sam_fp = [r.get('sam2_fp_rate') for r in scenario_rows if r.get('sam2_fp_rate') is not None]
            scenario_dl_fn = [r.get('dl_fn_rate') for r in scenario_rows if r.get('dl_fn_rate') is not None]
            scenario_sam_fn = [r.get('sam2_fn_rate') for r in scenario_rows if r.get('sam2_fn_rate') is not None]
            
            if scenario_dl_ious and scenario_sam_ious:
                avg_dl_iou = sum(scenario_dl_ious) / len(scenario_dl_ious)
                avg_sam_iou = sum(scenario_sam_ious) / len(scenario_sam_ious)
                f.write(f"**平均 IoU**：\n")
                f.write(f"- DL: {avg_dl_iou:.4f}\n")
                f.write(f"- SAM 2.0: {avg_sam_iou:.4f}\n")
                f.write(f"- 改善: {avg_sam_iou - avg_dl_iou:.4f} ({((avg_sam_iou - avg_dl_iou) / avg_dl_iou * 100):.1f}%)\n\n")
            
            if scenario_dl_fp and scenario_sam_fp:
                avg_dl_fp = sum(scenario_dl_fp) / len(scenario_dl_fp)
                avg_sam_fp = sum(scenario_sam_fp) / len(scenario_sam_fp)
                f.write(f"**平均 FP Rate**：\n")
                f.write(f"- DL: {avg_dl_fp:.4f}\n")
                f.write(f"- SAM 2.0: {avg_sam_fp:.4f}\n")
                f.write(f"- 改善: {avg_sam_fp - avg_dl_fp:.4f} ({((avg_sam_fp - avg_dl_fp) / avg_dl_fp * 100):.1f}%)\n\n")
            
            if scenario_dl_fn and scenario_sam_fn:
                avg_dl_fn = sum(scenario_dl_fn) / len(scenario_dl_fn)
                avg_sam_fn = sum(scenario_sam_fn) / len(scenario_sam_fn)
                f.write(f"**平均 FN Rate**：\n")
                f.write(f"- DL: {avg_dl_fn:.4f}\n")
                f.write(f"- SAM 2.0: {avg_sam_fn:.4f}\n")
                f.write(f"- 改善: {avg_sam_fn - avg_dl_fn:.4f} ({((avg_sam_fn - avg_dl_fn) / avg_dl_fn * 100):.1f}%)\n\n")
            
            # Top 改善/惡化案例
            improvements = []
            for row in scenario_rows:
                iou_diff = row.get('iou_diff')
                if iou_diff is not None:
                    improvements.append((iou_diff, row))
            
            if improvements:
                improvements_sorted = sorted(improvements, key=lambda x: x[0], reverse=True)
                
                f.write("### Top 改善案例（IoU 提升最多）\n\n")
                f.write("| Rank | Camera | Image | DL IoU | SAM IoU | 改善 | DL Overlay | SAM Overlay |\n")
                f.write("|------|--------|-------|--------|---------|------|------------|-------------|\n")
                
                for diff, row in improvements_sorted[:3]:
                    camera_id = row.get('camera_id', '')
                    image_id = row.get('image_id', '')
                    dl_iou = row.get('dl_iou', 0)
                    sam_iou = row.get('sam2_iou', 0)
                    dl_overlay = row.get('dl_overlay_path', '')
                    sam_overlay = row.get('sam2_overlay_path', '')
                    
                    dl_iou_str = f"{dl_iou:.4f}" if dl_iou is not None else "-"
                    sam_iou_str = f"{sam_iou:.4f}" if sam_iou is not None else "-"
                    diff_str = f"{diff:.4f}" if diff is not None else "-"
                    dl_link = f"[查看]({dl_overlay})" if dl_overlay else "-"
                    sam_link = f"[查看]({sam_overlay})" if sam_overlay else "-"
                    
                    f.write(f"| {row.get('rank', '')} | {camera_id} | {image_id} | {dl_iou_str} | {sam_iou_str} | {diff_str} | {dl_link} | {sam_link} |\n")
                
                f.write("\n")
                
                f.write("### Top 惡化案例（IoU 下降最多）\n\n")
                f.write("| Rank | Camera | Image | DL IoU | SAM IoU | 惡化 | DL Overlay | SAM Overlay |\n")
                f.write("|------|--------|-------|--------|---------|------|------------|-------------|\n")
                
                for diff, row in improvements_sorted[-3:]:
                    camera_id = row.get('camera_id', '')
                    image_id = row.get('image_id', '')
                    dl_iou = row.get('dl_iou', 0)
                    sam_iou = row.get('sam2_iou', 0)
                    dl_overlay = row.get('dl_overlay_path', '')
                    sam_overlay = row.get('sam2_overlay_path', '')
                    
                    dl_iou_str = f"{dl_iou:.4f}" if dl_iou is not None else "-"
                    sam_iou_str = f"{sam_iou:.4f}" if sam_iou is not None else "-"
                    diff_str = f"{diff:.4f}" if diff is not None else "-"
                    dl_link = f"[查看]({dl_overlay})" if dl_overlay else "-"
                    sam_link = f"[查看]({sam_overlay})" if sam_overlay else "-"
                    
                    f.write(f"| {row.get('rank', '')} | {camera_id} | {image_id} | {dl_iou_str} | {sam_iou_str} | {diff_str} | {dl_link} | {sam_link} |\n")
                
                f.write("\n")
            
            f.write("---\n\n")
        
        # 總結
        f.write("## 總結\n\n")
        
        # 統計改善/惡化數量
        improved_count = sum(1 for r in rows if r.get('iou_diff') is not None and r.get('iou_diff') > 0)
        worsened_count = sum(1 for r in rows if r.get('iou_diff') is not None and r.get('iou_diff') < 0)
        same_count = sum(1 for r in rows if r.get('iou_diff') is not None and abs(r.get('iou_diff', 0)) < 0.01)
        
        f.write("### IoU 改善統計\n\n")
        f.write(f"- **改善**：{improved_count} 個樣本\n")
        f.write(f"- **惡化**：{worsened_count} 個樣本\n")
        f.write(f"- **相近**：{same_count} 個樣本\n\n")
        
        # 各情境改善情況
        f.write("### 各情境改善情況\n\n")
        f.write("| 情境 | 樣本數 | 平均 IoU 改善 | 改善樣本數 | 惡化樣本數 |\n")
        f.write("|------|--------|--------------|-----------|-----------|\n")
        
        for scenario_name in ['no-sky', 'sea-sky-confusable', 'heavy-occlusion', 'night', 'urban']:
            if scenario_name not in scenario_data:
                continue
            
            scenario_rows = scenario_data[scenario_name]
            scenario_improvements = [r.get('iou_diff') for r in scenario_rows if r.get('iou_diff') is not None]
            
            if scenario_improvements:
                avg_improvement = sum(scenario_improvements) / len(scenario_improvements)
                improved = sum(1 for diff in scenario_improvements if diff > 0)
                worsened = sum(1 for diff in scenario_improvements if diff < 0)
                
                avg_str = f"{avg_improvement:.4f}"
                f.write(f"| {scenario_name} | {len(scenario_rows)} | {avg_str} | {improved} | {worsened} |\n")


def main():
    parser = argparse.ArgumentParser(description='生成 DL vs SAM 2.0 對比報告')
    parser.add_argument('--dl_csv', type=str, default='outputs/failure_cases_analysis.csv',
                       help='DL 結果 CSV 路徑')
    parser.add_argument('--sam_csv', type=str, default='outputs/sam2_failure_cases_metrics.csv',
                       help='SAM 2.0 結果 CSV 路徑')
    parser.add_argument('--comparison_csv', type=str, default='outputs/dl_vs_sam_comparison.csv',
                       help='對比 CSV 輸出路徑')
    parser.add_argument('--comparison_report', type=str, default='outputs/dl_vs_sam_comparison_report.md',
                       help='對比報告輸出路徑')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("  Generate DL vs SAM 2.0 Comparison")
    print("=" * 60)
    print()
    print(f"  DL CSV:          {args.dl_csv}")
    print(f"  SAM CSV:         {args.sam_csv}")
    print(f"  對比 CSV:        {args.comparison_csv}")
    print(f"  對比報告:        {args.comparison_report}")
    print()
    
    print("讀取 DL 結果...")
    dl_rows = load_csv(args.dl_csv)
    print(f"  讀取了 {len(dl_rows)} 筆資料")
    
    print("\n讀取 SAM 2.0 結果...")
    sam_rows = load_csv(args.sam_csv)
    print(f"  讀取了 {len(sam_rows)} 筆資料")
    
    generate_comparison_csv(dl_rows, sam_rows, args.comparison_csv)
    generate_comparison_report(args.comparison_csv, args.comparison_report)
    
    print("\n" + "=" * 60)
    print("  完成")
    print("=" * 60)


if __name__ == '__main__':
    main()
