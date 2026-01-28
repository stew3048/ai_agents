"""
生成更詳細的視覺化報告

包含統計圖表、失敗模式分布、情境分析等

用法:
  python scripts/generate_detailed_visualization_report.py
"""

import os
import sys
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
            for col in ['dl_iou', 'dl_fp_rate', 'dl_fn_rate', 'dl_pred_positive_ratio']:
                if col in row:
                    row[col] = parse_float(row[col])
            rows.append(row)
    return rows


def generate_detailed_visualization_report(rows, output_path):
    """生成詳細的視覺化報告"""
    print(f"\n生成報告: {output_path}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# Detailed Visualization Report\n\n")
        f.write("> 詳細的失敗模式分析和視覺化報告\n\n")
        
        # 1. 整體統計
        f.write("## 1. 整體統計\n\n")
        
        total = len(rows)
        has_sky_rows = [r for r in rows if r.get('has_sky', '').upper() == 'TRUE']
        
        f.write(f"- **總樣本數**：{total}\n")
        f.write(f"- **有天空樣本**：{len(has_sky_rows)}\n")
        f.write(f"- **無天空樣本**：{total - len(has_sky_rows)}\n\n")
        
        # Failure mode 分布
        mode_counts = defaultdict(int)
        for row in rows:
            mode_counts[row.get('failure_mode', 'unknown')] += 1
        
        f.write("### Failure Mode 分布\n\n")
        f.write("| Failure Mode | 數量 | 百分比 |\n")
        f.write("|--------------|------|--------|\n")
        for mode, count in sorted(mode_counts.items(), key=lambda x: x[1], reverse=True):
            f.write(f"| {mode} | {count} | {count/total*100:.1f}% |\n")
        f.write("\n")
        
        # 2. 按 Scene 分析
        f.write("## 2. 按 Scene 分析\n\n")
        
        scene_data = defaultdict(lambda: {
            'total': 0,
            'FP_dominant': 0,
            'FN_dominant': 0,
            'good': 0,
            'avg_iou': [],
            'avg_fp': [],
            'avg_fn': []
        })
        
        for row in has_sky_rows:
            scene = row.get('scene(sea|urban|forest|other)', '')
            if not scene:
                continue
            
            data = scene_data[scene]
            data['total'] += 1
            
            failure_mode = row.get('failure_mode', '')
            if failure_mode in data:
                data[failure_mode] += 1
            
            iou = row.get('dl_iou')
            fp_rate = row.get('dl_fp_rate')
            fn_rate = row.get('dl_fn_rate')
            
            if iou is not None:
                data['avg_iou'].append(iou)
            if fp_rate is not None:
                data['avg_fp'].append(fp_rate)
            if fn_rate is not None:
                data['avg_fn'].append(fn_rate)
        
        for scene_name in ['urban', 'forest', 'sea', 'other']:
            if scene_name not in scene_data:
                continue
            
            data = scene_data[scene_name]
            total_scene = data['total']
            
            if total_scene == 0:
                continue
            
            f.write(f"### {scene_name.upper()} Scene\n\n")
            f.write(f"- **總樣本數**：{total_scene}\n")
            f.write(f"- **FP 主導**：{data['FP_dominant']} 張 ({data['FP_dominant']/total_scene*100:.1f}%)\n")
            f.write(f"- **FN 主導**：{data['FN_dominant']} 張 ({data['FN_dominant']/total_scene*100:.1f}%)\n")
            f.write(f"- **表現良好**：{data['good']} 張 ({data['good']/total_scene*100:.1f}%)\n")
            
            if data['avg_iou']:
                avg_iou = sum(data['avg_iou']) / len(data['avg_iou'])
                f.write(f"- **平均 IoU**：{avg_iou:.4f}\n")
            if data['avg_fp']:
                avg_fp = sum(data['avg_fp']) / len(data['avg_fp'])
                f.write(f"- **平均 FP Rate**：{avg_fp:.4f}\n")
            if data['avg_fn']:
                avg_fn = sum(data['avg_fn']) / len(data['avg_fn'])
                f.write(f"- **平均 FN Rate**：{avg_fn:.4f}\n")
            
            f.write("\n")
        
        # 3. 按 Light 條件分析
        f.write("## 3. 按 Light 條件分析\n\n")
        
        light_data = defaultdict(lambda: {
            'total': 0,
            'FP_dominant': 0,
            'FN_dominant': 0,
            'good': 0,
            'avg_iou': []
        })
        
        for row in has_sky_rows:
            light = row.get('light (day|dusk|night)', '')
            if not light:
                continue
            
            data = light_data[light]
            data['total'] += 1
            
            failure_mode = row.get('failure_mode', '')
            if failure_mode in data:
                data[failure_mode] += 1
            
            iou = row.get('dl_iou')
            if iou is not None:
                data['avg_iou'].append(iou)
        
        f.write("| Light | 總樣本 | FP 主導 | FN 主導 | 表現良好 | 平均 IoU |\n")
        f.write("|-------|--------|---------|---------|-----------|----------|\n")
        
        for light_name in ['day', 'dusk', 'night']:
            if light_name not in light_data:
                continue
            
            data = light_data[light_name]
            total_light = data['total']
            
            if total_light == 0:
                continue
            
            avg_iou_str = f"{sum(data['avg_iou'])/len(data['avg_iou']):.4f}" if data['avg_iou'] else "-"
            
            f.write(f"| {light_name} | {total_light} | {data['FP_dominant']} ({data['FP_dominant']/total_light*100:.1f}%) | "
                   f"{data['FN_dominant']} ({data['FN_dominant']/total_light*100:.1f}%) | "
                   f"{data['good']} ({data['good']/total_light*100:.1f}%) | {avg_iou_str} |\n")
        
        f.write("\n")
        
        # 4. 按 Occlusion 條件分析
        f.write("## 4. 按 Occlusion 條件分析\n\n")
        
        occ_data = defaultdict(lambda: {
            'total': 0,
            'FP_dominant': 0,
            'FN_dominant': 0,
            'good': 0,
            'avg_iou': []
        })
        
        for row in has_sky_rows:
            occ = row.get('occlusion(none|partial|heavy)', '')
            if not occ:
                continue
            
            data = occ_data[occ]
            data['total'] += 1
            
            failure_mode = row.get('failure_mode', '')
            if failure_mode in data:
                data[failure_mode] += 1
            
            iou = row.get('dl_iou')
            if iou is not None:
                data['avg_iou'].append(iou)
        
        f.write("| Occlusion | 總樣本 | FP 主導 | FN 主導 | 表現良好 | 平均 IoU |\n")
        f.write("|-----------|--------|---------|---------|-----------|----------|\n")
        
        for occ_name in ['none', 'partial', 'heavy']:
            if occ_name not in occ_data:
                continue
            
            data = occ_data[occ_name]
            total_occ = data['total']
            
            if total_occ == 0:
                continue
            
            avg_iou_str = f"{sum(data['avg_iou'])/len(data['avg_iou']):.4f}" if data['avg_iou'] else "-"
            
            f.write(f"| {occ_name} | {total_occ} | {data['FP_dominant']} ({data['FP_dominant']/total_occ*100:.1f}%) | "
                   f"{data['FN_dominant']} ({data['FN_dominant']/total_occ*100:.1f}%) | "
                   f"{data['good']} ({data['good']/total_occ*100:.1f}%) | {avg_iou_str} |\n")
        
        f.write("\n")
        
        # 5. Top 失敗案例（各情境）
        f.write("## 5. Top 失敗案例（各情境）\n\n")
        
        scenarios = {
            'no-sky': {
                'filter': lambda r: r.get('has_sky', '').upper() == 'FALSE',
                'sort': lambda r: r.get('dl_pred_positive_ratio', 0) if r.get('dl_pred_positive_ratio') is not None else 0,
                'reverse': True
            },
            'sea-sky': {
                'filter': lambda r: r.get('sea_sky_confusable', '') == '1',
                'sort': lambda r: r.get('dl_fp_rate', 0) if r.get('dl_fp_rate') is not None else 0,
                'reverse': True
            },
            'heavy-occlusion': {
                'filter': lambda r: r.get('occlusion(none|partial|heavy)', '') == 'heavy',
                'sort': lambda r: r.get('dl_fp_rate', 0) if r.get('dl_fp_rate') is not None else 0,
                'reverse': True
            },
            'night': {
                'filter': lambda r: (r.get('light (day|dusk|night)', '').lower() == 'night' and 
                                   r.get('has_sky', '').upper() == 'TRUE'),
                'sort': lambda r: r.get('dl_iou', 1) if r.get('dl_iou') is not None else 1,
                'reverse': False
            },
            'urban': {
                'filter': lambda r: (r.get('scene(sea|urban|forest|other)', '').lower() == 'urban' and 
                                    r.get('has_sky', '').upper() == 'TRUE'),
                'sort': lambda r: r.get('dl_fp_rate', 0) if r.get('dl_fp_rate') is not None else 0,
                'reverse': True
            }
        }
        
        for scenario_name, config in scenarios.items():
            filtered = [r for r in rows if config['filter'](r)]
            sorted_rows = sorted(filtered, key=config['sort'], reverse=config['reverse'])
            top5 = sorted_rows[:5]
            
            if not top5:
                continue
            
            f.write(f"### {scenario_name.upper()} - Top 5 失敗案例\n\n")
            f.write("| Camera | Image | FP Rate | FN Rate | IoU | Failure Reason | Overlay |\n")
            f.write("|--------|-------|---------|---------|-----|----------------|---------|\n")
            
            for row in top5:
                camera_id = row.get('camera_id', '')
                image_id = row.get('image_id', '')
                fp_rate = row.get('dl_fp_rate', 0)
                fn_rate = row.get('dl_fn_rate', 0)
                iou = row.get('dl_iou', 0)
                reason = row.get('failure_reason', '')
                overlay_path = row.get('overlay_path', '')
                
                fp_str = f"{fp_rate:.4f}" if fp_rate is not None else "-"
                fn_str = f"{fn_rate:.4f}" if fn_rate is not None else "-"
                iou_str = f"{iou:.4f}" if iou is not None else "-"
                reason_short = reason[:40] + "..." if len(reason) > 40 else reason
                overlay_link = f"[查看]({overlay_path})" if overlay_path else "-"
                
                f.write(f"| {camera_id} | {image_id} | {fp_str} | {fn_str} | {iou_str} | {reason_short} | {overlay_link} |\n")
            
            f.write("\n")
        
        # 6. 失敗原因統計
        f.write("## 6. 失敗原因統計\n\n")
        
        reason_counts = defaultdict(int)
        for row in rows:
            if row.get('failure_mode') in ['FP_dominant', 'FN_dominant', 'balanced']:
                reason = row.get('failure_reason', '')
                # 簡化 reason（取第一個部分）
                core_reason = reason.split(' | ')[0] if ' | ' in reason else reason
                if core_reason and core_reason != 'both_high_fp' and 'both_high' not in core_reason:
                    reason_counts[core_reason] += 1
        
        f.write("| 失敗原因 | 出現次數 |\n")
        f.write("|----------|----------|\n")
        for reason, count in sorted(reason_counts.items(), key=lambda x: x[1], reverse=True)[:15]:
            f.write(f"| {reason} | {count} |\n")
        
        f.write("\n")


def main():
    input_csv = 'outputs/diagnostic_with_failure_modes.csv'
    output_report = 'outputs/detailed_visualization_report.md'
    
    print("=" * 60)
    print("  Generate Detailed Visualization Report")
    print("=" * 60)
    print()
    print(f"  輸入 CSV:     {input_csv}")
    print(f"  輸出報告:     {output_report}")
    print()
    
    print("讀取 CSV...")
    rows = load_csv(input_csv)
    print(f"  讀取了 {len(rows)} 筆資料")
    
    generate_detailed_visualization_report(rows, output_report)
    
    print("\n" + "=" * 60)
    print("  完成")
    print("=" * 60)


if __name__ == '__main__':
    main()
