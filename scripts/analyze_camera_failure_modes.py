"""
分析特定 camera 的失敗模式

針對每個 camera，分析其失敗模式分布和特徵

用法:
  python scripts/analyze_camera_failure_modes.py
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
            # 處理數值欄位
            for col in ['dl_iou', 'dl_fp_rate', 'dl_fn_rate', 'dl_pred_positive_ratio']:
                if col in row:
                    row[col] = parse_float(row[col])
            rows.append(row)
    return rows


def analyze_camera_failure_modes(rows):
    """分析每個 camera 的失敗模式"""
    camera_data = defaultdict(lambda: {
        'total': 0,
        'FP_dominant': 0,
        'FN_dominant': 0,
        'balanced': 0,
        'good': 0,
        'scenes': defaultdict(int),
        'occlusions': defaultdict(int),
        'lights': defaultdict(int),
        'avg_iou': [],
        'avg_fp_rate': [],
        'avg_fn_rate': [],
        'cases': []
    })
    
    for row in rows:
        camera_id = row.get('camera_id', '')
        failure_mode = row.get('failure_mode', '')
        scene = row.get('scene(sea|urban|forest|other)', '')
        occlusion = row.get('occlusion(none|partial|heavy)', '')
        light = row.get('light (day|dusk|night)', '')
        has_sky = row.get('has_sky', '').upper() == 'TRUE'
        
        if not camera_id:
            continue
        
        data = camera_data[camera_id]
        data['total'] += 1
        
        if failure_mode in data:
            data[failure_mode] += 1
        
        if scene:
            data['scenes'][scene] += 1
        if occlusion:
            data['occlusions'][occlusion] += 1
        if light:
            data['lights'][light] += 1
        
        # 收集指標
        if has_sky:
            iou = row.get('dl_iou')
            fp_rate = row.get('dl_fp_rate')
            fn_rate = row.get('dl_fn_rate')
            
            if iou is not None:
                data['avg_iou'].append(iou)
            if fp_rate is not None:
                data['avg_fp_rate'].append(fp_rate)
            if fn_rate is not None:
                data['avg_fn_rate'].append(fn_rate)
        
        # 記錄案例
        data['cases'].append({
            'image_id': row.get('image_id', ''),
            'failure_mode': failure_mode,
            'failure_reason': row.get('failure_reason', ''),
            'iou': row.get('dl_iou'),
            'fp_rate': row.get('dl_fp_rate'),
            'fn_rate': row.get('dl_fn_rate'),
            'overlay_path': row.get('overlay_path', '')
        })
    
    return camera_data


def generate_camera_analysis_report(camera_data, output_path):
    """生成 camera 分析報告"""
    print(f"\n生成報告: {output_path}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# Camera Failure Mode Analysis\n\n")
        f.write("> 針對每個 camera 分析其失敗模式分布和特徵\n\n")
        
        # 按總樣本數排序
        sorted_cameras = sorted(camera_data.items(), key=lambda x: x[1]['total'], reverse=True)
        
        for camera_id, data in sorted_cameras:
            total = data['total']
            fp_count = data['FP_dominant']
            fn_count = data['FN_dominant']
            balanced_count = data['balanced']
            good_count = data['good']
            
            f.write(f"## Camera {camera_id}\n\n")
            f.write(f"### 基本統計\n\n")
            f.write(f"- 總樣本數：{total}\n")
            f.write(f"- **FP 主導**：{fp_count} 張 ({fp_count/total*100:.1f}%)\n")
            f.write(f"- **FN 主導**：{fn_count} 張 ({fn_count/total*100:.1f}%)\n")
            f.write(f"- **FP+FN 都高**：{balanced_count} 張 ({balanced_count/total*100:.1f}%)\n")
            f.write(f"- **表現良好**：{good_count} 張 ({good_count/total*100:.1f}%)\n\n")
            
            # 平均指標
            if data['avg_iou']:
                avg_iou = sum(data['avg_iou']) / len(data['avg_iou'])
                f.write(f"- **平均 IoU**：{avg_iou:.4f}\n")
            if data['avg_fp_rate']:
                avg_fp = sum(data['avg_fp_rate']) / len(data['avg_fp_rate'])
                f.write(f"- **平均 FP Rate**：{avg_fp:.4f}\n")
            if data['avg_fn_rate']:
                avg_fn = sum(data['avg_fn_rate']) / len(data['avg_fn_rate'])
                f.write(f"- **平均 FN Rate**：{avg_fn:.4f}\n")
            f.write("\n")
            
            # 情境分布
            if data['scenes']:
                f.write(f"### Scene 分布\n\n")
                for scene, count in sorted(data['scenes'].items(), key=lambda x: x[1], reverse=True):
                    f.write(f"- {scene}: {count} 張 ({count/total*100:.1f}%)\n")
                f.write("\n")
            
            if data['occlusions']:
                f.write(f"### Occlusion 分布\n\n")
                for occ, count in sorted(data['occlusions'].items(), key=lambda x: x[1], reverse=True):
                    f.write(f"- {occ}: {count} 張 ({count/total*100:.1f}%)\n")
                f.write("\n")
            
            if data['lights']:
                f.write(f"### Light 分布\n\n")
                for light, count in sorted(data['lights'].items(), key=lambda x: x[1], reverse=True):
                    f.write(f"- {light}: {count} 張 ({count/total*100:.1f}%)\n")
                f.write("\n")
            
            # Top 失敗案例
            fp_cases = [c for c in data['cases'] if c['failure_mode'] == 'FP_dominant']
            fn_cases = [c for c in data['cases'] if c['failure_mode'] == 'FN_dominant']
            
            if fp_cases:
                f.write(f"### Top FP 失敗案例（前 5）\n\n")
                fp_sorted = sorted(fp_cases, key=lambda x: x['fp_rate'] if x['fp_rate'] else 0, reverse=True)[:5]
                f.write("| Image | FP Rate | FN Rate | IoU | Failure Reason | Overlay |\n")
                f.write("|-------|---------|---------|-----|----------------|---------|\n")
                for case in fp_sorted:
                    fp_str = f"{case['fp_rate']:.4f}" if case['fp_rate'] is not None else "-"
                    fn_str = f"{case['fn_rate']:.4f}" if case['fn_rate'] is not None else "-"
                    iou_str = f"{case['iou']:.4f}" if case['iou'] is not None else "-"
                    reason = case['failure_reason'][:40] + "..." if len(case['failure_reason']) > 40 else case['failure_reason']
                    overlay_link = f"[查看]({case['overlay_path']})" if case['overlay_path'] else "-"
                    f.write(f"| {case['image_id']} | {fp_str} | {fn_str} | {iou_str} | {reason} | {overlay_link} |\n")
                f.write("\n")
            
            if fn_cases:
                f.write(f"### Top FN 失敗案例（前 5）\n\n")
                fn_sorted = sorted(fn_cases, key=lambda x: x['fn_rate'] if x['fn_rate'] else 0, reverse=True)[:5]
                f.write("| Image | FP Rate | FN Rate | IoU | Failure Reason | Overlay |\n")
                f.write("|-------|---------|---------|-----|----------------|---------|\n")
                for case in fn_sorted:
                    fp_str = f"{case['fp_rate']:.4f}" if case['fp_rate'] is not None else "-"
                    fn_str = f"{case['fn_rate']:.4f}" if case['fn_rate'] is not None else "-"
                    iou_str = f"{case['iou']:.4f}" if case['iou'] is not None else "-"
                    reason = case['failure_reason'][:40] + "..." if len(case['failure_reason']) > 40 else case['failure_reason']
                    overlay_link = f"[查看]({case['overlay_path']})" if case['overlay_path'] else "-"
                    f.write(f"| {case['image_id']} | {fp_str} | {fn_str} | {iou_str} | {reason} | {overlay_link} |\n")
                f.write("\n")
            
            f.write("---\n\n")
        
        # 總結
        f.write("## 總結\n\n")
        f.write("### Camera 失敗模式分布\n\n")
        f.write("| Camera | 總樣本 | FP 主導 | FN 主導 | 表現良好 | 主要問題 |\n")
        f.write("|--------|--------|---------|---------|-----------|----------|\n")
        
        for camera_id, data in sorted_cameras:
            total = data['total']
            fp_count = data['FP_dominant']
            fn_count = data['FN_dominant']
            good_count = data['good']
            
            # 判斷主要問題
            if fp_count > fn_count * 1.5:
                main_issue = "FP 主導"
            elif fn_count > fp_count * 1.5:
                main_issue = "FN 主導"
            elif fp_count > 0 or fn_count > 0:
                main_issue = "FP+FN 混合"
            else:
                main_issue = "表現良好"
            
            f.write(f"| {camera_id} | {total} | {fp_count} ({fp_count/total*100:.1f}%) | {fn_count} ({fn_count/total*100:.1f}%) | {good_count} ({good_count/total*100:.1f}%) | {main_issue} |\n")


def main():
    input_csv = 'outputs/diagnostic_with_failure_modes.csv'
    output_report = 'outputs/camera_failure_modes_analysis.md'
    
    print("=" * 60)
    print("  Camera Failure Mode Analysis")
    print("=" * 60)
    print()
    print(f"  輸入 CSV:     {input_csv}")
    print(f"  輸出報告:     {output_report}")
    print()
    
    print("讀取 CSV...")
    rows = load_csv(input_csv)
    print(f"  讀取了 {len(rows)} 筆資料")
    
    print("\n分析 camera 失敗模式...")
    camera_data = analyze_camera_failure_modes(rows)
    print(f"  分析了 {len(camera_data)} 個 camera")
    
    generate_camera_analysis_report(camera_data, output_report)
    
    print("\n" + "=" * 60)
    print("  完成")
    print("=" * 60)


if __name__ == '__main__':
    main()
