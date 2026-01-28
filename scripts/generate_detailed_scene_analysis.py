"""
生成詳細的 scene failure mode 分析報告

列出每個 scene 的所有 FP/FN 案例，方便驗證歸納是否正確

用法:
  python scripts/generate_detailed_scene_analysis.py
"""

import os
import sys
import argparse
import csv

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


def load_csv(csv_path):
    """讀取 CSV"""
    rows = []
    with open(csv_path, 'r', encoding='utf-8-sig') as f:  # 使用 utf-8-sig 自動處理 BOM
        reader = csv.DictReader(f)
        for row in reader:
            # 處理數值欄位
            for col in ['dl_iou', 'dl_fp_rate', 'dl_fn_rate', 'dl_pred_positive_ratio']:
                if col in row:
                    row[col] = parse_float(row[col])
            rows.append(row)
    return rows


def generate_detailed_report(csv_path, output_md_path):
    """生成詳細報告"""
    print(f"讀取: {csv_path}")
    rows = load_csv(csv_path)
    
    # 按 scene 和 failure_mode 分組
    scene_data = {}
    
    for row in rows:
        scene = row.get('scene(sea|urban|forest|other)', '')
        has_sky = row.get('has_sky', '').upper() == 'TRUE'
        
        # 只分析 has_sky=TRUE 的樣本
        if not has_sky:
            continue
        
        if scene not in scene_data:
            scene_data[scene] = {
                'FP_dominant': [],
                'FN_dominant': [],
                'balanced': [],
                'good': [],
                'boundary_issue': []
            }
        
        failure_mode = row.get('failure_mode', 'unknown')
        if failure_mode in scene_data[scene]:
            scene_data[scene][failure_mode].append(row)
    
    print(f"\n生成報告: {output_md_path}")
    
    with open(output_md_path, 'w', encoding='utf-8') as f:
        f.write("# Detailed Scene Failure Mode Analysis\n\n")
        f.write("> 列出每個 scene 的所有 FP/FN 案例，方便驗證歸納是否正確\n\n")
        
        # 每個 scene 的詳細分析
        for scene_name in ['urban', 'forest', 'sea', 'other']:
            if scene_name not in scene_data:
                continue
            
            data = scene_data[scene_name]
            fp_cases = data['FP_dominant']
            fn_cases = data['FN_dominant']
            balanced_cases = data['balanced']
            
            f.write(f"## {scene_name.upper()} Scene\n\n")
            
            # 統計
            total = len(fp_cases) + len(fn_cases) + len(balanced_cases) + len(data['good'])
            f.write(f"### 統計\n\n")
            f.write(f"- 總樣本數：{total}\n")
            f.write(f"- **FP 主導**：{len(fp_cases)} 張 ({len(fp_cases)/total*100:.1f}%)\n")
            f.write(f"- **FN 主導**：{len(fn_cases)} 張 ({len(fn_cases)/total*100:.1f}%)\n")
            f.write(f"- **FP+FN 都高**：{len(balanced_cases)} 張\n")
            f.write(f"- **表現良好**：{len(data['good'])} 張\n\n")
            
            # FP 主導案例列表
            if len(fp_cases) > 0:
                f.write(f"### FP 主導案例（{len(fp_cases)} 張）\n\n")
                f.write("| Camera | Image | FP Rate | FN Rate | IoU | Occlusion | Weather | Light | Failure Reason | Overlay |\n")
                f.write("|--------|-------|---------|---------|-----|-----------|---------|-------|----------------|---------|\n")
                
                # 按 FP rate 排序
                fp_cases_sorted = sorted(fp_cases, key=lambda x: x.get('dl_fp_rate', 0) if x.get('dl_fp_rate') else 0, reverse=True)
                
                for row in fp_cases_sorted:
                    camera_id = row.get('camera_id', '')
                    image_id = row.get('image_id', '')
                    fp_rate = row.get('dl_fp_rate', 0)
                    fn_rate = row.get('dl_fn_rate', 0)
                    iou = row.get('dl_iou', 0)
                    occlusion = row.get('occlusion(none|partial|heavy)', '')
                    weather = row.get('weather(clear|cloudy|rain|fog|snow|unknown)', '')
                    light = row.get('light (day|dusk|night)', '')
                    reason = row.get('failure_reason', '')
                    overlay_path = row.get('overlay_path', '')
                    
                    fp_str = f"{fp_rate:.4f}" if fp_rate is not None else "-"
                    fn_str = f"{fn_rate:.4f}" if fn_rate is not None else "-"
                    iou_str = f"{iou:.4f}" if iou is not None else "-"
                    overlay_link = f"[查看]({overlay_path})" if overlay_path else "-"
                    
                    f.write(f"| {camera_id} | {image_id} | {fp_str} | {fn_str} | {iou_str} | {occlusion} | {weather} | {light} | {reason} | {overlay_link} |\n")
                
                f.write("\n")
            
            # FN 主導案例列表
            if len(fn_cases) > 0:
                f.write(f"### FN 主導案例（{len(fn_cases)} 張）\n\n")
                f.write("| Camera | Image | FP Rate | FN Rate | IoU | Occlusion | Weather | Light | Failure Reason | Overlay |\n")
                f.write("|--------|-------|---------|---------|-----|-----------|---------|-------|----------------|---------|\n")
                
                # 按 FN rate 排序
                fn_cases_sorted = sorted(fn_cases, key=lambda x: x.get('dl_fn_rate', 0) if x.get('dl_fn_rate') else 0, reverse=True)
                
                for row in fn_cases_sorted:
                    camera_id = row.get('camera_id', '')
                    image_id = row.get('image_id', '')
                    fp_rate = row.get('dl_fp_rate', 0)
                    fn_rate = row.get('dl_fn_rate', 0)
                    iou = row.get('dl_iou', 0)
                    occlusion = row.get('occlusion(none|partial|heavy)', '')
                    weather = row.get('weather(clear|cloudy|rain|fog|snow|unknown)', '')
                    light = row.get('light (day|dusk|night)', '')
                    reason = row.get('failure_reason', '')
                    overlay_path = row.get('overlay_path', '')
                    
                    fp_str = f"{fp_rate:.4f}" if fp_rate is not None else "-"
                    fn_str = f"{fn_rate:.4f}" if fn_rate is not None else "-"
                    iou_str = f"{iou:.4f}" if iou is not None else "-"
                    overlay_link = f"[查看]({overlay_path})" if overlay_path else "-"
                    
                    f.write(f"| {camera_id} | {image_id} | {fp_str} | {fn_str} | {iou_str} | {occlusion} | {weather} | {light} | {reason} | {overlay_link} |\n")
                
                f.write("\n")
            
            # FP+FN 都高的案例
            if len(balanced_cases) > 0:
                f.write(f"### FP+FN 都高的案例（{len(balanced_cases)} 張）\n\n")
                f.write("| Camera | Image | FP Rate | FN Rate | IoU | Failure Reason | Overlay |\n")
                f.write("|--------|-------|---------|---------|-----|----------------|---------|\n")
                
                for row in balanced_cases:
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
                    overlay_link = f"[查看]({overlay_path})" if overlay_path else "-"
                    
                    f.write(f"| {camera_id} | {image_id} | {fp_str} | {fn_str} | {iou_str} | {reason} | {overlay_link} |\n")
                
                f.write("\n")
            
            f.write("---\n\n")
        
        # 總結
        f.write("## 總結\n\n")
        f.write("### 各 Scene 的錯誤型態分布\n\n")
        
        for scene_name in ['urban', 'forest', 'sea', 'other']:
            if scene_name in scene_data:
                data = scene_data[scene_name]
                total = len(data['FP_dominant']) + len(data['FN_dominant']) + len(data['balanced']) + len(data['good'])
                fp_count = len(data['FP_dominant'])
                fn_count = len(data['FN_dominant'])
                
                f.write(f"- **{scene_name.upper()}**：\n")
                f.write(f"  - FP 主導：{fp_count} 張 ({fp_count/total*100:.1f}%)\n")
                f.write(f"  - FN 主導：{fn_count} 張 ({fn_count/total*100:.1f}%)\n")
                f.write(f"  - 總樣本：{total} 張\n\n")
    
    print(f"  報告已生成")


def main():
    parser = argparse.ArgumentParser(description='生成詳細的 scene failure mode 分析報告')
    parser.add_argument('--input_csv', type=str, default='outputs/diagnostic_with_failure_modes.csv',
                       help='輸入 CSV 路徑（已包含 failure_mode）')
    parser.add_argument('--output_report', type=str, default='outputs/detailed_scene_failure_analysis.md',
                       help='輸出 Markdown 報告路徑')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("  Detailed Scene Failure Mode Analysis")
    print("=" * 60)
    print()
    print(f"  輸入 CSV:     {args.input_csv}")
    print(f"  輸出報告:     {args.output_report}")
    print()
    
    generate_detailed_report(args.input_csv, args.output_report)
    
    print("\n" + "=" * 60)
    print("  完成")
    print("=" * 60)


if __name__ == '__main__':
    main()
