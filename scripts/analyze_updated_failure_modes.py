"""
分析用戶手動修改後的 failure_mode CSV

分析修改後的分布和模式

用法:
  python scripts/analyze_updated_failure_modes.py
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


def load_csv(csv_path):
    """讀取 CSV（處理編碼和分隔符問題）"""
    rows = []
    encodings = ['utf-8-sig', 'utf-8', 'big5', 'gb2312']
    delimiters = ['\t', ',']  # 嘗試 tab 和逗號
    
    for encoding in encodings:
        for delimiter in delimiters:
            try:
                with open(csv_path, 'r', encoding=encoding) as f:
                    reader = csv.DictReader(f, delimiter=delimiter)
                    test_row = next(reader, None)
                    if test_row and 'failure_mode' in test_row:
                        # 重置檔案指標
                        f.seek(0)
                        reader = csv.DictReader(f, delimiter=delimiter)
                        for row in reader:
                            # 處理數值欄位
                            for col in ['dl_iou', 'dl_fp_rate', 'dl_fn_rate', 'dl_pred_positive_ratio']:
                                if col in row:
                                    row[col] = parse_float(row[col])
                            rows.append(row)
                        delim_str = 'TAB' if delimiter == '\t' else 'COMMA'
                        print(f"成功使用編碼: {encoding}, 分隔符: {delim_str}")
                        return rows
            except (UnicodeDecodeError, UnicodeError, StopIteration, KeyError):
                continue
    
    # 如果都失敗，嘗試直接讀取（忽略編碼錯誤）
    try:
        with open(csv_path, 'r', encoding='utf-8-sig', errors='ignore') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                for col in ['dl_iou', 'dl_fp_rate', 'dl_fn_rate', 'dl_pred_positive_ratio']:
                    if col in row:
                        row[col] = parse_float(row[col])
                rows.append(row)
        print(f"使用 UTF-8-sig (忽略錯誤) 讀取了 {len(rows)} 筆")
    except Exception as e:
        print(f"讀取失敗: {e}")
    
    return rows


def analyze_failure_modes(rows):
    """分析 failure_mode 分布"""
    print("\n" + "=" * 60)
    print("  Failure Mode 整體分布")
    print("=" * 60)
    
    mode_counts = defaultdict(int)
    mode_by_scene = defaultdict(lambda: defaultdict(int))
    mode_by_light = defaultdict(lambda: defaultdict(int))
    mode_by_occlusion = defaultdict(lambda: defaultdict(int))
    
    for row in rows:
        failure_mode = row.get('failure_mode', 'unknown')
        scene = row.get('scene(sea|urban|forest|other)', '')
        light = row.get('light (day|dusk|night)', '')
        occlusion = row.get('occlusion(none|partial|heavy)', '')
        has_sky = row.get('has_sky', '').upper() == 'TRUE'
        
        mode_counts[failure_mode] += 1
        
        if has_sky:  # 只統計有天空的樣本
            if scene:
                mode_by_scene[scene][failure_mode] += 1
            if light:
                mode_by_light[light][failure_mode] += 1
            if occlusion:
                mode_by_occlusion[occlusion][failure_mode] += 1
    
    print("\n整體分布：")
    total = len(rows)
    for mode, count in sorted(mode_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {mode}: {count} 張 ({count/total*100:.1f}%)")
    
    return mode_counts, mode_by_scene, mode_by_light, mode_by_occlusion


def analyze_by_scene(mode_by_scene):
    """按 scene 分析"""
    print("\n" + "=" * 60)
    print("  按 Scene 分析")
    print("=" * 60)
    
    for scene_name in ['urban', 'forest', 'sea', 'other']:
        if scene_name not in mode_by_scene:
            continue
        
        scene_data = mode_by_scene[scene_name]
        total = sum(scene_data.values())
        
        if total == 0:
            continue
        
        print(f"\n{scene_name.upper()} Scene（{total} 個樣本）：")
        
        fp_count = scene_data.get('FP_dominant', 0)
        fn_count = scene_data.get('FN_dominant', 0)
        balanced_count = scene_data.get('balanced', 0)
        good_count = scene_data.get('good', 0)
        
        print(f"  FP 主導：{fp_count} 張 ({fp_count/total*100:.1f}%)")
        print(f"  FN 主導：{fn_count} 張 ({fn_count/total*100:.1f}%)")
        print(f"  FP+FN 都高：{balanced_count} 張 ({balanced_count/total*100:.1f}%)")
        print(f"  表現良好：{good_count} 張 ({good_count/total*100:.1f}%)")


def analyze_failure_reasons(rows):
    """分析 failure_reason 的模式"""
    print("\n" + "=" * 60)
    print("  Failure Reason 分析")
    print("=" * 60)
    
    reason_counts = defaultdict(int)
    reason_by_scene = defaultdict(lambda: defaultdict(int))
    
    for row in rows:
        failure_mode = row.get('failure_mode', '')
        failure_reason = row.get('failure_reason', '')
        scene = row.get('scene(sea|urban|forest|other)', '')
        has_sky = row.get('has_sky', '').upper() == 'TRUE'
        
        if failure_mode in ['FP_dominant', 'FN_dominant', 'balanced'] and has_sky:
            # 簡化 reason（移除情境標籤，只看核心原因）
            core_reason = failure_reason.split(' | ')[0] if ' | ' in failure_reason else failure_reason
            reason_counts[core_reason] += 1
            
            if scene:
                reason_by_scene[scene][core_reason] += 1
    
    print("\n主要失敗原因（Top 10）：")
    for reason, count in sorted(reason_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {reason}: {count} 次")
    
    print("\n按 Scene 的主要失敗原因：")
    for scene_name in ['urban', 'forest', 'sea']:
        if scene_name not in reason_by_scene:
            continue
        
        scene_reasons = reason_by_scene[scene_name]
        if not scene_reasons:
            continue
        
        print(f"\n{scene_name.upper()} Scene：")
        for reason, count in sorted(scene_reasons.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {reason}: {count} 次")


def analyze_specific_cases(rows):
    """分析特定案例（例如用戶可能修改的）"""
    print("\n" + "=" * 60)
    print("  特定案例分析")
    print("=" * 60)
    
    # 找出 FP/FN 都高的案例
    balanced_cases = []
    for row in rows:
        if row.get('failure_mode') == 'balanced':
            balanced_cases.append(row)
    
    print(f"\nFP+FN 都高的案例：{len(balanced_cases)} 張")
    if len(balanced_cases) > 0:
        print("\n前 5 個案例：")
        for i, row in enumerate(balanced_cases[:5], 1):
            camera_id = row.get('camera_id', '')
            image_id = row.get('image_id', '')
            fp_rate = row.get('dl_fp_rate', 0)
            fn_rate = row.get('dl_fn_rate', 0)
            iou = row.get('dl_iou', 0)
            scene = row.get('scene(sea|urban|forest|other)', '')
            
            fp_str = f"{fp_rate:.4f}" if fp_rate is not None else "-"
            fn_str = f"{fn_rate:.4f}" if fn_rate is not None else "-"
            iou_str = f"{iou:.4f}" if iou is not None else "-"
            
            print(f"  {i}. Camera {camera_id}, Image {image_id} ({scene})")
            print(f"     FP: {fp_str}, FN: {fn_str}, IoU: {iou_str}")


def generate_summary_report(rows, mode_counts, mode_by_scene, output_path):
    """生成摘要報告"""
    print(f"\n生成摘要報告: {output_path}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# Updated Failure Mode Analysis Summary\n\n")
        f.write("> 基於用戶手動修改後的 failure_mode CSV\n\n")
        
        # 整體統計
        f.write("## 整體統計\n\n")
        total = len(rows)
        f.write(f"- 總樣本數：{total}\n\n")
        
        f.write("| Failure Mode | 數量 | 百分比 |\n")
        f.write("|--------------|------|--------|\n")
        for mode, count in sorted(mode_counts.items(), key=lambda x: x[1], reverse=True):
            f.write(f"| {mode} | {count} | {count/total*100:.1f}% |\n")
        
        f.write("\n")
        
        # 按 Scene 分析
        f.write("## 按 Scene 分析\n\n")
        for scene_name in ['urban', 'forest', 'sea', 'other']:
            if scene_name not in mode_by_scene:
                continue
            
            scene_data = mode_by_scene[scene_name]
            total_scene = sum(scene_data.values())
            
            if total_scene == 0:
                continue
            
            f.write(f"### {scene_name.upper()} Scene（{total_scene} 個樣本）\n\n")
            
            fp_count = scene_data.get('FP_dominant', 0)
            fn_count = scene_data.get('FN_dominant', 0)
            balanced_count = scene_data.get('balanced', 0)
            good_count = scene_data.get('good', 0)
            
            f.write(f"- **FP 主導**：{fp_count} 張 ({fp_count/total_scene*100:.1f}%)\n")
            f.write(f"- **FN 主導**：{fn_count} 張 ({fn_count/total_scene*100:.1f}%)\n")
            f.write(f"- **FP+FN 都高**：{balanced_count} 張 ({balanced_count/total_scene*100:.1f}%)\n")
            f.write(f"- **表現良好**：{good_count} 張 ({good_count/total_scene*100:.1f}%)\n\n")
    
    print(f"  報告已生成")


def main():
    parser = argparse.ArgumentParser(description='分析用戶手動修改後的 failure_mode CSV')
    parser.add_argument('--input_csv', type=str, default='outputs/diagnostic_with_failure_modes.csv',
                       help='輸入 CSV 路徑')
    parser.add_argument('--output_report', type=str, default='outputs/updated_failure_analysis_summary.md',
                       help='輸出摘要報告路徑')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("  Analyze Updated Failure Modes")
    print("=" * 60)
    print()
    print(f"  輸入 CSV:     {args.input_csv}")
    print(f"  輸出報告:     {args.output_report}")
    print()
    
    # 讀取 CSV
    print("讀取 CSV...")
    rows = load_csv(args.input_csv)
    print(f"  讀取了 {len(rows)} 筆資料")
    
    # 分析
    mode_counts, mode_by_scene, mode_by_light, mode_by_occlusion = analyze_failure_modes(rows)
    analyze_by_scene(mode_by_scene)
    analyze_failure_reasons(rows)
    analyze_specific_cases(rows)
    
    # 生成報告
    generate_summary_report(rows, mode_counts, mode_by_scene, args.output_report)
    
    print("\n" + "=" * 60)
    print("  完成")
    print("=" * 60)


if __name__ == '__main__':
    main()
