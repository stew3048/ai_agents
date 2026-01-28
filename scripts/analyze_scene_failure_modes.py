"""
針對每個 scene 進行 failure mode 分析

重點：
- 不是比較平均分數
- 而是比較錯誤型態（FP vs FN）
- 找出具體的失敗模式

用法:
  python scripts/analyze_scene_failure_modes.py
"""

import os
import sys
import argparse
import csv
from collections import defaultdict
from pathlib import Path

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


def filter_scene_rows(rows, scene_name, has_sky=True):
    """篩選特定 scene 的樣本"""
    filtered = []
    for row in rows:
        scene = row.get('scene(sea|urban|forest|other)', '')
        has_sky_val = row.get('has_sky', '').upper()
        
        if scene == scene_name:
            if has_sky:
                if has_sky_val == 'TRUE':
                    filtered.append(row)
            else:
                filtered.append(row)
    
    return filtered


def analyze_failure_mode(scene_rows, scene_name):
    """
    分析 scene 的 failure mode
    
    返回：
    - fp_dominant: FP 主導的失敗案例
    - fn_dominant: FN 主導的失敗案例
    - failure_mode_description: 失敗模式描述
    """
    if len(scene_rows) == 0:
        return None, None, "無樣本"
    
    # 分類失敗案例
    fp_cases = []  # FP 主導（FP rate 高，FN rate 低）
    fn_cases = []  # FN 主導（FN rate 高，FP rate 低）
    balanced_cases = []  # 兩者都高或都低
    
    for row in scene_rows:
        fp_rate = row.get('dl_fp_rate')
        fn_rate = row.get('dl_fn_rate')
        iou = row.get('dl_iou')
        
        # 只分析有指標值的樣本
        if fp_rate is None or fn_rate is None:
            continue
        
        # 定義閾值
        fp_high = fp_rate > 0.15
        fn_high = fn_rate > 0.05
        
        if fp_high and not fn_high:
            fp_cases.append(row)
        elif fn_high and not fp_high:
            fn_cases.append(row)
        else:
            balanced_cases.append(row)
    
    # 計算統計
    fp_count = len(fp_cases)
    fn_count = len(fn_cases)
    total_with_metrics = fp_count + fn_count + len(balanced_cases)
    
    # 找出最典型的失敗案例
    top_fp_cases = sorted(fp_cases, key=lambda x: x.get('dl_fp_rate', 0), reverse=True)[:3]
    top_fn_cases = sorted(fn_cases, key=lambda x: x.get('dl_fn_rate', 0), reverse=True)[:3]
    
    # 生成失敗模式描述
    failure_mode = generate_scene_failure_description(
        scene_name, fp_count, fn_count, total_with_metrics,
        top_fp_cases, top_fn_cases, scene_rows
    )
    
    return {
        'fp_cases': top_fp_cases,
        'fn_cases': top_fn_cases,
        'fp_count': fp_count,
        'fn_count': fn_count,
        'total': total_with_metrics,
        'failure_mode': failure_mode
    }, top_fp_cases, top_fn_cases


def generate_scene_failure_description(scene_name, fp_count, fn_count, total, top_fp_cases, top_fn_cases, all_rows):
    """根據 scene 類型生成失敗模式描述"""
    
    fp_ratio = fp_count / total if total > 0 else 0
    fn_ratio = fn_count / total if total > 0 else 0
    
    if scene_name == 'urban':
        # Urban: 分析是 FP 多還是 FN 多
        if fp_ratio > 0.5:
            # FP 主導
            fp_rates = [c.get('dl_fp_rate', 0) for c in top_fp_cases if c.get('dl_fp_rate')]
            avg_fp = sum(fp_rates) / len(fp_rates) if fp_rates else 0
            
            if avg_fp > 0.25:
                return "FP 主導：建築物頂部/玻璃反光/亮牆被誤判為天空（FP rate 高，建築邊緣誤判）"
            else:
                return "FP 主導：建築邊緣與天空混淆，誤判非天空為天空"
        elif fn_ratio > 0.3:
            # FN 主導
            fn_rates = [c.get('dl_fn_rate', 0) for c in top_fn_cases if c.get('dl_fn_rate')]
            avg_fn = sum(fn_rates) / len(fn_rates) if fn_rates else 0
            
            if avg_fn > 0.1:
                return "FN 主導：漏掉狹長天空區域或被建築物遮擋的天空（FN rate 高）"
            else:
                return "FN 主導：部分天空區域漏檢"
        else:
            # 平衡：檢查哪個更嚴重
            if fp_count > fn_count:
                return "FP 和 FN 都有，但 FP 較多：建築邊緣誤判（FP）為主，也有狹長天空漏檢（FN）"
            elif fn_count > fp_count:
                return "FP 和 FN 都有，但 FN 較多：狹長天空漏檢（FN）為主，也有建築邊緣誤判（FP）"
            else:
                return "FP 和 FN 都有：建築邊緣誤判（FP）與狹長天空漏檢（FN）並存"
    
    elif scene_name == 'forest':
        # Forest: 分析遮擋 vs 邊界
        if fp_ratio > 0.3:  # FP 比例 > 30% 就算 FP 主導
            # FP 主導
            fp_rates = [c.get('dl_fp_rate', 0) for c in top_fp_cases if c.get('dl_fp_rate')]
            avg_fp = sum(fp_rates) / len(fp_rates) if fp_rates else 0
            
            # 檢查 IoU 是否低（邊界破碎）
            ious = [r.get('dl_iou') for r in all_rows if r.get('dl_iou') is not None]
            avg_iou = sum(ious) / len(ious) if ious else 0
            
            if avg_iou < 0.7:
                return "FP 主導 + 邊界破碎：樹葉間隙被誤判為天空，且天空邊界被樹葉分割導致邊界抖動"
            else:
                return "FP 主導：樹葉間隙或亮葉被誤判為天空"
        elif fn_ratio > 0.2:  # FN 比例 > 20% 就算 FN 主導
            # FN 主導
            fn_rates = [c.get('dl_fn_rate', 0) for c in top_fn_cases if c.get('dl_fn_rate')]
            avg_fn = sum(fn_rates) / len(fn_rates) if fn_rates else 0
            
            if avg_fn > 0.1:
                return "FN 主導：樹葉/樹枝遮擋導致天空區域漏檢（FN rate 高）"
            else:
                return "FN 主導：部分天空區域被樹葉遮擋漏檢"
        else:
            # 平衡：檢查是否有邊界問題
            ious = [r.get('dl_iou') for r in all_rows if r.get('dl_iou') is not None]
            avg_iou = sum(ious) / len(ious) if ious else 0
            
            if avg_iou < 0.7:
                return "邊界破碎：天空邊界被樹葉分割，導致整體 IoU 下降"
            else:
                # 檢查哪個更多
                if fp_count > fn_count:
                    return "FP 和 FN 都有，但 FP 較多：樹葉間隙誤判（FP）為主，也有樹葉遮擋（FN）"
                else:
                    return "FP 和 FN 都有，但 FN 較多：樹葉遮擋（FN）為主，也有邊界破碎"
    
    elif scene_name == 'sea':
        # Sea: 分析水面誤判 vs 邊界抖動
        # 先檢查 IoU（邊界抖動）
        ious = [r.get('dl_iou') for r in all_rows if r.get('dl_iou') is not None]
        avg_iou = sum(ious) / len(ious) if ious else 0
        
        if fp_ratio > 0.5:
            # FP 主導
            fp_rates = [c.get('dl_fp_rate', 0) for c in top_fp_cases if c.get('dl_fp_rate')]
            avg_fp = sum(fp_rates) / len(fp_rates) if fp_rates else 0
            
            if avg_fp > 0.3:
                if avg_iou < 0.65:
                    return "FP 主導 + 邊界抖動：海面反光/波浪被誤判為天空（FP rate 高），且海天邊界不穩定導致邊界抖動"
                else:
                    return "FP 主導：海面反光/波浪被誤判為天空（FP rate 高，海天混淆）"
            else:
                if avg_iou < 0.65:
                    return "FP 主導 + 邊界抖動：水面與天空顏色相似導致誤判（FP），且海天邊界不穩定"
                else:
                    return "FP 主導：水面與天空顏色相似導致誤判"
        elif fn_ratio > 0.3:
            # FN 主導（較少見）
            return "FN 主導：部分天空區域漏檢"
        else:
            # 檢查邊界問題
            ious = [r.get('dl_iou') for r in all_rows if r.get('dl_iou') is not None]
            avg_iou = sum(ious) / len(ious) if ious else 0
            
            if avg_iou < 0.65:
                return "邊界抖動：海天邊界不穩定，預測邊界與真實邊界不一致"
            else:
                return "FP 和 FN 都有：海面誤判（FP）與邊界抖動並存"
    
    elif scene_name == 'other':
        # Other: 通用分析
        if fp_ratio > fn_ratio:
            return f"FP 主導：誤判非天空為天空（FP: {fp_count}, FN: {fn_count}）"
        elif fn_ratio > fp_ratio:
            return f"FN 主導：漏檢天空區域（FP: {fp_count}, FN: {fn_count}）"
        else:
            return f"FP 和 FN 平衡：兩種錯誤型態都有（FP: {fp_count}, FN: {fn_count}）"
    
    else:
        return "需要進一步分析"


def generate_scene_analysis_report(scene_analyses, output_md_path):
    """生成 scene failure mode 分析報告"""
    print(f"\n生成報告: {output_md_path}")
    
    with open(output_md_path, 'w', encoding='utf-8') as f:
        f.write("# Scene Failure Mode Analysis\n\n")
        f.write("> 針對每個 scene 分析錯誤型態（FP vs FN），而非平均分數\n\n")
        
        # 每個 scene 的分析
        for scene_name in ['urban', 'forest', 'sea', 'other']:
            if scene_name not in scene_analyses:
                continue
            
            analysis = scene_analyses[scene_name]
            fp_cases = analysis['fp_cases']
            fn_cases = analysis['fn_cases']
            
            f.write(f"## {scene_name.upper()} Scene\n\n")
            
            # 錯誤型態統計
            f.write("### 錯誤型態統計\n\n")
            f.write(f"- 總樣本數（有指標值）：{analysis['total']}\n")
            f.write(f"- **FP 主導**：{analysis['fp_count']} 張 ({analysis['fp_count']/analysis['total']*100:.1f}%)\n")
            f.write(f"- **FN 主導**：{analysis['fn_count']} 張 ({analysis['fn_count']/analysis['total']*100:.1f}%)\n\n")
            
            # 失敗模式描述
            f.write("### 失敗模式\n\n")
            f.write(f"**{analysis['failure_mode']}**\n\n")
            
            # FP 主導案例
            if len(fp_cases) > 0:
                f.write("#### FP 主導案例（誤判非天空為天空）\n\n")
                f.write("| Rank | Camera | Image | FP Rate | FN Rate | IoU | Overlay |\n")
                f.write("|------|--------|-------|---------|---------|-----|---------|\n")
                
                for i, row in enumerate(fp_cases, 1):
                    fp_rate = row.get('dl_fp_rate', 0)
                    fn_rate = row.get('dl_fn_rate', 0)
                    iou = row.get('dl_iou', 0)
                    overlay_path = row.get('overlay_path', '')
                    
                    fp_str = f"{fp_rate:.4f}" if fp_rate is not None else "-"
                    fn_str = f"{fn_rate:.4f}" if fn_rate is not None else "-"
                    iou_str = f"{iou:.4f}" if iou is not None else "-"
                    
                    overlay_link = f"[查看]({overlay_path})" if overlay_path else "-"
                    
                    f.write(f"| {i} | {row.get('camera_id', '?')} | {row.get('image_id', '?')} | {fp_str} | {fn_str} | {iou_str} | {overlay_link} |\n")
                
                f.write("\n")
            
            # FN 主導案例
            if len(fn_cases) > 0:
                f.write("#### FN 主導案例（漏檢天空）\n\n")
                f.write("| Rank | Camera | Image | FP Rate | FN Rate | IoU | Overlay |\n")
                f.write("|------|--------|-------|---------|---------|-----|---------|\n")
                
                for i, row in enumerate(fn_cases, 1):
                    fp_rate = row.get('dl_fp_rate', 0)
                    fn_rate = row.get('dl_fn_rate', 0)
                    iou = row.get('dl_iou', 0)
                    overlay_path = row.get('overlay_path', '')
                    
                    fp_str = f"{fp_rate:.4f}" if fp_rate is not None else "-"
                    fn_str = f"{fn_rate:.4f}" if fn_rate is not None else "-"
                    iou_str = f"{iou:.4f}" if iou is not None else "-"
                    
                    overlay_link = f"[查看]({overlay_path})" if overlay_path else "-"
                    
                    f.write(f"| {i} | {row.get('camera_id', '?')} | {row.get('image_id', '?')} | {fp_str} | {fn_str} | {iou_str} | {overlay_link} |\n")
                
                f.write("\n")
            
            f.write("---\n\n")
        
        # 總結
        f.write("## 總結\n\n")
        f.write("### 各 Scene 的錯誤型態\n\n")
        
        for scene_name in ['urban', 'forest', 'sea', 'other']:
            if scene_name in scene_analyses:
                analysis = scene_analyses[scene_name]
                f.write(f"- **{scene_name.upper()}**：{analysis['failure_mode']}\n")
        
        f.write("\n### 關鍵發現\n\n")
        f.write("- 不同 scene 的失敗模式不同，需要針對性的解決方案\n")
        f.write("- FP 主導的場景：需要更好的邊界識別或語義理解\n")
        f.write("- FN 主導的場景：需要更好的遮擋處理或邊界檢測\n")
        f.write("- 邊界問題：可能需要後處理或更精細的模型\n")
    
    print(f"  報告已生成")


def main():
    parser = argparse.ArgumentParser(description='Scene Failure Mode 分析')
    parser.add_argument('--input_csv', type=str, default='outputs/diagnostic_with_dl_metrics.csv',
                       help='輸入 CSV 路徑')
    parser.add_argument('--output_report', type=str, default='outputs/scene_failure_modes_report.md',
                       help='輸出 Markdown 報告路徑')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("  Scene Failure Mode Analysis")
    print("=" * 60)
    print()
    print(f"  輸入 CSV:     {args.input_csv}")
    print(f"  輸出報告:     {args.output_report}")
    print()
    
    # 讀取資料
    rows = load_diagnostic_data(args.input_csv)
    
    # 分析每個 scene
    scene_analyses = {}
    scenes = ['urban', 'forest', 'sea', 'other']
    
    for scene_name in scenes:
        print(f"\n分析 Scene: {scene_name}")
        print("-" * 60)
        
        # 篩選 scene 樣本（只分析 has_sky=TRUE 的）
        scene_rows = filter_scene_rows(rows, scene_name, has_sky=True)
        print(f"  樣本數: {len(scene_rows)}")
        
        if len(scene_rows) == 0:
            print(f"  跳過（無樣本）")
            continue
        
        # 分析 failure mode
        analysis, fp_cases, fn_cases = analyze_failure_mode(scene_rows, scene_name)
        
        if analysis:
            scene_analyses[scene_name] = analysis
            print(f"  FP 主導: {analysis['fp_count']} 張")
            print(f"  FN 主導: {analysis['fn_count']} 張")
            print(f"  失敗模式: {analysis['failure_mode']}")
    
    # 生成報告
    if scene_analyses:
        generate_scene_analysis_report(scene_analyses, args.output_report)
        
        print("\n" + "=" * 60)
        print("  完成")
        print("=" * 60)
        print(f"  分析 Scene 數: {len(scene_analyses)}")
        print(f"  報告: {args.output_report}")
    else:
        print("\n[警告] 沒有找到任何 scene 樣本")


if __name__ == '__main__':
    main()
