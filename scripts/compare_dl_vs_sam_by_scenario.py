"""
依「五大情境」彙總 DL vs SAM 的每張圖 IoU/FP/FN，產出比較報告：
同一張圖上兩者的差異 + 各情境下誰的整體表現較好。

資料來源：outputs/dl_vs_sam_per_image_diagnostic.csv
產出：outputs/dl_vs_sam_by_scenario_report.md、outputs/dl_vs_sam_by_scenario_summary.csv
"""

import os
import sys
import csv
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8')

# 五大情境 + default（其餘）
SCENARIO_ORDER = [
    'no-sky',
    'sea-sky-confusable',
    'heavy-occlusion',
    'night',
    'urban',
    'default',
]


def parse_float(v):
    if v is None or v == '':
        return None
    s = (v or '').strip()
    if not s:
        return None
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def load_per_image_csv(path):
    rows = []
    with open(path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            subset = (row.get('derived_subset_name') or '').strip()
            dl_iou = parse_float(row.get('dl_iou'))
            dl_fp = parse_float(row.get('dl_fp_rate'))
            dl_fn = parse_float(row.get('dl_fn_rate'))
            sam_iou = parse_float(row.get('sam2_iou'))
            sam_fp = parse_float(row.get('sam2_fp_rate'))
            sam_fn = parse_float(row.get('sam2_fn_rate'))
            rows.append({
                'camera_id': row.get('camera_id', ''),
                'image_id': row.get('image_id', ''),
                'weather': row.get('weather', ''),
                'subset': subset,
                'dl_iou': dl_iou,
                'dl_fp_rate': dl_fp,
                'dl_fn_rate': dl_fn,
                'sam2_iou': sam_iou,
                'sam2_fp_rate': sam_fp,
                'sam2_fn_rate': sam_fn,
            })
    return rows


def main():
    path = 'outputs/dl_vs_sam_per_image_diagnostic.csv'
    if not os.path.exists(path):
        print(f"找不到 {path}，請先執行 report_dl_vs_sam_per_image_diagnostic.py")
        sys.exit(1)

    rows = load_per_image_csv(path)
    by_subset = defaultdict(list)
    for r in rows:
        by_subset[r['subset']].append(r)

    os.makedirs('outputs', exist_ok=True)

    # ---- 1) 彙總各情境的平均 IoU / FP / FN 與「誰贏」
    summary_rows = []
    for scenario in SCENARIO_ORDER:
        group = by_subset.get(scenario, [])
        if not group:
            summary_rows.append({
                'scenario': scenario,
                'n': 0,
                'dl_iou': None, 'dl_fp': None, 'dl_fn': None,
                'sam_iou': None, 'sam_fp': None, 'sam_fn': None,
                'winner_iou': '-', 'winner_fp': '-', 'winner_fn': '-',
                'dl_win_iou': 0, 'sam_win_iou': 0, 'tie_iou': 0,
            })
            continue

        dl_ious = [r['dl_iou'] for r in group if r['dl_iou'] is not None]
        dl_fps = [r['dl_fp_rate'] for r in group if r['dl_fp_rate'] is not None]
        dl_fns = [r['dl_fn_rate'] for r in group if r['dl_fn_rate'] is not None]
        sam_ious = [r['sam2_iou'] for r in group if r['sam2_iou'] is not None]
        sam_fps = [r['sam2_fp_rate'] for r in group if r['sam2_fp_rate'] is not None]
        sam_fns = [r['sam2_fn_rate'] for r in group if r['sam2_fn_rate'] is not None]

        avg_dl_iou = sum(dl_ious) / len(dl_ious) if dl_ious else None
        avg_dl_fp = sum(dl_fps) / len(dl_fps) if dl_fps else None
        avg_dl_fn = sum(dl_fns) / len(dl_fns) if dl_fns else None
        avg_sam_iou = sum(sam_ious) / len(sam_ious) if sam_ious else None
        avg_sam_fp = sum(sam_fps) / len(sam_fps) if sam_fps else None
        avg_sam_fn = sum(sam_fns) / len(sam_fns) if sam_fns else None

        winner_iou = '-'
        if avg_dl_iou is not None and avg_sam_iou is not None:
            winner_iou = 'SAM' if avg_sam_iou > avg_dl_iou else ('DL' if avg_dl_iou > avg_sam_iou else 'tie')
        winner_fp = '-'
        if avg_dl_fp is not None and avg_sam_fp is not None:
            winner_fp = 'SAM' if avg_sam_fp < avg_dl_fp else ('DL' if avg_dl_fp < avg_sam_fp else 'tie')
        winner_fn = '-'
        if avg_dl_fn is not None and avg_sam_fn is not None:
            winner_fn = 'SAM' if avg_sam_fn < avg_dl_fn else ('DL' if avg_dl_fn < avg_sam_fn else 'tie')

        dl_win_iou = sam_win_iou = tie_iou = 0
        for r in group:
            di, si = r.get('dl_iou'), r.get('sam2_iou')
            if di is not None and si is not None:
                if si > di:
                    sam_win_iou += 1
                elif di > si:
                    dl_win_iou += 1
                else:
                    tie_iou += 1

        summary_rows.append({
            'scenario': scenario,
            'n': len(group),
            'dl_iou': avg_dl_iou, 'dl_fp': avg_dl_fp, 'dl_fn': avg_dl_fn,
            'sam_iou': avg_sam_iou, 'sam_fp': avg_sam_fp, 'sam_fn': avg_sam_fn,
            'winner_iou': winner_iou, 'winner_fp': winner_fp, 'winner_fn': winner_fn,
            'dl_win_iou': dl_win_iou, 'sam_win_iou': sam_win_iou, 'tie_iou': tie_iou,
        })

    # 寫出 summary CSV
    summary_csv = 'outputs/dl_vs_sam_by_scenario_summary.csv'
    with open(summary_csv, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f)
        w.writerow([
            'scenario', 'n',
            'dl_avg_iou', 'dl_avg_fp_rate', 'dl_avg_fn_rate',
            'sam_avg_iou', 'sam_avg_fp_rate', 'sam_avg_fn_rate',
            'winner_iou', 'winner_fp', 'winner_fn',
            'dl_win_iou_count', 'sam_win_iou_count', 'tie_iou_count',
        ])
        for s in summary_rows:
            w.writerow([
                s['scenario'], s['n'],
                f"{s['dl_iou']:.6f}" if s['dl_iou'] is not None else '',
                f"{s['dl_fp']:.6f}" if s['dl_fp'] is not None else '',
                f"{s['dl_fn']:.6f}" if s['dl_fn'] is not None else '',
                f"{s['sam_iou']:.6f}" if s['sam_iou'] is not None else '',
                f"{s['sam_fp']:.6f}" if s['sam_fp'] is not None else '',
                f"{s['sam_fn']:.6f}" if s['sam_fn'] is not None else '',
                s['winner_iou'], s['winner_fp'], s['winner_fn'],
                s['dl_win_iou'], s['sam_win_iou'], s['tie_iou'],
            ])
    print(f"已寫入: {summary_csv}")

    # ---- 2) Markdown 報告
    md_path = 'outputs/dl_vs_sam_by_scenario_report.md'
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# DL vs SAM 依五大情境比較報告\n\n")
        f.write("資料：同一批 diagnostic 圖，每張分別跑 DL 與 SAM，比較 IoU / FP rate / FN rate。\n\n")
        f.write("---\n\n")

        # 總表：各情境平均與誰贏
        f.write("## 各情境平均指標與勝出方\n\n")
        f.write("| 情境 | 樣本數 | DL IoU | DL FP | DL FN | SAM IoU | SAM FP | SAM FN | 誰 IoU 較高 | 誰 FP 較低 | 誰 FN 較低 |\n")
        f.write("|------|--------|--------|-------|-------|---------|--------|--------|--------------|------------|------------|\n")
        for s in summary_rows:
            if s['n'] == 0:
                f.write(f"| {s['scenario']} | 0 | - | - | - | - | - | - | - | - | - |\n")
                continue
            dl_iou_s = f"{s['dl_iou']:.4f}" if s['dl_iou'] is not None else '-'
            dl_fp_s = f"{s['dl_fp']:.4f}" if s['dl_fp'] is not None else '-'
            dl_fn_s = f"{s['dl_fn']:.4f}" if s['dl_fn'] is not None else '-'
            sam_iou_s = f"{s['sam_iou']:.4f}" if s['sam_iou'] is not None else '-'
            sam_fp_s = f"{s['sam_fp']:.4f}" if s['sam_fp'] is not None else '-'
            sam_fn_s = f"{s['sam_fn']:.4f}" if s['sam_fn'] is not None else '-'
            f.write(f"| {s['scenario']} | {s['n']} | {dl_iou_s} | {dl_fp_s} | {dl_fn_s} | {sam_iou_s} | {sam_fp_s} | {sam_fn_s} | {s['winner_iou']} | {s['winner_fp']} | {s['winner_fn']} |\n")

        f.write("\n---\n\n")
        f.write("## 各情境「每張圖」IoU 誰贏的次數\n\n")
        f.write("| 情境 | 樣本數 | DL 贏 (IoU 較高) | SAM 贏 (IoU 較高) | 平手 |\n")
        f.write("|------|--------|------------------|-------------------|------|\n")
        for s in summary_rows:
            f.write(f"| {s['scenario']} | {s['n']} | {s['dl_win_iou']} | {s['sam_win_iou']} | {s['tie_iou']} |\n")

        f.write("\n---\n\n")
        f.write("## 各情境結論：誰表現較好\n\n")
        for s in summary_rows:
            if s['n'] == 0:
                f.write(f"### {s['scenario']}\n\n（此 diagnostic 集內無樣本）\n\n")
                continue
            f.write(f"### {s['scenario']}（共 {s['n']} 張）\n\n")
            iou_w = s['winner_iou']
            fp_w = s['winner_fp']
            fn_w = s['winner_fn']
            # 綜合：若三項裡多數是同一方贏，則該方整體較好
            wins = {'DL': 0, 'SAM': 0}
            if iou_w in ('DL', 'SAM'):
                wins[iou_w] += 1
            if fp_w in ('DL', 'SAM'):
                wins[fp_w] += 1
            if fn_w in ('DL', 'SAM'):
                wins[fn_w] += 1
            if wins['DL'] > wins['SAM']:
                overall = "**DL 整體較好**（在 IoU/FP/FN 三項中贏的次數較多）"
            elif wins['SAM'] > wins['DL']:
                overall = "**SAM 整體較好**（在 IoU/FP/FN 三項中贏的次數較多）"
            else:
                overall = "**兩者互有勝負**（依指標不同而異）"
            f.write(f"- **IoU 較高**：{iou_w}\n")
            f.write(f"- **FP 較低**：{fp_w}\n")
            f.write(f"- **FN 較低**：{fn_w}\n")
            f.write(f"- **綜合**：{overall}\n\n")

    print(f"已寫入: {md_path}")


if __name__ == '__main__':
    main()
