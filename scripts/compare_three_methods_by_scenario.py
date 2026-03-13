"""
依「五大情境」彙總 DL / SAM / CLIPSeg 三方法的每張圖 IoU/FP/FN，產出三方法比較報告。

資料來源：合併 diagnostic_sam_metrics.csv（DL + SAM）與 diagnostic_clipseg_metrics.csv（DL + CLIPSeg）
產出：outputs/three_methods_by_scenario_summary.csv、outputs/three_methods_by_scenario_report.md
"""

import os
import sys
import csv
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8')

SCENARIO_ORDER = [
    'no-sky',
    'sea-sky-confusable',
    'heavy-occlusion',
    'night',
    'urban',
    'default',
]

METHODS = ['DL', 'SAM', 'CLIPSeg']


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


def load_merged_three_methods(sam_csv, clipseg_csv):
    """
    讀取 SAM 與 CLIPSeg 的 diagnostic CSV，以 (camera_id, image_id) 合併，
    每筆含 dl_*, sam2_*, clipseg_* 與 derived_subset_name。
    """
    # SAM CSV: camera_id, image_id, dl_*, sam2_*, derived_subset_name, weather
    sam_rows = {}
    with open(sam_csv, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row.get('camera_id', ''), row.get('image_id', ''))
            sam_rows[key] = {
                'camera_id': row.get('camera_id', ''),
                'image_id': row.get('image_id', ''),
                'weather': row.get('weather', '') or 'unknown',
                'subset': (row.get('derived_subset_name') or '').strip(),
                'dl_iou': parse_float(row.get('dl_iou')),
                'dl_fp_rate': parse_float(row.get('dl_fp_rate')),
                'dl_fn_rate': parse_float(row.get('dl_fn_rate')),
                'sam2_iou': parse_float(row.get('sam2_iou')),
                'sam2_fp_rate': parse_float(row.get('sam2_fp_rate')),
                'sam2_fn_rate': parse_float(row.get('sam2_fn_rate')),
            }

    # CLIPSeg CSV: 補上 clipseg_*
    with open(clipseg_csv, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row.get('camera_id', ''), row.get('image_id', ''))
            if key not in sam_rows:
                continue
            sam_rows[key]['clipseg_iou'] = parse_float(row.get('clipseg_iou'))
            sam_rows[key]['clipseg_fp_rate'] = parse_float(row.get('clipseg_fp_rate'))
            sam_rows[key]['clipseg_fn_rate'] = parse_float(row.get('clipseg_fn_rate'))

    # 只保留三種都有 clipseg 的（表示兩份 CSV 都有這筆）
    rows = []
    for key, r in sam_rows.items():
        if r.get('clipseg_iou') is not None or r.get('clipseg_fp_rate') is not None:
            rows.append(r)
        else:
            # 若 CLIPSeg 缺漏，仍列入但 clipseg 為 None
            rows.append(r)
    return rows


def main():
    sam_csv = 'outputs/diagnostic_sam_metrics.csv'
    clipseg_csv = 'outputs/diagnostic_clipseg_metrics.csv'
    if not os.path.exists(sam_csv):
        print(f"找不到 {sam_csv}，請先執行 eval_sam2_diagnostic_set.py")
        sys.exit(1)
    if not os.path.exists(clipseg_csv):
        print(f"找不到 {clipseg_csv}，請先執行 eval_clipseg_diagnostic_set.py")
        sys.exit(1)

    rows = load_merged_three_methods(sam_csv, clipseg_csv)
    print(f"合併後共 {len(rows)} 筆（DL + SAM + CLIPSeg）")

    by_subset = defaultdict(list)
    for r in rows:
        by_subset[r['subset']].append(r)

    os.makedirs('outputs', exist_ok=True)

    # ---- 各情境平均與「三方法誰贏」
    summary_rows = []
    for scenario in SCENARIO_ORDER:
        group = by_subset.get(scenario, [])
        if not group:
            summary_rows.append({
                'scenario': scenario,
                'n': 0,
                'dl_iou': None, 'dl_fp': None, 'dl_fn': None,
                'sam_iou': None, 'sam_fp': None, 'sam_fn': None,
                'clipseg_iou': None, 'clipseg_fp': None, 'clipseg_fn': None,
                'winner_iou': '-', 'winner_fp': '-', 'winner_fn': '-',
                'dl_win_iou': 0, 'sam_win_iou': 0, 'clipseg_win_iou': 0, 'tie_iou': 0,
            })
            continue

        def avg(vals):
            vals = [x for x in vals if x is not None]
            return sum(vals) / len(vals) if vals else None

        dl_ious = [r['dl_iou'] for r in group]
        dl_fps = [r['dl_fp_rate'] for r in group]
        dl_fns = [r['dl_fn_rate'] for r in group]
        sam_ious = [r['sam2_iou'] for r in group]
        sam_fps = [r['sam2_fp_rate'] for r in group]
        sam_fns = [r['sam2_fn_rate'] for r in group]
        clipseg_ious = [r.get('clipseg_iou') for r in group]
        clipseg_fps = [r.get('clipseg_fp_rate') for r in group]
        clipseg_fns = [r.get('clipseg_fn_rate') for r in group]

        avg_dl_iou = avg(dl_ious)
        avg_dl_fp = avg(dl_fps)
        avg_dl_fn = avg(dl_fns)
        avg_sam_iou = avg(sam_ious)
        avg_sam_fp = avg(sam_fps)
        avg_sam_fn = avg(sam_fns)
        avg_clipseg_iou = avg(clipseg_ious)
        avg_clipseg_fp = avg(clipseg_fps)
        avg_clipseg_fn = avg(clipseg_fns)

        def winner_iou(dl, sam, cg):
            if dl is None and sam is None and cg is None:
                return '-'
            vals = [(dl, 'DL'), (sam, 'SAM'), (cg, 'CLIPSeg')]
            vals = [(v, name) for v, name in vals if v is not None]
            if not vals:
                return '-'
            best = max(vals, key=lambda x: x[0])
            ties = [name for v, name in vals if v == best[0]]
            return ties[0] if len(ties) == 1 else 'tie'

        def winner_fp(dl, sam, cg):
            if dl is None and sam is None and cg is None:
                return '-'
            vals = [(dl, 'DL'), (sam, 'SAM'), (cg, 'CLIPSeg')]
            vals = [(v, name) for v, name in vals if v is not None]
            if not vals:
                return '-'
            best = min(vals, key=lambda x: x[0])
            ties = [name for v, name in vals if v == best[0]]
            return ties[0] if len(ties) == 1 else 'tie'

        winner_fn = winner_fp  # 愈低愈好

        dl_win = sam_win = clipseg_win = tie = 0
        for r in group:
            di, si, ci = r.get('dl_iou'), r.get('sam2_iou'), r.get('clipseg_iou')
            valid = [(di, 'DL'), (si, 'SAM'), (ci, 'CLIPSeg')]
            valid = [(v, name) for v, name in valid if v is not None]
            if not valid:
                continue
            best_val = max(v for v, _ in valid)
            winners = [name for v, name in valid if v == best_val]
            if len(winners) == 1:
                if winners[0] == 'DL':
                    dl_win += 1
                elif winners[0] == 'SAM':
                    sam_win += 1
                else:
                    clipseg_win += 1
            else:
                tie += 1

        summary_rows.append({
            'scenario': scenario,
            'n': len(group),
            'dl_iou': avg_dl_iou, 'dl_fp': avg_dl_fp, 'dl_fn': avg_dl_fn,
            'sam_iou': avg_sam_iou, 'sam_fp': avg_sam_fp, 'sam_fn': avg_sam_fn,
            'clipseg_iou': avg_clipseg_iou, 'clipseg_fp': avg_clipseg_fp, 'clipseg_fn': avg_clipseg_fn,
            'winner_iou': winner_iou(avg_dl_iou, avg_sam_iou, avg_clipseg_iou),
            'winner_fp': winner_fp(avg_dl_fp, avg_sam_fp, avg_clipseg_fp),
            'winner_fn': winner_fn(avg_dl_fn, avg_sam_fn, avg_clipseg_fn),
            'dl_win_iou': dl_win, 'sam_win_iou': sam_win, 'clipseg_win_iou': clipseg_win, 'tie_iou': tie,
        })

    # ---- 寫出 summary CSV
    summary_csv = 'outputs/three_methods_by_scenario_summary.csv'
    with open(summary_csv, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f)
        w.writerow([
            'scenario', 'n',
            'dl_avg_iou', 'dl_avg_fp_rate', 'dl_avg_fn_rate',
            'sam_avg_iou', 'sam_avg_fp_rate', 'sam_avg_fn_rate',
            'clipseg_avg_iou', 'clipseg_avg_fp_rate', 'clipseg_avg_fn_rate',
            'winner_iou', 'winner_fp', 'winner_fn',
            'dl_win_iou_count', 'sam_win_iou_count', 'clipseg_win_iou_count', 'tie_iou_count',
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
                f"{s['clipseg_iou']:.6f}" if s['clipseg_iou'] is not None else '',
                f"{s['clipseg_fp']:.6f}" if s['clipseg_fp'] is not None else '',
                f"{s['clipseg_fn']:.6f}" if s['clipseg_fn'] is not None else '',
                s['winner_iou'], s['winner_fp'], s['winner_fn'],
                s['dl_win_iou'], s['sam_win_iou'], s['clipseg_win_iou'], s['tie_iou'],
            ])
    print(f"已寫入: {summary_csv}")

    # ---- Markdown 報告
    md_path = 'outputs/three_methods_by_scenario_report.md'
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# DL / SAM / CLIPSeg 三方法依情境比較報告\n\n")
        f.write("資料：同一批 diagnostic 圖，每張分別跑 DL（U-Net）、SAM 2.0、CLIPSeg（VLM），比較 IoU / FP rate / FN rate。\n\n")
        f.write("---\n\n")

        f.write("## 各情境平均指標與勝出方\n\n")
        f.write("| 情境 | 樣本數 | DL IoU | DL FP | DL FN | SAM IoU | SAM FP | SAM FN | CLIPSeg IoU | CLIPSeg FP | CLIPSeg FN | 誰 IoU 較高 | 誰 FP 較低 | 誰 FN 較低 |\n")
        f.write("|------|--------|--------|-------|-------|---------|--------|--------|-------------|------------|------------|--------------|------------|------------|\n")
        for s in summary_rows:
            if s['n'] == 0:
                f.write(f"| {s['scenario']} | 0 | - | - | - | - | - | - | - | - | - | - | - | - |\n")
                continue
            def fmt(v):
                return f"{v:.4f}" if v is not None else '-'
            f.write(f"| {s['scenario']} | {s['n']} | {fmt(s['dl_iou'])} | {fmt(s['dl_fp'])} | {fmt(s['dl_fn'])} | "
                    f"{fmt(s['sam_iou'])} | {fmt(s['sam_fp'])} | {fmt(s['sam_fn'])} | "
                    f"{fmt(s['clipseg_iou'])} | {fmt(s['clipseg_fp'])} | {fmt(s['clipseg_fn'])} | "
                    f"{s['winner_iou']} | {s['winner_fp']} | {s['winner_fn']} |\n")

        f.write("\n---\n\n")
        f.write("## 各情境「每張圖」IoU 誰贏的次數\n\n")
        f.write("| 情境 | 樣本數 | DL 贏 | SAM 贏 | CLIPSeg 贏 | 平手 |\n")
        f.write("|------|--------|-------|--------|------------|------|\n")
        for s in summary_rows:
            f.write(f"| {s['scenario']} | {s['n']} | {s['dl_win_iou']} | {s['sam_win_iou']} | {s['clipseg_win_iou']} | {s['tie_iou']} |\n")

        f.write("\n---\n\n")
        f.write("## 各情境結論：誰表現較好\n\n")
        for s in summary_rows:
            if s['n'] == 0:
                f.write(f"### {s['scenario']}\n\n（此 diagnostic 集內無樣本）\n\n")
                continue
            f.write(f"### {s['scenario']}（共 {s['n']} 張）\n\n")
            f.write(f"- **IoU 較高**：{s['winner_iou']}\n")
            f.write(f"- **FP 較低**：{s['winner_fp']}\n")
            f.write(f"- **FN 較低**：{s['winner_fn']}\n")
            # 綜合：三項中哪個方法贏最多次
            wins = {'DL': 0, 'SAM': 0, 'CLIPSeg': 0}
            for key in ('winner_iou', 'winner_fp', 'winner_fn'):
                w = s[key]
                if w in wins:
                    wins[w] += 1
            if wins['DL'] > wins['SAM'] and wins['DL'] > wins['CLIPSeg']:
                overall = "**DL 整體較好**"
            elif wins['SAM'] > wins['DL'] and wins['SAM'] > wins['CLIPSeg']:
                overall = "**SAM 整體較好**"
            elif wins['CLIPSeg'] > wins['DL'] and wins['CLIPSeg'] > wins['SAM']:
                overall = "**CLIPSeg 整體較好**"
            else:
                overall = "**互有勝負**（依指標而異）"
            f.write(f"- **綜合**：{overall}\n\n")

    print(f"已寫入: {md_path}")


if __name__ == '__main__':
    main()
