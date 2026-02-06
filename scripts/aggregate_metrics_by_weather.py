"""
依天氣欄位彙總 diagnostic 的 DL 指標

讀取 diagnostic_with_failure_modes.csv（或 diagnostic_with_dl_metrics.csv），
依 weather(clear|cloudy|rain|fog|snow|unknown) 分組，計算每組的樣本數與
dl_iou / dl_fp_rate / dl_fn_rate 的 mean，輸出 dl_metrics_by_weather.csv。

用法:
  python scripts/aggregate_metrics_by_weather.py
  python scripts/aggregate_metrics_by_weather.py --input outputs/diagnostic_with_dl_metrics.csv
"""

import os
import sys
import argparse
import csv
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

WEATHER_COL = 'weather(clear|cloudy|rain|fog|snow|unknown)'
METRIC_COLS = ['dl_iou', 'dl_fp_rate', 'dl_fn_rate']


def parse_float(value):
    if value == '' or value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def load_diagnostic(csv_path):
    rows = []
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            weather = row.get(WEATHER_COL, '').strip() or 'unknown'
            row['_weather'] = weather
            for col in METRIC_COLS:
                if col in row:
                    row[col] = parse_float(row[col])
            rows.append(row)
    return rows


def aggregate_by_weather(rows):
    by_weather = defaultdict(list)
    for row in rows:
        by_weather[row['_weather']].append(row)
    return by_weather


def compute_group_stats(group_rows):
    stats = {'count': len(group_rows)}
    for col in METRIC_COLS:
        values = [r[col] for r in group_rows if r.get(col) is not None]
        if values:
            stats[f'mean_{col}'] = sum(values) / len(values)
        else:
            stats[f'mean_{col}'] = None
    return stats


def main():
    parser = argparse.ArgumentParser(description='依天氣彙總 DL 指標')
    parser.add_argument('--input', type=str, default='outputs/diagnostic_with_failure_modes.csv',
                        help='輸入 CSV（含 weather 與 dl_* 欄位）')
    parser.add_argument('--output', type=str, default='outputs/dl_metrics_by_weather.csv',
                        help='輸出 CSV')
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[ERROR] 找不到輸入: {args.input}")
        sys.exit(1)

    print(f"讀取: {args.input}")
    rows = load_diagnostic(args.input)
    print(f"  總筆數: {len(rows)}")

    by_weather = aggregate_by_weather(rows)
    # 固定順序：clear, cloudy, rain, fog, snow, unknown，其餘按字母
    order = ['clear', 'cloudy', 'rain', 'fog', 'snow', 'unknown']
    weathers = [w for w in order if w in by_weather]
    weathers += sorted(k for k in by_weather if k not in order)

    out_rows = []
    for w in weathers:
        group = by_weather[w]
        st = compute_group_stats(group)
        out_rows.append({
            'weather': w,
            'count': st['count'],
            'mean_dl_iou': st['mean_dl_iou'],
            'mean_dl_fp_rate': st['mean_dl_fp_rate'],
            'mean_dl_fn_rate': st['mean_dl_fn_rate'],
        })

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    fieldnames = ['weather', 'count', 'mean_dl_iou', 'mean_dl_fp_rate', 'mean_dl_fn_rate']
    with open(args.output, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in out_rows:
            row = {k: r[k] if r.get(k) is not None else '' for k in fieldnames}
            w.writerow(row)

    print(f"輸出: {args.output}")
    print(f"  天氣組數: {len(out_rows)}")


if __name__ == '__main__':
    main()
