"""
依天氣分情境產出 DL vs SAM 對比報告

讀取 dl_metrics_by_weather.csv（DL 依天氣彙總）與 diagnostic_sam_metrics.csv（全 diagnostic 的 SAM 結果），
將 SAM 依 weather 彙總後與 DL 對齊，產出 dl_vs_sam_by_weather_report.md 與 dl_vs_sam_by_weather.csv。

用法:
  python scripts/generate_dl_vs_sam_by_weather.py
"""

import os
import sys
import argparse
import csv
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')


def parse_float(value):
    if value == '' or value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def load_dl_by_weather(csv_path):
    """回傳 list of dict: weather, count, mean_dl_iou, mean_dl_fp_rate, mean_dl_fn_rate"""
    rows = []
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            row['count'] = int(row.get('count', 0) or 0)
            row['mean_dl_iou'] = parse_float(row.get('mean_dl_iou'))
            row['mean_dl_fp_rate'] = parse_float(row.get('mean_dl_fp_rate'))
            row['mean_dl_fn_rate'] = parse_float(row.get('mean_dl_fn_rate'))
            rows.append(row)
    return rows


def aggregate_sam_by_weather(csv_path):
    """
    讀取 diagnostic_sam_metrics.csv，依 weather 分組計算 mean(sam2_iou), mean(sam2_fp_rate), mean(sam2_fn_rate)。
    回傳 dict: weather -> { count, mean_sam2_iou, mean_sam2_fp_rate, mean_sam2_fn_rate }
    """
    by_weather = defaultdict(lambda: {'iou': [], 'fp': [], 'fn': [], 'count': 0})
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            w = (row.get('weather') or '').strip() or 'unknown'
            by_weather[w]['count'] += 1
            iou = parse_float(row.get('sam2_iou'))
            fp = parse_float(row.get('sam2_fp_rate'))
            fn = parse_float(row.get('sam2_fn_rate'))
            if iou is not None:
                by_weather[w]['iou'].append(iou)
            if fp is not None:
                by_weather[w]['fp'].append(fp)
            if fn is not None:
                by_weather[w]['fn'].append(fn)
    result = {}
    for w, v in by_weather.items():
        result[w] = {
            'count': v['count'],
            'mean_sam2_iou': sum(v['iou']) / len(v['iou']) if v['iou'] else None,
            'mean_sam2_fp_rate': sum(v['fp']) / len(v['fp']) if v['fp'] else None,
            'mean_sam2_fn_rate': sum(v['fn']) / len(v['fn']) if v['fn'] else None,
        }
    return result


def main():
    parser = argparse.ArgumentParser(description='依天氣產出 DL vs SAM 對比報告')
    parser.add_argument('--dl_csv', type=str, default='outputs/dl_metrics_by_weather.csv')
    parser.add_argument('--sam_csv', type=str, default='outputs/diagnostic_sam_metrics.csv')
    parser.add_argument('--sam_agg_csv', type=str, default='outputs/sam_metrics_by_weather.csv',
                        help='可選：輸出的 SAM 依天氣彙總 CSV')
    parser.add_argument('--output_report', type=str, default='outputs/dl_vs_sam_by_weather_report.md')
    parser.add_argument('--output_csv', type=str, default='outputs/dl_vs_sam_by_weather.csv')
    args = parser.parse_args()

    if not os.path.exists(args.dl_csv):
        print(f"[ERROR] 找不到 DL 彙總: {args.dl_csv}")
        sys.exit(1)
    if not os.path.exists(args.sam_csv):
        print(f"[ERROR] 找不到 SAM 結果: {args.sam_csv}")
        sys.exit(1)

    dl_rows = load_dl_by_weather(args.dl_csv)
    sam_by_weather = aggregate_sam_by_weather(args.sam_csv)

    # 固定天氣順序，與 aggregate_metrics_by_weather 一致
    order = ['clear', 'cloudy', 'rain', 'fog', 'snow', 'unknown']
    all_weathers = set(r['weather'] for r in dl_rows) | set(sam_by_weather.keys())
    weathers = [w for w in order if w in all_weathers]
    weathers += sorted(w for w in all_weathers if w not in order)

    dl_lookup = {r['weather']: r for r in dl_rows}
    combined = []
    for w in weathers:
        dl = dl_lookup.get(w, {})
        sam = sam_by_weather.get(w, {})
        row = {
            'weather': w,
            'count': dl.get('count') or sam.get('count', 0),
            'sam_count': sam.get('count', 0),
            'mean_dl_iou': dl.get('mean_dl_iou'),
            'mean_dl_fp_rate': dl.get('mean_dl_fp_rate'),
            'mean_dl_fn_rate': dl.get('mean_dl_fn_rate'),
            'mean_sam2_iou': sam.get('mean_sam2_iou'),
            'mean_sam2_fp_rate': sam.get('mean_sam2_fp_rate'),
            'mean_sam2_fn_rate': sam.get('mean_sam2_fn_rate'),
        }
        if row['mean_dl_iou'] is not None and row['mean_sam2_iou'] is not None:
            row['iou_diff'] = row['mean_sam2_iou'] - row['mean_dl_iou']
        else:
            row['iou_diff'] = None
        combined.append(row)

    # 寫入 sam_metrics_by_weather.csv（SAM 依天氣彙總）
    os.makedirs(os.path.dirname(args.sam_agg_csv) or '.', exist_ok=True)
    with open(args.sam_agg_csv, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['weather', 'count', 'mean_sam2_iou', 'mean_sam2_fp_rate', 'mean_sam2_fn_rate'])
        w.writeheader()
        for r in combined:
            w.writerow({
                'weather': r['weather'],
                'count': r['sam_count'],
                'mean_sam2_iou': r['mean_sam2_iou'] if r['mean_sam2_iou'] is not None else '',
                'mean_sam2_fp_rate': r['mean_sam2_fp_rate'] if r['mean_sam2_fp_rate'] is not None else '',
                'mean_sam2_fn_rate': r['mean_sam2_fn_rate'] if r['mean_sam2_fn_rate'] is not None else '',
            })
    print(f"已寫入 SAM 彙總: {args.sam_agg_csv}")

    # 寫入 dl_vs_sam_by_weather.csv
    os.makedirs(os.path.dirname(args.output_csv) or '.', exist_ok=True)
    fieldnames = ['weather', 'count', 'mean_dl_iou', 'mean_dl_fp_rate', 'mean_dl_fn_rate',
                  'mean_sam2_iou', 'mean_sam2_fp_rate', 'mean_sam2_fn_rate', 'iou_diff']
    with open(args.output_csv, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in combined:
            out = {k: (r[k] if r.get(k) is not None else '') for k in fieldnames}
            w.writerow(out)
    print(f"已寫入對比 CSV: {args.output_csv}")

    # 撰寫報告
    os.makedirs(os.path.dirname(args.output_report) or '.', exist_ok=True)
    with open(args.output_report, 'w', encoding='utf-8') as f:
        f.write("# DL vs SAM 2.0 依天氣分情境對比報告\n\n")
        f.write("> 資料來源：diagnostic set 依 `weather(clear|cloudy|rain|fog|snow|unknown)` 分組，")
        f.write("DL 為既有指標彙總，SAM 為對全 diagnostic 跑 SAM 2.0 後依天氣彙總。\n\n")
        f.write("## 各天氣彙總表\n\n")
        f.write("| 天氣 | 樣本數 | DL IoU | DL FP rate | DL FN rate | SAM IoU | SAM FP rate | SAM FN rate | IoU 差異 |\n")
        f.write("|------|--------|--------|------------|------------|--------|--------------|--------------|----------|\n")
        for r in combined:
            dl_iou = f"{r['mean_dl_iou']:.4f}" if r['mean_dl_iou'] is not None else "-"
            dl_fp = f"{r['mean_dl_fp_rate']:.4f}" if r['mean_dl_fp_rate'] is not None else "-"
            dl_fn = f"{r['mean_dl_fn_rate']:.4f}" if r['mean_dl_fn_rate'] is not None else "-"
            sam_iou = f"{r['mean_sam2_iou']:.4f}" if r['mean_sam2_iou'] is not None else "-"
            sam_fp = f"{r['mean_sam2_fp_rate']:.4f}" if r['mean_sam2_fp_rate'] is not None else "-"
            sam_fn = f"{r['mean_sam2_fn_rate']:.4f}" if r['mean_sam2_fn_rate'] is not None else "-"
            diff = f"{r['iou_diff']:+.4f}" if r['iou_diff'] is not None else "-"
            f.write(f"| {r['weather']} | {r['count']} | {dl_iou} | {dl_fp} | {dl_fn} | {sam_iou} | {sam_fp} | {sam_fn} | {diff} |\n")
        f.write("\n## 說明\n\n")
        f.write("- **DL**：來自 `outputs/dl_metrics_by_weather.csv`（diagnostic 既有 dl_* 依天氣 mean）。\n")
        f.write("- **SAM**：來自 `outputs/diagnostic_sam_metrics.csv` 依 weather 分組 mean。\n")
        f.write("- **IoU 差異**：SAM IoU − DL IoU（正表示該天氣下 SAM 平均優於 DL）。\n")
    print(f"已寫入報告: {args.output_report}")


if __name__ == '__main__':
    main()
