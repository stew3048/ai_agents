"""
比較 SAM legacy prompt 與 DINO prompt 在 25 筆 failure cases 上的表現。

讀取：
- outputs/sam2_failure_cases_metrics.csv（legacy）
- outputs/sam2_failure_cases_metrics_dino.csv（DINO prompt）

產出：簡短報告（整體與各 subset 的 IoU/FP/FN 平均，以及 DINO vs legacy 差異）。

用法:
  python scripts/compare_sam_legacy_vs_dino.py
"""

import os
import sys
import csv

sys.stdout.reconfigure(encoding='utf-8')


def parse_float(v):
    if v is None or v == '':
        return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def load_metrics(path):
    rows = []
    with open(path, 'r', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            row['sam2_iou'] = parse_float(row.get('sam2_iou'))
            row['sam2_fp_rate'] = parse_float(row.get('sam2_fp_rate'))
            row['sam2_fn_rate'] = parse_float(row.get('sam2_fn_rate'))
            rows.append(row)
    return rows


def main():
    legacy_path = 'outputs/sam2_failure_cases_metrics.csv'
    dino_path = 'outputs/sam2_failure_cases_metrics_dino.csv'
    if not os.path.exists(legacy_path):
        print(f"找不到 legacy: {legacy_path}")
        sys.exit(1)
    if not os.path.exists(dino_path):
        print(f"找不到 DINO: {dino_path}，請先執行: python scripts/eval_sam2_failure_cases.py --prompt dino")
        sys.exit(1)

    legacy = load_metrics(legacy_path)
    dino = load_metrics(dino_path)
    key = lambda r: (str(r.get('camera_id')), str(r.get('image_id')))
    dino_by_key = {key(r): r for r in dino}

    # 整體
    def avg(rows, col):
        vals = [r[col] for r in rows if r.get(col) is not None]
        return sum(vals) / len(vals) if vals else None

    print("=" * 60)
    print("  SAM Legacy vs DINO Prompt（25 failure cases）")
    print("=" * 60)
    print()
    l_iou = avg(legacy, 'sam2_iou')
    l_fp = avg(legacy, 'sam2_fp_rate')
    l_fn = avg(legacy, 'sam2_fn_rate')
    d_iou = avg(dino, 'sam2_iou')
    d_fp = avg(dino, 'sam2_fp_rate')
    d_fn = avg(dino, 'sam2_fn_rate')
    print("整體平均（25 筆）")
    print(f"  Legacy:  IoU={l_iou:.4f}  FP_rate={l_fp:.4f}  FN_rate={l_fn:.4f}")
    print(f"  DINO:    IoU={d_iou:.4f}  FP_rate={d_fp:.4f}  FN_rate={d_fn:.4f}")
    if l_iou and d_iou:
        print(f"  IoU 差異 (DINO - Legacy): {d_iou - l_iou:+.4f}")
    print()

    # 依 subset
    from collections import defaultdict
    l_by_sub = defaultdict(list)
    d_by_sub = defaultdict(list)
    for r in legacy:
        l_by_sub[r.get('subset_name', '')].append(r)
    for r in dino:
        d_by_sub[r.get('subset_name', '')].append(r)

    print("依情境 (subset)")
    print(f"  {'subset':<22} {'Legacy IoU':>10} {'DINO IoU':>10} {'IoU diff':>10}")
    print("  " + "-" * 54)
    for sub in ['no-sky', 'sea-sky-confusable', 'heavy-occlusion', 'night', 'urban']:
        lr = l_by_sub.get(sub, [])
        dr = d_by_sub.get(sub, [])
        li = avg(lr, 'sam2_iou')
        di = avg(dr, 'sam2_iou')
        diff = (di - li) if (li is not None and di is not None) else None
        li_s = f"{li:.4f}" if li is not None else "-"
        di_s = f"{di:.4f}" if di is not None else "-"
        diff_s = f"{diff:+.4f}" if diff is not None else "-"
        print(f"  {sub:<22} {li_s:>10} {di_s:>10} {diff_s:>10}")
    print()
    print("說明: IoU diff = DINO - Legacy，正表示 DINO prompt 在該情境平均較佳。")


if __name__ == '__main__':
    main()
