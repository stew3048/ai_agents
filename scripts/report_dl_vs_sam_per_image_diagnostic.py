"""
從 diagnostic_sam_metrics.csv 產出「每張圖」的 DL vs SAM 對照（IoU / FP / FN），
方便確認同一張圖上 SAM 與 DL 的 fp/fn 是否不一樣。

產出：outputs/dl_vs_sam_per_image_diagnostic.csv 與 .md
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


def main():
    path = 'outputs/diagnostic_sam_metrics.csv'
    if not os.path.exists(path):
        print(f"找不到 {path}，請先執行 eval_sam2_diagnostic_set.py 產出 diagnostic_sam_metrics.csv")
        sys.exit(1)

    rows = []
    with open(path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            row['dl_iou'] = parse_float(row.get('dl_iou'))
            row['dl_fp_rate'] = parse_float(row.get('dl_fp_rate'))
            row['dl_fn_rate'] = parse_float(row.get('dl_fn_rate'))
            row['sam2_iou'] = parse_float(row.get('sam2_iou'))
            row['sam2_fp_rate'] = parse_float(row.get('sam2_fp_rate'))
            row['sam2_fn_rate'] = parse_float(row.get('sam2_fn_rate'))
            rows.append(row)

    # 寫出 CSV：每張圖一行，DL 與 SAM 並排 + 差異
    out_csv = 'outputs/dl_vs_sam_per_image_diagnostic.csv'
    os.makedirs('outputs', exist_ok=True)
    fieldnames = [
        'camera_id', 'image_id', 'weather', 'derived_subset_name',
        'dl_iou', 'dl_fp_rate', 'dl_fn_rate',
        'sam2_iou', 'sam2_fp_rate', 'sam2_fn_rate',
        'iou_diff', 'fp_rate_diff', 'fn_rate_diff',
    ]
    with open(out_csv, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            di = r.get('dl_iou')
            si = r.get('sam2_iou')
            df = r.get('dl_fp_rate')
            sf = r.get('sam2_fp_rate')
            dn = r.get('dl_fn_rate')
            sn = r.get('sam2_fn_rate')
            iou_diff = (si - di) if (di is not None and si is not None) else None
            fp_diff = (sf - df) if (df is not None and sf is not None) else None
            fn_diff = (sn - dn) if (dn is not None and sn is not None) else None
            w.writerow({
                'camera_id': r.get('camera_id', ''),
                'image_id': r.get('image_id', ''),
                'weather': r.get('weather', ''),
                'derived_subset_name': r.get('derived_subset_name', ''),
                'dl_iou': f"{di:.6f}" if di is not None else '',
                'dl_fp_rate': f"{df:.6f}" if df is not None else '',
                'dl_fn_rate': f"{dn:.6f}" if dn is not None else '',
                'sam2_iou': f"{si:.6f}" if si is not None else '',
                'sam2_fp_rate': f"{sf:.6f}" if sf is not None else '',
                'sam2_fn_rate': f"{sn:.6f}" if sn is not None else '',
                'iou_diff': f"{iou_diff:+.6f}" if iou_diff is not None else '',
                'fp_rate_diff': f"{fp_diff:+.6f}" if fp_diff is not None else '',
                'fn_rate_diff': f"{fn_diff:+.6f}" if fn_diff is not None else '',
            })
    print(f"已寫入: {out_csv}（{len(rows)} 筆）")

    # 寫出簡短 Markdown 報告（前 20 筆 + 說明）
    out_md = 'outputs/dl_vs_sam_per_image_diagnostic.md'
    with open(out_md, 'w', encoding='utf-8') as f:
        f.write("# Diagnostic 每張圖：DL vs SAM（IoU / FP rate / FN rate）\n\n")
        f.write("資料來源：`outputs/diagnostic_sam_metrics.csv`（同一批 test/diagnostic 資料，DL 與 SAM 各跑一次）。\n\n")
        f.write("**同一張圖上 DL 與 SAM 的 FP/FN 通常不一樣**，因為兩者預測的 mask 不同。\n\n")
        f.write("| camera_id | image_id | weather | subset | DL IoU | DL FP | DL FN | SAM IoU | SAM FP | SAM FN | IoU diff | FP diff | FN diff |\n")
        f.write("|-----------|----------|--------|--------|--------|-------|-------|---------|--------|--------|----------|---------|----------|\n")
        for r in rows[:50]:  # 前 50 筆
            di, df, dn = r.get('dl_iou'), r.get('dl_fp_rate'), r.get('dl_fn_rate')
            si, sf, sn = r.get('sam2_iou'), r.get('sam2_fp_rate'), r.get('sam2_fn_rate')
            iou_diff = (si - di) if (di is not None and si is not None) else None
            fp_diff = (sf - df) if (df is not None and sf is not None) else None
            fn_diff = (sn - dn) if (dn is not None and sn is not None) else None
            dl_iou_s = f"{di:.4f}" if di is not None else "-"
            dl_fp_s = f"{df:.4f}" if df is not None else "-"
            dl_fn_s = f"{dn:.4f}" if dn is not None else "-"
            sam_iou_s = f"{si:.4f}" if si is not None else "-"
            sam_fp_s = f"{sf:.4f}" if sf is not None else "-"
            sam_fn_s = f"{sn:.4f}" if sn is not None else "-"
            iou_d = f"{iou_diff:+.4f}" if iou_diff is not None else "-"
            fp_d = f"{fp_diff:+.4f}" if fp_diff is not None else "-"
            fn_d = f"{fn_diff:+.4f}" if fn_diff is not None else "-"
            f.write(f"| {r.get('camera_id')} | {r.get('image_id')} | {r.get('weather', '')} | {r.get('derived_subset_name', '')} | {dl_iou_s} | {dl_fp_s} | {dl_fn_s} | {sam_iou_s} | {sam_fp_s} | {sam_fn_s} | {iou_d} | {fp_d} | {fn_d} |\n")
        f.write("\n（上表為前 50 筆；完整 166 筆見 `dl_vs_sam_per_image_diagnostic.csv`。）\n")
    print(f"已寫入: {out_md}")


if __name__ == '__main__':
    main()
