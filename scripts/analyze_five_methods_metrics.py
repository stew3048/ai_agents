"""
分析五方法評估結果：整體與 by-camera 的 IoU / FP rate / FN rate。

輸入：outputs/eval_five_methods/per_image_metrics.csv
輸出：
  - 每方法整體平均 IoU / fp_rate / fn_rate
  - 每方法 × 每 camera 的 IoU / fp_rate / fn_rate
  - 每 camera 跨方法平均（辨識哪個 camera 預測難度高）
"""

import os
import sys
import csv
import json
import argparse
from collections import defaultdict

_script_dir = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_script_dir)
sys.path.insert(0, _root)


def load_csv(path):
    """回傳 List[Dict]，數值欄位轉 float（空字串為 None）。"""
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        r = csv.DictReader(f)
        for row in r:
            for key in ('iou', 'fp_rate', 'fn_rate', 'dice', 'bce'):
                if key in row and row[key].strip() != '':
                    try:
                        row[key] = float(row[key])
                    except ValueError:
                        row[key] = None
                else:
                    row[key] = None
            row['camera_id'] = str(row['camera_id']).strip()
            rows.append(row)
    return rows


def analyze(rows):
    """從 per-image 列計算三種統計。"""
    # 只算有 iou 的列（有 GT 的圖）
    valid = [r for r in rows if r.get('iou') is not None]

    methods = sorted({r['method'] for r in valid})
    cameras = sorted({r['camera_id'] for r in valid}, key=lambda x: (len(x), x))

    # 1) 每方法整體平均
    by_method = {}
    for m in methods:
        sub = [r for r in valid if r['method'] == m]
        by_method[m] = {
            'n': len(sub),
            'mean_iou': sum(r['iou'] for r in sub) / len(sub),
            'mean_fp_rate': sum(r['fp_rate'] for r in sub if r['fp_rate'] is not None) / len(sub),
            'mean_fn_rate': sum(r['fn_rate'] for r in sub if r['fn_rate'] is not None) / len(sub),
        }

    # 2) 每方法 × 每 camera
    by_method_camera = defaultdict(dict)
    for m in methods:
        for c in cameras:
            sub = [r for r in valid if r['method'] == m and r['camera_id'] == c]
            if not sub:
                continue
            by_method_camera[m][c] = {
                'n': len(sub),
                'mean_iou': sum(r['iou'] for r in sub) / len(sub),
                'mean_fp_rate': sum(r['fp_rate'] for r in sub if r['fp_rate'] is not None) / len(sub),
                'mean_fn_rate': sum(r['fn_rate'] for r in sub if r['fn_rate'] is not None) / len(sub),
            }

    # 3) 每 camera 跨方法平均（該 camera 的預測難度）
    by_camera = {}
    for c in cameras:
        sub = [r for r in valid if r['camera_id'] == c]
        if not sub:
            continue
        by_camera[c] = {
            'n_images': len(sub) // len(methods) if methods else 0,  # 約略每方法張數
            'n_rows': len(sub),
            'mean_iou': sum(r['iou'] for r in sub) / len(sub),
            'mean_fp_rate': sum(r['fp_rate'] for r in sub if r['fp_rate'] is not None) / len(sub),
            'mean_fn_rate': sum(r['fn_rate'] for r in sub if r['fn_rate'] is not None) / len(sub),
        }

    return {
        'by_method': by_method,
        'by_method_camera': dict(by_method_camera),
        'by_camera': by_camera,
        'methods': methods,
        'cameras': cameras,
    }


def report_txt(data, out_path):
    """輸出純文字報告。"""
    lines = []
    lines.append('=' * 60)
    lines.append('五方法評估分析：IoU / FP rate / FN rate')
    lines.append('=' * 60)

    lines.append('\n【1】每方法整體平均（僅有 GT 的圖）')
    lines.append('-' * 60)
    for m in data['methods']:
        d = data['by_method'][m]
        lines.append(f"  {m}")
        lines.append(f"    n={d['n']}  mean_iou={d['mean_iou']:.4f}  mean_fp_rate={d['mean_fp_rate']:.4f}  mean_fn_rate={d['mean_fn_rate']:.4f}")

    lines.append('\n【2】每方法 × 每 camera（各方法在不同 camera 上的表現）')
    lines.append('-' * 60)
    for m in data['methods']:
        lines.append(f"\n  {m}:")
        for c in data['cameras']:
            if c not in data['by_method_camera'].get(m, {}):
                continue
            d = data['by_method_camera'][m][c]
            lines.append(f"    camera {c}: n={d['n']}  iou={d['mean_iou']:.4f}  fp={d['mean_fp_rate']:.4f}  fn={d['mean_fn_rate']:.4f}")

    lines.append('\n【3】每 camera 跨方法平均（該 camera 預測難度：IoU 低 / FP 或 FN 高 = 較難）')
    lines.append('-' * 60)
    # 依 mean_iou 升序排，難的 camera 在前
    cam_list = [(c, data['by_camera'][c]) for c in data['cameras'] if c in data['by_camera']]
    cam_list.sort(key=lambda x: x[1]['mean_iou'])
    for c, d in cam_list:
        lines.append(f"  camera {c}: n_rows={d['n_rows']}  mean_iou={d['mean_iou']:.4f}  mean_fp_rate={d['mean_fp_rate']:.4f}  mean_fn_rate={d['mean_fn_rate']:.4f}")

    lines.append('\n' + '=' * 60)
    text = '\n'.join(lines)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(text)
    print(text)
    return text


def report_json(data, out_path):
    """輸出 JSON 供後續使用。"""
    # 轉成可 JSON 序列化（defaultdict -> dict）
    out = {
        'by_method': data['by_method'],
        'by_method_camera': data['by_method_camera'],
        'by_camera': data['by_camera'],
        'methods': data['methods'],
        'cameras': data['cameras'],
    }
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f'JSON 已寫入: {out_path}')


def main():
    parser = argparse.ArgumentParser(description='分析五方法 IoU/FP/FN')
    parser.add_argument('--csv', type=str, default=os.path.join(_root, 'outputs', 'eval_five_methods', 'per_image_metrics.csv'))
    parser.add_argument('--out_dir', type=str, default=os.path.join(_root, 'outputs', 'eval_five_methods'))
    args = parser.parse_args()

    if not os.path.isfile(args.csv):
        print(f'找不到 CSV: {args.csv}')
        sys.exit(1)
    os.makedirs(args.out_dir, exist_ok=True)

    rows = load_csv(args.csv)
    data = analyze(rows)

    txt_path = os.path.join(args.out_dir, 'five_methods_analysis_report.txt')
    json_path = os.path.join(args.out_dir, 'five_methods_analysis.json')
    report_txt(data, txt_path)
    report_json(data, json_path)


if __name__ == '__main__':
    main()
