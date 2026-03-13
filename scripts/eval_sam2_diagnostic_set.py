"""
對全 diagnostic set 執行 SAM 2.0 inference，並輸出含 weather 的指標 CSV

讀取 diagnostic_with_failure_modes.csv，依每筆的 has_sky / scene / occlusion / light /
sea_sky_confusable 推導 prompt 策略（對齊五大情境），逐張跑 SAM，計算指標，
輸出 diagnostic_sam_metrics.csv（含 weather、dl_*、sam2_*），供後續依天氣彙總與報告使用。

用法:
  python scripts/eval_sam2_diagnostic_set.py
  python scripts/eval_sam2_diagnostic_set.py --limit 10
  python scripts/eval_sam2_diagnostic_set.py --stub
"""

import os
import sys
import argparse
import csv
from tqdm import tqdm

# 專案根目錄與 scripts 皆加入 path，以便載入 eval_sam2_failure_cases
_script_dir = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_script_dir)
sys.path.insert(0, _root)
sys.path.insert(0, _script_dir)
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

from eval_sam2_failure_cases import (
    SAM2_AVAILABLE,
    load_sam2_model,
    get_prompt_strategy,
    predict_with_sam2,
    load_gt_mask,
    calculate_metrics_per_image,
    create_overlay_for_sam2,
)
from eval_sam2_failure_cases import parse_float

WEATHER_COL = 'weather(clear|cloudy|rain|fog|snow|unknown)'


def derive_subset_name(row):
    """
    從 diagnostic 一筆的欄位推導對應五大情境的 subset_name，供 get_prompt_strategy 使用。
    優先順序：no-sky -> sea-sky-confusable -> heavy-occlusion -> night -> urban -> default.
    """
    has_sky = row.get('has_sky', '').upper() == 'TRUE'
    scene = (row.get('scene(sea|urban|forest|other)', '') or '').strip().lower()
    occlusion = (row.get('occlusion(none|partial|heavy)', '') or '').strip().lower()
    light = (row.get('light (day|dusk|night)', '') or '').strip().lower()
    sea_sky = (row.get('sea_sky_confusable', '') or '').strip() == '1'

    if not has_sky:
        return 'no-sky'
    if sea_sky:
        return 'sea-sky-confusable'
    if occlusion == 'heavy':
        return 'heavy-occlusion'
    if light == 'night' and has_sky:
        return 'night'
    if scene == 'urban' and has_sky:
        return 'urban'
    return 'default'


def load_diagnostic_rows(csv_path):
    rows = []
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        for row in reader:
            for col in ['dl_iou', 'dl_fp_rate', 'dl_fn_rate']:
                if col in row:
                    row[col] = parse_float(row[col])
            rows.append(row)
    return rows, fieldnames


def resolve_image_path(row):
    path_val = row.get('path', '')
    if path_val:
        p = os.path.join('data', path_val.replace('\\', os.sep))
        if os.path.exists(p):
            return os.path.abspath(p)
    camera_id = row.get('camera_id', '')
    image_id = row.get('image_id', '')
    for name in [f'{int(image_id):03d}.jpg', f'{image_id}.jpg']:
        p = os.path.join('data', f'skyfinder_{camera_id}', 'images', name)
        if os.path.exists(p):
            return os.path.abspath(p)
    return None


def process_diagnostic_set(
    input_csv,
    output_csv,
    overlay_dir,
    model_type='sam2_hiera_small',
    device='cpu',
    limit=None,
):
    rows, orig_fieldnames = load_diagnostic_rows(input_csv)
    if limit is not None:
        rows = rows[:limit]
        print(f"  [LIMIT] 僅處理前 {limit} 筆")
    print(f"  共 {len(rows)} 筆待處理")

    if not SAM2_AVAILABLE:
        print("[WARNING] SAM 2.0 未安裝，寫入 stub 結果（sam2_* 為空）")
        results = []
        for row in rows:
            r = dict(row)
            r['weather'] = row.get(WEATHER_COL, '') or 'unknown'
            r['derived_subset_name'] = derive_subset_name(row)
            r['sam2_iou'] = ''
            r['sam2_fp_rate'] = ''
            r['sam2_fn_rate'] = ''
            r['sam2_pred_positive_ratio'] = ''
            r['sam2_overlay_path'] = ''
            results.append(r)
        out_fields = [c for c in orig_fieldnames if c != WEATHER_COL] + [
            'weather', 'derived_subset_name',
            'sam2_iou', 'sam2_fp_rate', 'sam2_fn_rate', 'sam2_pred_positive_ratio', 'sam2_overlay_path'
        ]
        os.makedirs(os.path.dirname(output_csv) or '.', exist_ok=True)
        with open(output_csv, 'w', encoding='utf-8-sig', newline='') as f:
            w = csv.DictWriter(f, fieldnames=out_fields, extrasaction='ignore')
            w.writeheader()
            w.writerows(results)
        print(f"寫入 stub: {output_csv}")
        return

    predictor = load_sam2_model(model_type=model_type, device=device)
    os.makedirs(overlay_dir, exist_ok=True)
    results = []

    for row in tqdm(rows, desc='SAM diagnostic'):
        camera_id = row.get('camera_id', '')
        image_id = row.get('image_id', '')
        image_path = resolve_image_path(row)
        if not image_path:
            continue
        has_sky = row.get('has_sky', '').upper() == 'TRUE'
        subset_name = derive_subset_name(row)

        try:
            pred_mask = predict_with_sam2(predictor, image_path, subset_name)
        except Exception as e:
            tqdm.write(f"[ERROR] {camera_id}/{image_id}: {e}")
            continue

        gt_mask, gt_exists = load_gt_mask(image_path, camera_id, image_id, has_sky)
        metrics = calculate_metrics_per_image(pred_mask, gt_mask, has_gt=gt_exists and has_sky)

        overlay_filename = f'camera_{camera_id}_image_{image_id}_sam2_overlay.png'
        overlay_path = os.path.join(overlay_dir, overlay_filename)
        create_overlay_for_sam2(
            image_path,
            gt_mask if gt_exists else None,
            pred_mask,
            overlay_path,
        )

        out = dict(row)
        out['weather'] = row.get(WEATHER_COL, '') or 'unknown'
        out['derived_subset_name'] = subset_name
        out['sam2_iou'] = metrics.get('sam2_iou')
        out['sam2_fp_rate'] = metrics.get('sam2_fp_rate')
        out['sam2_fn_rate'] = metrics.get('sam2_fn_rate')
        out['sam2_pred_positive_ratio'] = metrics.get('sam2_pred_positive_ratio')
        out['sam2_overlay_path'] = overlay_path
        results.append(out)

    if not results:
        print("  無有效結果")
        return
    out_fields = [c for c in (orig_fieldnames or list(results[0].keys())) if c != WEATHER_COL] + [
        'weather', 'derived_subset_name',
        'sam2_iou', 'sam2_fp_rate', 'sam2_fn_rate', 'sam2_pred_positive_ratio', 'sam2_overlay_path'
    ]
    os.makedirs(os.path.dirname(output_csv) or '.', exist_ok=True)
    with open(output_csv, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=out_fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(results)
    print(f"寫入: {output_csv}（{len(results)} 筆）")


def main():
    parser = argparse.ArgumentParser(description='對全 diagnostic set 執行 SAM 2.0')
    parser.add_argument('--input', type=str, default='outputs/diagnostic_with_failure_modes.csv')
    parser.add_argument('--output', type=str, default='outputs/diagnostic_sam_metrics.csv')
    parser.add_argument('--overlay_dir', type=str, default='outputs/sam2_diagnostic_overlays')
    parser.add_argument('--model_type', type=str, default='sam2_hiera_small')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--limit', type=int, default=None, help='僅處理前 N 筆（測試用）')
    parser.add_argument('--stub', action='store_true', help='不載入 SAM，產出空白 sam2_* CSV')
    args = parser.parse_args()

    if args.stub:
        rows, orig_fieldnames = load_diagnostic_rows(args.input)
        results = []
        for row in rows:
            r = dict(row)
            r['weather'] = row.get(WEATHER_COL, '') or 'unknown'
            r['derived_subset_name'] = derive_subset_name(row)
            r['sam2_iou'] = r['sam2_fp_rate'] = r['sam2_fn_rate'] = ''
            r['sam2_pred_positive_ratio'] = r['sam2_overlay_path'] = ''
            results.append(r)
        out_fields = [c for c in orig_fieldnames if c != WEATHER_COL] + [
            'weather', 'derived_subset_name',
            'sam2_iou', 'sam2_fp_rate', 'sam2_fn_rate', 'sam2_pred_positive_ratio', 'sam2_overlay_path'
        ]
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        with open(args.output, 'w', encoding='utf-8-sig', newline='') as f:
            w = csv.DictWriter(f, fieldnames=out_fields, extrasaction='ignore')
            w.writeheader()
            w.writerows(results)
        print(f"Stub 寫入: {args.output}（{len(results)} 筆）")
        return

    print("讀取:", args.input)
    process_diagnostic_set(
        args.input,
        args.output,
        args.overlay_dir,
        model_type=args.model_type,
        device=args.device,
        limit=args.limit,
    )


if __name__ == '__main__':
    main()
