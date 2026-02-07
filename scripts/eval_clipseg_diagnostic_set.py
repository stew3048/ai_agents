"""
對全 diagnostic set 執行 CLIPSeg（VLM）inference，輸出與 DL/SAM 同格式的指標 CSV。

讀取 diagnostic_with_failure_modes.csv（與 eval_sam2_diagnostic_set 相同），
每張圖用文字 prompt "sky" 跑 CLIPSeg，計算 IoU / FP rate / FN rate，
輸出 diagnostic_clipseg_metrics.csv（含 weather、derived_subset_name、dl_*、clipseg_*），
供三方法（DL / SAM / CLIPSeg）比較。

依賴：pip install transformers timm
用法:
  python scripts/eval_clipseg_diagnostic_set.py
  python scripts/eval_clipseg_diagnostic_set.py --limit 10
"""

import os
import sys
import argparse
import csv
import numpy as np
from PIL import Image
from tqdm import tqdm

_script_dir = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_script_dir)
sys.path.insert(0, _root)
sys.path.insert(0, _script_dir)
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# 從現有模組複用 GT 載入與指標計算
from eval_sam2_failure_cases import (
    load_gt_mask,
    calculate_metrics_per_image,
    create_overlay_for_sam2,
    parse_float,
)
from eval_sam2_diagnostic_set import (
    load_diagnostic_rows,
    derive_subset_name,
    resolve_image_path,
)

WEATHER_COL = 'weather(clear|cloudy|rain|fog|snow|unknown)'

try:
    import torch
    from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
    CLIPSEG_AVAILABLE = True
except ImportError as e:
    CLIPSEG_AVAILABLE = False
    print(f"[WARNING] CLIPSeg 未安裝: {e}")
    print("請執行: pip install transformers timm")


def load_clipseg_model(model_id='CIDAS/clipseg-rd64-refined', device='cpu'):
    """載入 CLIPSeg 模型與 processor。"""
    processor = CLIPSegProcessor.from_pretrained(model_id)
    model = CLIPSegForImageSegmentation.from_pretrained(model_id)
    model.to(device)
    model.eval()
    return processor, model


def predict_clipseg(processor, model, image_path, text_prompts=None, device='cpu', threshold=0.5):
    """
    單張圖用 CLIPSeg 預測 sky mask。
    image_path: 圖片路徑
    text_prompts: 文字列表，例如 ["sky"] 或 ["sky", "blue sky"]（多句時取平均 logits）
    回傳: pred_mask (H, W) float32 0/1，與原圖同尺寸
    """
    if text_prompts is None:
        text_prompts = ['sky']
    image = Image.open(image_path).convert('RGB')
    image_np = np.array(image)
    orig_h, orig_w = image_np.shape[:2]

    inputs = processor(
        text=text_prompts,
        images=image,
        return_tensors='pt',
        padding=True,
    )
    inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    # logits: (batch, num_prompts, H, W) 或 (batch, H, W)；取第一個 batch、多 prompt 時取平均
    logits = outputs.logits
    if logits.dim() == 4:
        logits = logits.mean(dim=1)
    logits = logits.cpu().float().numpy()
    logits = np.squeeze(logits)
    if logits.ndim != 2:
        logits = logits[0]

    # sigmoid（CLIPSeg 輸出為 logits）
    pred = 1.0 / (1.0 + np.exp(-logits))
    pred = (pred > threshold).astype(np.float32)

    # resize 回原圖尺寸以與 GT 對齊
    if pred.shape[0] != orig_h or pred.shape[1] != orig_w:
        pil = Image.fromarray((pred * 255).astype(np.uint8))
        pil = pil.resize((orig_w, orig_h), Image.NEAREST)
        pred = np.array(pil, dtype=np.float32) / 255.0

    return pred


def process_diagnostic_set(
    input_csv,
    output_csv,
    overlay_dir,
    model_id='CIDAS/clipseg-rd64-refined',
    device='cpu',
    limit=None,
    threshold=0.5,
):
    rows, orig_fieldnames = load_diagnostic_rows(input_csv)
    if limit is not None:
        rows = rows[:limit]
        print(f"  [LIMIT] 僅處理前 {limit} 筆")
    print(f"  共 {len(rows)} 筆待處理")

    if not CLIPSEG_AVAILABLE:
        print("[WARNING] CLIPSeg 未安裝，寫入 stub 結果（clipseg_* 為空）")
        results = []
        for row in rows:
            r = dict(row)
            r['weather'] = row.get(WEATHER_COL, '') or 'unknown'
            r['derived_subset_name'] = derive_subset_name(row)
            r['clipseg_iou'] = r['clipseg_fp_rate'] = r['clipseg_fn_rate'] = ''
            r['clipseg_pred_positive_ratio'] = r['clipseg_overlay_path'] = ''
            results.append(r)
        out_fields = [c for c in orig_fieldnames if c != WEATHER_COL] + [
            'weather', 'derived_subset_name',
            'clipseg_iou', 'clipseg_fp_rate', 'clipseg_fn_rate',
            'clipseg_pred_positive_ratio', 'clipseg_overlay_path',
        ]
        os.makedirs(os.path.dirname(output_csv) or '.', exist_ok=True)
        with open(output_csv, 'w', encoding='utf-8-sig', newline='') as f:
            w = csv.DictWriter(f, fieldnames=out_fields, extrasaction='ignore')
            w.writeheader()
            w.writerows(results)
        print(f"寫入 stub: {output_csv}")
        return

    processor, model = load_clipseg_model(model_id=model_id, device=device)
    os.makedirs(overlay_dir, exist_ok=True)
    results = []

    for row in tqdm(rows, desc='CLIPSeg diagnostic'):
        camera_id = row.get('camera_id', '')
        image_id = row.get('image_id', '')
        image_path = resolve_image_path(row)
        if not image_path:
            continue
        has_sky = row.get('has_sky', '').upper() == 'TRUE'
        subset_name = derive_subset_name(row)

        try:
            pred_mask = predict_clipseg(
                processor, model, image_path,
                text_prompts=['sky'],
                device=device,
                threshold=threshold,
            )
        except Exception as e:
            tqdm.write(f"[ERROR] {camera_id}/{image_id}: {e}")
            continue

        gt_mask, gt_exists = load_gt_mask(image_path, camera_id, image_id, has_sky)
        metrics = calculate_metrics_per_image(
            pred_mask, gt_mask,
            has_gt=gt_exists and has_sky,
        )

        overlay_filename = f'camera_{camera_id}_image_{image_id}_clipseg_overlay.png'
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
        out['clipseg_iou'] = metrics.get('sam2_iou')
        out['clipseg_fp_rate'] = metrics.get('sam2_fp_rate')
        out['clipseg_fn_rate'] = metrics.get('sam2_fn_rate')
        out['clipseg_pred_positive_ratio'] = metrics.get('sam2_pred_positive_ratio')
        out['clipseg_overlay_path'] = overlay_path
        results.append(out)

    if not results:
        print("  無有效結果")
        return
    out_fields = [c for c in (orig_fieldnames or list(results[0].keys())) if c != WEATHER_COL] + [
        'weather', 'derived_subset_name',
        'clipseg_iou', 'clipseg_fp_rate', 'clipseg_fn_rate',
        'clipseg_pred_positive_ratio', 'clipseg_overlay_path',
    ]
    os.makedirs(os.path.dirname(output_csv) or '.', exist_ok=True)
    with open(output_csv, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=out_fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(results)
    print(f"寫入: {output_csv}（{len(results)} 筆）")


def main():
    parser = argparse.ArgumentParser(description='對全 diagnostic set 執行 CLIPSeg（VLM）')
    parser.add_argument('--input', type=str, default='outputs/diagnostic_with_failure_modes.csv')
    parser.add_argument('--output', type=str, default='outputs/diagnostic_clipseg_metrics.csv')
    parser.add_argument('--overlay_dir', type=str, default='outputs/clipseg_diagnostic_overlays')
    parser.add_argument('--model_id', type=str, default='CIDAS/clipseg-rd64-refined')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--threshold', type=float, default=0.5, help='二值化閾值')
    parser.add_argument('--limit', type=int, default=None, help='僅處理前 N 筆（測試用）')
    args = parser.parse_args()

    print("讀取:", args.input)
    process_diagnostic_set(
        args.input,
        args.output,
        args.overlay_dir,
        model_id=args.model_id,
        device=args.device,
        limit=args.limit,
        threshold=args.threshold,
    )


if __name__ == '__main__':
    main()
