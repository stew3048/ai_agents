"""
五種方法在 in-domain test 上統一評估：per-image IoU/Dice/BCE/FP/FN、
overall best_10/worst_10、每個 camera 的 best_5/worst_5。

與既有 pipeline 一致：
- test 名單：scripts.train_in_domain.load_list_file（同 train_in_domain / U-Net eval）
- 指標：utils.metrics.calculate_metrics（logits）、utils.metrics.compute_metrics_numpy（mask）
- overlay：僅對 overall best_10 / worst_10 輸出

設計：每個方法獨立 try/except，某方法失敗只記錄錯誤並繼續跑其餘方法。
用法:
  python scripts/eval_five_methods_in_domain_test.py
  # 一次只跑一個 model（推薦，較穩）：
  python scripts/eval_five_methods_in_domain_test.py --method grounding
  python scripts/eval_five_methods_in_domain_test.py --method clipseg
  python scripts/eval_five_methods_in_domain_test.py --method clipseg_improved
  python scripts/eval_five_methods_in_domain_test.py --method dino_linear
  python scripts/eval_five_methods_in_domain_test.py --method dino_segmentation

對齊昨日腳本（GT + 指標公式一致）：
- Grounding DINO+SAM：eval_grounding_dino_sam_final（load_gt_mask, calculate_metrics_per_image，fp_rate=FP/TN, fn_rate=FN/TP）
- CLIPSeg / CLIPSeg 改善：eval_sam2_failure_cases.load_gt_mask + utils.metrics.compute_metrics_numpy（fp_rate=FP/(TN+FP), fn_rate=FN/(TP+FN)）
- DINO+Linear：eval_dino_linear_small（load_gt_mask, calculate_metrics；輸入 [0,1]，無 ImageNet 正規化）
- DINO+dino_segmentation：eval_dino_segmentation_small（同上）
"""

import os
import sys
import json
import csv
import argparse
from pathlib import Path
from collections import defaultdict

_script_dir = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_script_dir)
sys.path.insert(0, _root)
sys.path.insert(0, _script_dir)
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')


def _log(msg, file=None, **kwargs):
    """即時輸出到終端（flush），方便看進度。"""
    (file or sys.stdout).write(msg + '\n')
    (file or sys.stdout).flush()


def _patch_dinov2_py39():
    """Python 3.9 不支援 float|None，在載入 DINO 前 patch torch hub 快取裡的 dinov2 layers。"""
    if sys.version_info >= (3, 10):
        return
    try:
        import torch
        hub_dir = torch.hub.get_dir()
        base = os.path.join(hub_dir, 'facebookresearch_dinov2_main', 'dinov2', 'layers')
        for fname in ('attention.py', 'block.py'):
            path = os.path.join(base, fname)
            if not os.path.isfile(path):
                continue
            with open(path, 'r', encoding='utf-8') as f:
                text = f.read()
            if 'float | None' not in text:
                continue
            text = text.replace('float | None', 'Optional[float]')
            if 'from typing import' in text and 'Optional' not in text:
                import re
                text = re.sub(r'(from typing import [^\n]+)', r'\1, Optional', text, count=1)
            elif 'from typing import' not in text:
                text = 'from typing import Optional\n' + text
            with open(path, 'w', encoding='utf-8') as f:
                f.write(text)
            _log(f'  [patch] 已為 Python 3.9 修正: {path}', file=sys.stderr)
    except Exception as e:
        _log(f'  [patch] DINO Python 3.9 相容 patch 跳過: {e}', file=sys.stderr)


import numpy as np
from tqdm import tqdm

# 與 train_in_domain / U-Net eval 同一套：list 來源 + 指標
from scripts.train_in_domain import load_list_file as load_list_file_in_domain
from utils.metrics import calculate_metrics, compute_metrics_numpy


def _smooth():
    return 1e-6


def compute_bce_from_logits(logits, gt_tensor):
    """logits (1,1,H,W), gt (1,1,H,W) 0~1，回傳 per-image BCE 標量。"""
    import torch
    if logits.shape != gt_tensor.shape:
        gt_tensor = torch.nn.functional.interpolate(
            gt_tensor.float(), size=logits.shape[2:], mode='nearest'
        )
    return float(
        torch.nn.functional.binary_cross_entropy_with_logits(logits, gt_tensor).item()
    )


def to_records(bunch):
    """將結果列表轉成 JSON 用的 records（rank, source_id, camera_id, iou, fp_rate, fn_rate, orig_path）。"""
    return [
        {
            'rank': i + 1,
            'source_id': r.get('source_id'),
            'camera_id': r.get('camera_id'),
            'iou': round(float(r['iou']), 6),
            'fp_rate': round(float(r.get('fp_rate', 0)), 6),
            'fn_rate': round(float(r.get('fn_rate', 0)), 6),
            'orig_path': r.get('orig_path'),
        }
        for i, r in enumerate(bunch)
    ]


def build_best_worst_json(results, n_overall=10, n_per_camera=5):
    """
    results: List[Dict] 每筆含 iou, fp_rate, fn_rate, camera_id, orig_path, source_id 等。
    回傳 { summary: { mean_iou, mean_fp_rate, mean_fn_rate, n_images }, overall: { best_10, worst_10 }, per_camera: {...} }。
    """
    # 只考慮有 iou 的（有 GT 的）
    with_iou = [r for r in results if r.get('iou') is not None]
    with_iou.sort(key=lambda x: x['iou'], reverse=True)

    n = len(with_iou)
    if n > 0:
        mean_iou = sum(r['iou'] for r in with_iou) / n
        mean_fp = sum(r.get('fp_rate', 0) for r in with_iou) / n
        mean_fn = sum(r.get('fn_rate', 0) for r in with_iou) / n
        dice_vals = [r['dice'] for r in with_iou if r.get('dice') is not None]
        mean_dice = sum(dice_vals) / len(dice_vals) if dice_vals else None
    else:
        mean_iou = mean_fp = mean_fn = mean_dice = None
    summary = {
        'n_images': n,
        'mean_iou': round(mean_iou, 6) if mean_iou is not None else None,
        'mean_fp_rate': round(mean_fp, 6) if mean_fp is not None else None,
        'mean_fn_rate': round(mean_fn, 6) if mean_fn is not None else None,
        'mean_dice': round(mean_dice, 6) if mean_dice is not None else None,
    }

    overall_best = with_iou[:n_overall]
    overall_worst = with_iou[-n_overall:] if len(with_iou) >= n_overall else with_iou

    by_camera = defaultdict(list)
    for r in with_iou:
        by_camera[r['camera_id']].append(r)

    per_camera = {}
    for cid, lst in by_camera.items():
        lst_sorted = sorted(lst, key=lambda x: x['iou'], reverse=True)
        best = lst_sorted[:n_per_camera]
        worst = lst_sorted[-n_per_camera:] if len(lst_sorted) >= n_per_camera else lst_sorted
        per_camera[cid] = {'best_5': to_records(best), 'worst_5': to_records(worst)}

    return {
        'summary': summary,
        'overall': {'best_10': to_records(overall_best), 'worst_10': to_records(overall_worst)},
        'per_camera': per_camera,
    }


def run_clipseg(test_list_with_path, device, improved=True, overlay_dir=None):
    """跑 CLIPSeg（improved=True 為空間改善版），回傳 List[Dict]。若 overlay_dir 有給，只對 overall best_10/worst_10 寫 overlay。"""
    from scripts.eval_clipseg_final import (
        load_clipseg_model,
        predict_clipseg,
        resolve_image_path,
        create_overlay as create_overlay_clipseg,
    )
    from scripts.eval_sam2_failure_cases import load_gt_mask

    processor, model = load_clipseg_model(device=device)
    results = []
    for idx, item in enumerate(tqdm(test_list_with_path, desc='  CLIPSeg', file=sys.stderr, dynamic_ncols=True)):
        orig_path = item['image_path']
        camera_id = item['camera_id']
        image_filename = item['image']
        image_path = os.path.join(_root, 'data', f'skyfinder_{camera_id}', 'images', image_filename)
        if not os.path.exists(image_path):
            image_path = resolve_image_path(orig_path, base_data_dir=os.path.join(_root, 'data'))
        if not image_path or not os.path.exists(image_path):
            continue
        try:
            pred_mask = predict_clipseg(
                processor, model, image_path,
                device=device,
                use_dynamic_threshold=improved,
                use_spatial_prior=improved,
                use_morphology_open=improved,
                use_multi_prompt_contrast=improved,
            )
        except Exception as e:
            print(f'    CLIPSeg 推論失敗 {image_path}: {e}')
            continue
        try:
            image_id = int(os.path.splitext(image_filename)[0])
        except ValueError:
            image_id = 0
        gt_mask, has_gt = load_gt_mask(image_path, camera_id, image_id, has_sky=True)
        if not has_gt or gt_mask is None:
            results.append({
                'orig_path': orig_path, 'camera_id': camera_id, 'source_id': f'{camera_id}_{os.path.splitext(image_filename)[0]}',
                'iou': None, 'dice': None, 'bce': None, 'fp_rate': 0.0, 'fn_rate': None,
            })
            continue
        iou, dice, fp_rate, fn_rate = compute_metrics_numpy(pred_mask, gt_mask)
        results.append({
            'orig_path': orig_path, 'camera_id': camera_id, 'source_id': f'{camera_id}_{os.path.splitext(image_filename)[0]}',
            'iou': iou, 'dice': dice, 'bce': None, 'fp_rate': fp_rate, 'fn_rate': fn_rate,
            '_full_image_path': image_path, '_camera_id': camera_id, '_image_filename': image_filename, '_image_id': image_id,
        })
    if overlay_dir and results:
        os.makedirs(overlay_dir, exist_ok=True)
        with_iou = [r for r in results if r.get('iou') is not None]
        with_iou.sort(key=lambda x: x['iou'], reverse=True)
        best_10 = with_iou[:10]
        worst_10 = with_iou[-10:] if len(with_iou) >= 10 else []
        for rank, r in enumerate(best_10, start=1):
            path_img = r.get('_full_image_path')
            if not path_img or not os.path.exists(path_img):
                continue
            try:
                pred_mask = predict_clipseg(processor, model, path_img, device=device,
                    use_dynamic_threshold=improved, use_spatial_prior=improved,
                    use_morphology_open=improved, use_multi_prompt_contrast=improved)
                gt_mask, _ = load_gt_mask(path_img, r['_camera_id'], r['_image_id'], has_sky=True)
                if gt_mask is not None:
                    overlay_img = create_overlay_clipseg(path_img, pred_mask, gt_mask)
                    base = os.path.splitext(os.path.basename(path_img))[0]
                    fname = f"best_{rank:02d}_iou{r['iou']:.4f}_{r['camera_id']}_{base}_overlay.png"
                    overlay_img.save(os.path.join(overlay_dir, fname))
            except Exception:
                pass
        for rank, r in enumerate(worst_10, start=1):
            path_img = r.get('_full_image_path')
            if not path_img or not os.path.exists(path_img):
                continue
            try:
                pred_mask = predict_clipseg(processor, model, path_img, device=device,
                    use_dynamic_threshold=improved, use_spatial_prior=improved,
                    use_morphology_open=improved, use_multi_prompt_contrast=improved)
                gt_mask, _ = load_gt_mask(path_img, r['_camera_id'], r['_image_id'], has_sky=True)
                if gt_mask is not None:
                    overlay_img = create_overlay_clipseg(path_img, pred_mask, gt_mask)
                    base = os.path.splitext(os.path.basename(path_img))[0]
                    fname = f"worst_{rank:02d}_iou{r['iou']:.4f}_{r['camera_id']}_{base}_overlay.png"
                    overlay_img.save(os.path.join(overlay_dir, fname))
            except Exception:
                pass
    for r in results:
        r.pop('_full_image_path', None)
        r.pop('_camera_id', None)
        r.pop('_image_filename', None)
        r.pop('_image_id', None)
    return results


def run_grounding_dino_sam(test_list_with_path, device, data_dir, overlay_dir=None):
    """跑 Grounding DINO + SAM，回傳同上。若 overlay_dir 有給，只對 overall best_10 / worst_10 寫出 overlay。"""
    from scripts.eval_grounding_dino_sam_final import (
        load_grounding_dino_model,
        load_sam_model,
        predict_grounding_dino_sam,
        load_gt_mask as load_gt_grounding,
        calculate_metrics_per_image as calc_metrics_grounding,
        create_overlay as create_overlay_gdino,
    )

    grounding_processor, grounding_model = load_grounding_dino_model(device=device)
    sam_predictor = load_sam_model(device=device)
    results = []
    for item in tqdm(test_list_with_path, desc='  Grounding DINO + SAM', file=sys.stderr, dynamic_ncols=True):
        camera_id = item['camera_id']
        image_filename = item['image']
        mask_filename = item['mask']
        full_image_path = os.path.join(data_dir, f'skyfinder_{camera_id}', 'images', image_filename)
        full_mask_path = os.path.join(data_dir, f'skyfinder_{camera_id}', 'masks', mask_filename)
        if not os.path.exists(full_image_path):
            continue
        try:
            pred_mask = predict_grounding_dino_sam(
                grounding_processor, grounding_model, sam_predictor,
                full_image_path, device=device,
            )
        except Exception as e:
            print(f'    Grounding DINO+SAM 推論失敗 {full_image_path}: {e}')
            continue
        gt_mask, has_gt = load_gt_grounding(full_image_path, full_mask_path)
        if not has_gt or gt_mask is None:
            results.append({
                'orig_path': item['image_path'], 'camera_id': camera_id,
                'source_id': f'{camera_id}_{os.path.splitext(image_filename)[0]}',
                'iou': None, 'dice': None, 'bce': None, 'fp_rate': 0.0, 'fn_rate': None,
            })
            continue
        gt_np = np.asarray(gt_mask).squeeze()
        # 與 eval_grounding_dino_sam_final 一致：用原腳本 metrics（fp_rate=FP/TN, fn_rate=FN/TP）
        m = calc_metrics_grounding(pred_mask, gt_np, has_gt=True)
        iou = m['iou']
        fp_rate = m['fp_rate']
        fn_rate = m['fn_rate']
        _, dice, _, _ = compute_metrics_numpy(pred_mask, gt_np)  # dice 用標準公式補上
        results.append({
            'orig_path': item['image_path'], 'camera_id': camera_id,
            'source_id': f'{camera_id}_{os.path.splitext(image_filename)[0]}',
            'iou': iou, 'dice': dice, 'bce': None, 'fp_rate': fp_rate, 'fn_rate': fn_rate,
            '_full_image_path': full_image_path, '_full_mask_path': full_mask_path,
        })
    # 只對 overall best_10 / worst_10 寫 overlay
    if overlay_dir and results:
        os.makedirs(overlay_dir, exist_ok=True)
        with_iou = [r for r in results if r.get('iou') is not None]
        with_iou.sort(key=lambda x: x['iou'], reverse=True)
        best_10 = with_iou[:10]
        worst_10 = with_iou[-10:] if len(with_iou) >= 10 else []
        for rank, r in enumerate(best_10, start=1):
            path_img = r.get('_full_image_path')
            path_mask = r.get('_full_mask_path')
            if not path_img or not os.path.exists(path_img):
                continue
            try:
                pred_mask = predict_grounding_dino_sam(grounding_processor, grounding_model, sam_predictor, path_img, device=device)
                gt_mask, _ = load_gt_grounding(path_img, path_mask or path_img.replace('/images/', '/masks/').replace('\\images\\', '\\masks\\'))
                gt_np = np.asarray(gt_mask).squeeze() if gt_mask is not None else None
                overlay_img = create_overlay_gdino(path_img, pred_mask, gt_np)
                base = os.path.splitext(os.path.basename(path_img))[0]
                fname = f"best_{rank:02d}_iou{r['iou']:.4f}_{r['camera_id']}_{base}_overlay.png"
                overlay_img.save(os.path.join(overlay_dir, fname))
            except Exception:
                pass
        for rank, r in enumerate(worst_10, start=1):
            path_img = r.get('_full_image_path')
            path_mask = r.get('_full_mask_path')
            if not path_img or not os.path.exists(path_img):
                continue
            try:
                pred_mask = predict_grounding_dino_sam(grounding_processor, grounding_model, sam_predictor, path_img, device=device)
                gt_mask, _ = load_gt_grounding(path_img, path_mask or path_img.replace('/images/', '/masks/').replace('\\images\\', '\\masks\\'))
                gt_np = np.asarray(gt_mask).squeeze() if gt_mask is not None else None
                overlay_img = create_overlay_gdino(path_img, pred_mask, gt_np)
                base = os.path.splitext(os.path.basename(path_img))[0]
                fname = f"worst_{rank:02d}_iou{r['iou']:.4f}_{r['camera_id']}_{base}_overlay.png"
                overlay_img.save(os.path.join(overlay_dir, fname))
            except Exception:
                pass
    for r in results:
        r.pop('_full_image_path', None)
        r.pop('_full_mask_path', None)
    return results


def run_dino_linear(test_list_with_path, device, checkpoint_path, image_size, data_dir, overlay_dir=None):
    """跑 DINO + Linear，回傳含 logits 的 bce。"""
    import torch
    from models import create_sky_seg_dinov2_linear
    from utils.metrics import calculate_metrics
    from scripts.eval_dino_segmentation_small import load_gt_mask as load_gt_resize

    model = create_sky_seg_dinov2_linear().to(device)
    ck = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ck.get('model_state_dict', ck), strict=False)
    model.eval()

    results = []
    for item in tqdm(test_list_with_path, desc='  DINO + Linear', file=sys.stderr, dynamic_ncols=True):
        cid = item['camera_id']
        img_path = os.path.join(data_dir, f'skyfinder_{cid}', 'images', item['image'])
        mask_path = os.path.join(data_dir, f'skyfinder_{cid}', 'masks', item['mask'])
        if not os.path.exists(img_path) or not os.path.exists(mask_path):
            continue
        from PIL import Image
        img_pil = Image.open(img_path).convert('RGB')
        img_np = np.array(img_pil).astype(np.float32) / 255.0
        img_np = img_np.transpose(2, 0, 1)
        img_t = torch.from_numpy(img_np).unsqueeze(0).to(device)
        if img_t.shape[2:] != image_size:
            img_t = torch.nn.functional.interpolate(img_t, size=image_size, mode='bilinear', align_corners=False)
        # 與 train_dino_linear_in_domain / utils.dataset 一致：輸入 [0,1]，不做 ImageNet 正規化
        with torch.no_grad():
            logits = model(img_t)
        pred_h, pred_w = logits.shape[2], logits.shape[3]
        gt = load_gt_resize(mask_path, pred_h, pred_w)
        if gt is None:
            continue
        gt = gt.to(device)
        metrics = calculate_metrics(logits, gt)
        iou = float(metrics['iou'])
        dice = float(metrics['dice'])
        bce = compute_bce_from_logits(logits, gt)
        pred_b = (torch.sigmoid(logits) > 0.5).float()
        gt_b = (gt > 0.5).float()
        fp = ((pred_b == 1) & (gt_b == 0)).sum().item()
        fn = ((pred_b == 0) & (gt_b == 1)).sum().item()
        n_neg = (gt_b == 0).sum().item()
        n_pos = (gt_b == 1).sum().item()
        s = _smooth()
        fp_rate = fp / (n_neg + s)
        fn_rate = fn / (n_pos + s) if n_pos > 0 else 0.0
        results.append({
            'orig_path': item['image_path'], 'camera_id': cid,
            'source_id': f'{cid}_{os.path.splitext(item["image"])[0]}',
            'iou': iou, 'dice': dice, 'bce': bce, 'fp_rate': fp_rate, 'fn_rate': fn_rate,
            '_img_path': img_path, '_mask_path': mask_path,
        })
    if overlay_dir and results:
        from PIL import Image as PILImage
        from scripts.eval_grounding_dino_sam_final import create_overlay as create_overlay_dino
        os.makedirs(overlay_dir, exist_ok=True)
        with_iou = [r for r in results if r.get('iou') is not None]
        with_iou.sort(key=lambda x: x['iou'], reverse=True)
        best_10 = with_iou[:10]
        worst_10 = with_iou[-10:] if len(with_iou) >= 10 else []
        def _write_overlays(entries, prefix):
            for rank, r in enumerate(entries, start=1):
                img_path = r.get('_img_path')
                mask_path = r.get('_mask_path')
                if not img_path or not os.path.exists(img_path) or not mask_path or not os.path.exists(mask_path):
                    continue
                try:
                    img_pil = PILImage.open(img_path).convert('RGB')
                    img_np = np.array(img_pil).astype(np.float32) / 255.0
                    img_np = img_np.transpose(2, 0, 1)
                    img_t = torch.from_numpy(img_np).unsqueeze(0).to(device)
                    if img_t.shape[2:] != image_size:
                        img_t = torch.nn.functional.interpolate(img_t, size=image_size, mode='bilinear', align_corners=False)
                    with torch.no_grad():
                        logits = model(img_t)
                    pred_np = torch.sigmoid(logits).squeeze().cpu().numpy()
                    if pred_np.shape != (img_pil.height, img_pil.width):
                        pred_pil = PILImage.fromarray((pred_np * 255).astype(np.uint8))
                        pred_pil = pred_pil.resize((img_pil.width, img_pil.height), PILImage.NEAREST)
                        pred_np = np.array(pred_pil, dtype=np.float32) / 255.0
                    gt_pil = PILImage.open(mask_path).convert('L')
                    gt_np = np.array(gt_pil, dtype=np.float32)
                    if gt_np.max() > 1.0:
                        gt_np = gt_np / 255.0
                    if gt_np.shape != (img_pil.height, img_pil.width):
                        gt_pil = gt_pil.resize((img_pil.width, img_pil.height), PILImage.NEAREST)
                        gt_np = np.array(gt_pil, dtype=np.float32) / 255.0
                    overlay_img = create_overlay_dino(img_path, pred_np, gt_np)
                    base = os.path.splitext(os.path.basename(img_path))[0]
                    fname = f"{prefix}_{rank:02d}_iou{r['iou']:.4f}_{r['camera_id']}_{base}_overlay.png"
                    overlay_img.save(os.path.join(overlay_dir, fname))
                except Exception:
                    pass
        _write_overlays(best_10, 'best')
        _write_overlays(worst_10, 'worst')
    for r in results:
        r.pop('_img_path', None)
        r.pop('_mask_path', None)
    return results


def run_dino_segmentation(test_list_with_path, device, checkpoint_path, image_size, data_dir, overlay_dir=None):
    """跑 DINO + dino_segmentation，回傳含 bce。"""
    import torch
    from models import create_dinov2_segmentation_model
    from utils.metrics import calculate_metrics
    from scripts.eval_dino_segmentation_small import load_gt_mask as load_gt_resize

    model = create_dinov2_segmentation_model().to(device)
    ck = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ck.get('model_state_dict', ck), strict=False)
    model.eval()

    results = []
    for item in tqdm(test_list_with_path, desc='  DINO + dino_segmentation', file=sys.stderr, dynamic_ncols=True):
        cid = item['camera_id']
        img_path = os.path.join(data_dir, f'skyfinder_{cid}', 'images', item['image'])
        mask_path = os.path.join(data_dir, f'skyfinder_{cid}', 'masks', item['mask'])
        if not os.path.exists(img_path) or not os.path.exists(mask_path):
            continue
        from PIL import Image
        img_pil = Image.open(img_path).convert('RGB')
        img_np = np.array(img_pil).astype(np.float32) / 255.0
        img_np = img_np.transpose(2, 0, 1)
        img_t = torch.from_numpy(img_np).unsqueeze(0).to(device)
        if img_t.shape[2:] != image_size:
            img_t = torch.nn.functional.interpolate(img_t, size=image_size, mode='bilinear', align_corners=False)
        # 與 train_dino_segmentation_in_domain / eval_dino_segmentation_small 一致：輸入 [0,1]，不做 ImageNet 正規化
        with torch.no_grad():
            logits = model(img_t)
        pred_h, pred_w = logits.shape[2], logits.shape[3]
        gt = load_gt_resize(mask_path, pred_h, pred_w)
        if gt is None:
            continue
        gt = gt.to(device)
        metrics = calculate_metrics(logits, gt)
        iou = float(metrics['iou'])
        dice = float(metrics['dice'])
        bce = compute_bce_from_logits(logits, gt)
        pred_b = (torch.sigmoid(logits) > 0.5).float()
        gt_b = (gt > 0.5).float()
        fp = ((pred_b == 1) & (gt_b == 0)).sum().item()
        fn = ((pred_b == 0) & (gt_b == 1)).sum().item()
        n_neg = (gt_b == 0).sum().item()
        n_pos = (gt_b == 1).sum().item()
        s = _smooth()
        fp_rate = fp / (n_neg + s)
        fn_rate = fn / (n_pos + s) if n_pos > 0 else 0.0
        results.append({
            'orig_path': item['image_path'], 'camera_id': cid,
            'source_id': f'{cid}_{os.path.splitext(item["image"])[0]}',
            'iou': iou, 'dice': dice, 'bce': bce, 'fp_rate': fp_rate, 'fn_rate': fn_rate,
            '_img_path': img_path, '_mask_path': mask_path,
        })
    if overlay_dir and results:
        from PIL import Image as PILImage
        from scripts.eval_grounding_dino_sam_final import create_overlay as create_overlay_dino
        os.makedirs(overlay_dir, exist_ok=True)
        with_iou = [r for r in results if r.get('iou') is not None]
        with_iou.sort(key=lambda x: x['iou'], reverse=True)
        best_10 = with_iou[:10]
        worst_10 = with_iou[-10:] if len(with_iou) >= 10 else []
        def _write_overlays(entries, prefix):
            for rank, r in enumerate(entries, start=1):
                img_path = r.get('_img_path')
                mask_path = r.get('_mask_path')
                if not img_path or not os.path.exists(img_path) or not mask_path or not os.path.exists(mask_path):
                    continue
                try:
                    img_pil = PILImage.open(img_path).convert('RGB')
                    img_np = np.array(img_pil).astype(np.float32) / 255.0
                    img_np = img_np.transpose(2, 0, 1)
                    img_t = torch.from_numpy(img_np).unsqueeze(0).to(device)
                    if img_t.shape[2:] != image_size:
                        img_t = torch.nn.functional.interpolate(img_t, size=image_size, mode='bilinear', align_corners=False)
                    with torch.no_grad():
                        logits = model(img_t)
                    pred_np = torch.sigmoid(logits).squeeze().cpu().numpy()
                    if pred_np.shape != (img_pil.height, img_pil.width):
                        pred_pil = PILImage.fromarray((pred_np * 255).astype(np.uint8))
                        pred_pil = pred_pil.resize((img_pil.width, img_pil.height), PILImage.NEAREST)
                        pred_np = np.array(pred_pil, dtype=np.float32) / 255.0
                    gt_pil = PILImage.open(mask_path).convert('L')
                    gt_np = np.array(gt_pil, dtype=np.float32)
                    if gt_np.max() > 1.0:
                        gt_np = gt_np / 255.0
                    if gt_np.shape != (img_pil.height, img_pil.width):
                        gt_pil = gt_pil.resize((img_pil.width, img_pil.height), PILImage.NEAREST)
                        gt_np = np.array(gt_pil, dtype=np.float32) / 255.0
                    overlay_img = create_overlay_dino(img_path, pred_np, gt_np)
                    base = os.path.splitext(os.path.basename(img_path))[0]
                    fname = f"{prefix}_{rank:02d}_iou{r['iou']:.4f}_{r['camera_id']}_{base}_overlay.png"
                    overlay_img.save(os.path.join(overlay_dir, fname))
                except Exception:
                    pass
        _write_overlays(best_10, 'best')
        _write_overlays(worst_10, 'worst')
    for r in results:
        r.pop('_img_path', None)
        r.pop('_mask_path', None)
    return results


def main():
    parser = argparse.ArgumentParser(description='五方法 in-domain test 統一評估')
    parser.add_argument('--test_list', type=str, default=os.path.join(_root, 'outputs', 'test_list.txt'))
    parser.add_argument('--out_dir', type=str, default=os.path.join(_root, 'outputs', 'eval_five_methods'))
    parser.add_argument('--data_dir', type=str, default=os.path.join(_root, 'data'))
    parser.add_argument('--device', type=str, default='cuda' if __import__('torch').cuda.is_available() else 'cpu')
    parser.add_argument('--dino_linear_checkpoint', type=str, default=None,
                        help='DINO+Linear best.pth；預設找 outputs/train_dino_linear_*/checkpoints/best.pth')
    parser.add_argument('--dino_segmentation_checkpoint', type=str, default=None,
                        help='DINO+Seg best.pth；預設找 outputs/train_dino_segmentation_*/checkpoints/best.pth')
    parser.add_argument('--skip_grounding', action='store_true', help='跳過 Grounding DINO+SAM（耗時）')
    parser.add_argument('--skip_clipseg', action='store_true', help='跳過 CLIPSeg')
    parser.add_argument('--skip_dino', action='store_true', help='跳過兩支 DINO')
    parser.add_argument('--method', type=str, default=None,
                        choices=['grounding', 'clipseg', 'clipseg_improved', 'dino_linear', 'dino_segmentation'],
                        help='只跑這一個方法（推薦一個一個跑，避免整支掛掉）')
    parser.add_argument('--overlay_dir', type=str, default=None,
                        help='可選：為每張 test 圖寫出 overlay（目錄按 method 分子目錄）')
    args = parser.parse_args()

    test_list_file = args.test_list
    out_dir = args.out_dir
    data_dir = args.data_dir
    device = args.device
    image_size = (160, 160)

    if not os.path.exists(test_list_file):
        print(f'[錯誤] 找不到 test list: {test_list_file}')
        sys.exit(1)
    test_list = load_list_file_in_domain(test_list_file, base_data_dir=data_dir)
    test_list_with_path = []
    for item in test_list:
        item = dict(item)
        item['image_path'] = f"skyfinder_{item['camera_id']}/images/{item['image']}"
        test_list_with_path.append(item)
    n_images = len(test_list_with_path)
    _log(f'載入 test list: {n_images} 張')
    os.makedirs(out_dir, exist_ok=True)
    overlay_base = args.overlay_dir

    # 決定要跑哪些方法（--method 只跑一個）
    only_one = args.method
    if only_one:
        _log(f'只跑一個方法: {only_one}（共 {n_images} 張圖）')
    else:
        _log(f'開始評估，共 {n_images} 張圖、最多 5 個方法；進度會即時顯示於下方。')

    tasks = []  # [(method_name, run_fn, kwargs), ...]
    if only_one == 'grounding':
        od = os.path.join(overlay_base, 'GroundingDINO-SAM') if overlay_base else None
        tasks.append(('GroundingDINO-SAM', run_grounding_dino_sam, {
            'test_list_with_path': test_list_with_path, 'device': device, 'data_dir': data_dir, 'overlay_dir': od}))
    elif only_one == 'clipseg':
        od = os.path.join(overlay_base, 'CLIPSeg') if overlay_base else None
        tasks.append(('CLIPSeg', run_clipseg, {
            'test_list_with_path': test_list_with_path, 'device': device, 'improved': False, 'overlay_dir': od}))
    elif only_one == 'clipseg_improved':
        od = os.path.join(overlay_base, 'CLIPSeg_spatial_improved') if overlay_base else None
        tasks.append(('CLIPSeg_spatial_improved', run_clipseg, {
            'test_list_with_path': test_list_with_path, 'device': device, 'improved': True, 'overlay_dir': od}))
    elif only_one == 'dino_linear':
        _patch_dinov2_py39()
        ck_linear = args.dino_linear_checkpoint or None
        if not ck_linear:
            candidates = list(Path(os.path.join(_root, 'outputs')).glob('train_dino_linear_*/checkpoints/best.pth'))
            if candidates:
                ck_linear = str(max(candidates, key=os.path.getmtime))
        if ck_linear and os.path.isfile(ck_linear):
            od = os.path.join(overlay_base, 'DINO-Linear') if overlay_base else None
            tasks.append(('DINO-Linear', run_dino_linear, {
                'test_list_with_path': test_list_with_path, 'device': device, 'checkpoint_path': ck_linear,
                'image_size': image_size, 'data_dir': data_dir, 'overlay_dir': od}))
        else:
            _log('[錯誤] 未找到 DINO+Linear checkpoint，請指定 --dino_linear_checkpoint')
            sys.exit(1)
    elif only_one == 'dino_segmentation':
        _patch_dinov2_py39()
        ck_seg = args.dino_segmentation_checkpoint or None
        if not ck_seg:
            candidates = list(Path(os.path.join(_root, 'outputs')).glob('train_dino_segmentation_*/checkpoints/best.pth'))
            if candidates:
                ck_seg = str(max(candidates, key=os.path.getmtime))
        if ck_seg and os.path.isfile(ck_seg):
            od = os.path.join(overlay_base, 'DINO-dino_segmentation') if overlay_base else None
            tasks.append(('DINO-dino_segmentation', run_dino_segmentation, {
                'test_list_with_path': test_list_with_path, 'device': device, 'checkpoint_path': ck_seg,
                'image_size': image_size, 'data_dir': data_dir, 'overlay_dir': od}))
        else:
            _log('[錯誤] 未找到 DINO+dino_segmentation checkpoint，請指定 --dino_segmentation_checkpoint')
            sys.exit(1)
    else:
        # 跑多個：依 skip_* 與 checkpoint 組 tasks
        if not args.skip_grounding:
            od = os.path.join(overlay_base, 'GroundingDINO-SAM') if overlay_base else None
            tasks.append(('GroundingDINO-SAM', run_grounding_dino_sam, {
                'test_list_with_path': test_list_with_path, 'device': device, 'data_dir': data_dir, 'overlay_dir': od}))
        if not args.skip_clipseg:
            od = os.path.join(overlay_base, 'CLIPSeg') if overlay_base else None
            tasks.append(('CLIPSeg', run_clipseg, {
                'test_list_with_path': test_list_with_path, 'device': device, 'improved': False, 'overlay_dir': od}))
            od = os.path.join(overlay_base, 'CLIPSeg_spatial_improved') if overlay_base else None
            tasks.append(('CLIPSeg_spatial_improved', run_clipseg, {
                'test_list_with_path': test_list_with_path, 'device': device, 'improved': True, 'overlay_dir': od}))
        if not args.skip_dino:
            _patch_dinov2_py39()
            ck_linear = args.dino_linear_checkpoint
            if not ck_linear:
                candidates = list(Path(os.path.join(_root, 'outputs')).glob('train_dino_linear_*/checkpoints/best.pth'))
                if candidates:
                    ck_linear = str(max(candidates, key=os.path.getmtime))
            if ck_linear and os.path.isfile(ck_linear):
                od = os.path.join(overlay_base, 'DINO-Linear') if overlay_base else None
                tasks.append(('DINO-Linear', run_dino_linear, {
                    'test_list_with_path': test_list_with_path, 'device': device, 'checkpoint_path': ck_linear,
                    'image_size': image_size, 'data_dir': data_dir, 'overlay_dir': od}))
            ck_seg = args.dino_segmentation_checkpoint
            if not ck_seg:
                candidates = list(Path(os.path.join(_root, 'outputs')).glob('train_dino_segmentation_*/checkpoints/best.pth'))
                if candidates:
                    ck_seg = str(max(candidates, key=os.path.getmtime))
            if ck_seg and os.path.isfile(ck_seg):
                od = os.path.join(overlay_base, 'DINO-dino_segmentation') if overlay_base else None
                tasks.append(('DINO-dino_segmentation', run_dino_segmentation, {
                    'test_list_with_path': test_list_with_path, 'device': device, 'checkpoint_path': ck_seg,
                    'image_size': image_size, 'data_dir': data_dir, 'overlay_dir': od}))

    all_per_image = []
    errors_log = []
    n_tasks = len(tasks)

    def run_one_method(method_name, run_fn, kwargs, idx):
        _log(f'  [{idx}/{n_tasks}] 開始: {method_name} ...', file=sys.stderr)
        sys.stderr.flush()
        try:
            res = run_fn(**kwargs)
            n_done = len(res)
            for r in res:
                all_per_image.append({
                    'method': method_name, 'orig_path': r['orig_path'], 'camera_id': r['camera_id'],
                    'iou': r.get('iou'), 'dice': r.get('dice'), 'bce': r.get('bce'),
                    'fp_rate': r.get('fp_rate'), 'fn_rate': r.get('fn_rate'),
                })
            json_path = os.path.join(out_dir, f'{method_name}_test_best_worst_metrics.json')
            _log(f'  寫入 JSON: {json_path}')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(build_best_worst_json(res), f, ensure_ascii=False, indent=2)
            _log(f'  [OK] {method_name} 完成（{n_done} 張），已寫入 {os.path.basename(json_path)}')
            return True
        except Exception as e:
            err_msg = f'{type(e).__name__}: {e}'
            errors_log.append((method_name, err_msg))
            _log(f'  [FAIL] {method_name}: {err_msg}', file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            sys.stderr.flush()
            return False

    for i, (method_name, run_fn, kwargs) in enumerate(tasks, start=1):
        _log(f'Running {method_name}...')
        run_one_method(method_name, run_fn, kwargs, i)

    # 錯誤日誌寫入檔案
    if errors_log:
        err_path = os.path.join(out_dir, 'eval_five_methods_errors.txt')
        _log(f'寫入錯誤日誌: {err_path}', file=sys.stderr)
        with open(err_path, 'w', encoding='utf-8') as f:
            f.write('以下方法執行失敗，其餘已寫入。\n\n')
            for name, msg in errors_log:
                f.write(f'{name}: {msg}\n\n')
        _log(f'[警告] {len(errors_log)} 個方法失敗，錯誤已寫入 {err_path}', file=sys.stderr)

    # Per-image CSV（數字保留 6 位小數）
    csv_path = os.path.join(out_dir, 'per_image_metrics.csv')
    fieldnames = ['method', 'orig_path', 'camera_id', 'iou', 'dice', 'bce', 'fp_rate', 'fn_rate']

    def _row_to_csv_row(row):
        out = {}
        for k in fieldnames:
            v = row.get(k)
            if v is None or v == '':
                out[k] = ''
            elif isinstance(v, float):
                out[k] = f'{v:.6f}'
            else:
                out[k] = str(v)
        return out

    if only_one and os.path.isfile(csv_path):
        # 單一方法：合併進既有 CSV（保留其他方法、覆寫此方法）
        method_display_name = {'grounding': 'GroundingDINO-SAM', 'clipseg': 'CLIPSeg', 'clipseg_improved': 'CLIPSeg_spatial_improved', 'dino_linear': 'DINO-Linear', 'dino_segmentation': 'DINO-dino_segmentation'}.get(only_one, only_one)
        try:
            with open(csv_path, 'r', newline='', encoding='utf-8') as f:
                existing = [row for row in csv.DictReader(f) if row.get('method') != method_display_name]
        except Exception:
            existing = []
        rows_to_write = existing + [_row_to_csv_row(r) for r in all_per_image]
        _log(f'合併既有 CSV，寫入 per-image CSV: {csv_path}')
    else:
        rows_to_write = [_row_to_csv_row(r) for r in all_per_image]
        _log(f'寫入 per-image CSV: {csv_path}')

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()
        for row in rows_to_write:
            w.writerow(row)
    n_ok = len(tasks) - len(errors_log)
    _log(f'Done. 成功 {n_ok} 個方法，失敗 {len(errors_log)} 個。')


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        import traceback
        sys.stderr.write(f'[Fatal] {type(e).__name__}: {e}\n')
        sys.stderr.flush()
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
