"""
評估 CLIPSeg（使用固定 test_list.txt）

輸入：
- outputs/test_list.txt（固定，每行一個相對路徑）
- CLIPSeg 模型（CIDAS/clipseg-rd64-refined）

處理：
- 讀取 test_list.txt 的每張圖（相對路徑需加上 data/ 前綴載入）
- 用 text prompt "sky" 跑 CLIPSeg
- 計算 IoU / FP / FN（與 GT mask 對齊）
- Per-camera 統計

輸出：
- 追加到 outputs/metrics_summary.csv（method='CLIPSeg'）
"""

import os
import sys
import csv
import argparse
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

_script_dir = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_script_dir)
sys.path.insert(0, _root)
sys.path.insert(0, _script_dir)
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

from eval_sam2_failure_cases import (
    load_gt_mask,
    calculate_metrics_per_image,
)
from scripts.train_in_domain import load_list_file
import numpy as np
from PIL import Image

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import torch
    from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
    CLIPSEG_AVAILABLE = True
except ImportError as e:
    CLIPSEG_AVAILABLE = False
    print(f"[WARNING] CLIPSeg 未安裝: {e}")
    print("請執行: pip install transformers timm")


def load_clipseg_model(model_id='CIDAS/clipseg-rd64-refined', device='cpu'):
    """載入 CLIPSeg 模型與 processor"""
    if not CLIPSEG_AVAILABLE:
        raise ImportError("CLIPSeg 未安裝。請執行: pip install transformers timm")
    processor = CLIPSegProcessor.from_pretrained(model_id)
    model = CLIPSegForImageSegmentation.from_pretrained(model_id)
    model.to(device)
    model.eval()
    return processor, model


def _dynamic_threshold(heatmap, low=0.25, high=0.75):
    """依 heatmap 最大值與平均值計算動態二值化門檻。"""
    h_max = float(np.max(heatmap))
    h_mean = float(np.mean(heatmap))
    # 介於 mean 與 max 之間，並限制在 [low, high]
    thresh = 0.4 * h_max + 0.6 * h_mean
    return max(low, min(high, thresh))


def _spatial_weight_map(h, w, top_weight=1.2, bottom_weight=0.8):
    """上 50% 權重 top_weight，下 50% 權重 bottom_weight。"""
    weight = np.ones((h, w), dtype=np.float32) * bottom_weight
    mid = h // 2
    weight[:mid, :] = top_weight
    return weight


def predict_clipseg(processor, model, image_path, text_prompts=None, device='cpu', threshold=0.5,
                    use_dynamic_threshold=True,
                    use_spatial_prior=True,
                    use_morphology_open=True,
                    use_multi_prompt_contrast=True,
                    morph_open_kernel_size=5):
    """
    單張圖用 CLIPSeg 預測 sky mask。
    
    可選改進：
    - use_dynamic_threshold: 依 heatmap max/mean 動態門檻
    - use_spatial_prior: 上 50% 權重 1.2、下 50% 權重 0.8
    - use_morphology_open: OPEN 運算移除細碎雜訊
    - use_multi_prompt_contrast: 同時預測 sky 與 building，building > sky 的像素強制為 0
    """
    if text_prompts is None:
        text_prompts = ['sky']

    image = Image.open(image_path).convert('RGB')
    image_np = np.array(image)
    orig_h, orig_w = image_np.shape[:2]

    # Sky heatmap：單一 prompt 推論
    inputs_sky = processor(
        text=['sky'],
        images=image,
        return_tensors='pt',
        padding=True,
    )
    inputs_sky = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs_sky.items()}
    with torch.no_grad():
        out_sky = model(**inputs_sky)
    sky_logits = out_sky.logits.cpu().float().numpy()
    sky_logits = np.squeeze(sky_logits)
    if sky_logits.ndim != 2:
        sky_logits = sky_logits[0]

    # Building heatmap：僅在需要多 prompt 對比時再跑一次
    if use_multi_prompt_contrast:
        inputs_building = processor(
            text=['building'],
            images=image,
            return_tensors='pt',
            padding=True,
        )
        inputs_building = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs_building.items()}
        with torch.no_grad():
            out_building = model(**inputs_building)
        building_logits = out_building.logits.cpu().float().numpy()
        building_logits = np.squeeze(building_logits)
        if building_logits.ndim != 2:
            building_logits = building_logits[0]
    else:
        building_logits = None

    # sigmoid -> heatmap（先不二值化）
    sky_heat = 1.0 / (1.0 + np.exp(-np.asarray(sky_logits, dtype=np.float64)))
    sky_heat = np.asarray(sky_heat, dtype=np.float32)
    if building_logits is not None:
        building_heat = 1.0 / (1.0 + np.exp(-np.asarray(building_logits, dtype=np.float64)))
        building_heat = np.asarray(building_heat, dtype=np.float32)
    else:
        building_heat = None

    # 空間先驗：上 50% 權重 1.2，下 50% 權重 0.8
    h, w = sky_heat.shape[:2]
    if use_spatial_prior:
        weight = _spatial_weight_map(h, w, top_weight=1.2, bottom_weight=0.8)
        sky_heat = sky_heat * weight
        sky_heat = np.clip(sky_heat, 0.0, 1.0)

    # 多 prompt 對比：building > sky 的像素之後強制為 0（先標記，二值化後再遮掉）
    if use_multi_prompt_contrast and building_heat is not None:
        building_wins = (building_heat >= sky_heat)
    else:
        building_wins = None

    # 動態閾值或固定 0.5
    if use_dynamic_threshold:
        thresh = _dynamic_threshold(sky_heat, low=0.25, high=0.75)
    else:
        thresh = threshold
    pred = (sky_heat > thresh).astype(np.float32)
    if building_wins is not None:
        pred[building_wins] = 0.0

    # 形態學 OPEN 移除細碎雜訊（FP）
    if use_morphology_open and CV2_AVAILABLE and pred.shape[0] >= 3 and pred.shape[1] >= 3:
        pred_uint8 = (pred * 255).astype(np.uint8)
        k = max(3, morph_open_kernel_size)
        if k % 2 == 0:
            k += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        pred_uint8 = cv2.morphologyEx(pred_uint8, cv2.MORPH_OPEN, kernel)
        pred = pred_uint8.astype(np.float32) / 255.0

    # resize 回原圖尺寸
    if pred.shape[0] != orig_h or pred.shape[1] != orig_w:
        pil = Image.fromarray((pred * 255).astype(np.uint8))
        pil = pil.resize((orig_w, orig_h), Image.NEAREST)
        pred = np.array(pil, dtype=np.float32) / 255.0

    return pred


def resolve_image_path(relative_path, base_data_dir="data"):
    """將相對路徑轉換為絕對路徑"""
    full_path = os.path.join(base_data_dir, relative_path)
    if os.path.exists(full_path):
        return os.path.abspath(full_path)
    return None


def create_overlay(image_path, pred_mask, gt_mask=None, alpha=0.5):
    """
    創建 overlay 視覺化
    
    參數:
        image_path: 原始影像路徑
        pred_mask: 預測 mask（numpy array，0-1 範圍或 0-255）
        gt_mask: GT mask（numpy array，0-1 範圍或 0-255），可選
        alpha: overlay 透明度（0-1）
    
    返回:
        PIL Image: overlay 影像
    """
    # 載入原始影像
    image = Image.open(image_path).convert('RGB')
    image_np = np.array(image, dtype=np.float32)
    
    # 正規化 pred_mask
    if pred_mask.max() > 1.0:
        pred_mask = pred_mask.astype(np.float32) / 255.0
    else:
        pred_mask = pred_mask.astype(np.float32)
    
    # 調整 pred_mask 尺寸（如果需要）
    if pred_mask.shape[:2] != image_np.shape[:2]:
        pred_img = Image.fromarray((pred_mask * 255).astype(np.uint8))
        pred_img = pred_img.resize((image_np.shape[1], image_np.shape[0]), Image.NEAREST)
        pred_mask = np.array(pred_img, dtype=np.float32) / 255.0
    
    pred_b = (pred_mask > 0.5)
    overlay = image_np.copy()
    
    if gt_mask is not None:
        # 正規化 gt_mask
        if gt_mask.max() > 1.0:
            gt_mask = gt_mask.astype(np.float32) / 255.0
        else:
            gt_mask = gt_mask.astype(np.float32)
        
        # 調整 gt_mask 尺寸（如果需要）
        if gt_mask.shape[:2] != image_np.shape[:2]:
            gt_img = Image.fromarray((gt_mask * 255).astype(np.uint8))
            gt_img = gt_img.resize((image_np.shape[1], image_np.shape[0]), Image.NEAREST)
            gt_mask = np.array(gt_img, dtype=np.float32) / 255.0
        
        gt_b = (gt_mask > 0.5)
        
        # TP: 綠色（預測正確的天空區域）
        # FP: 紅色（誤判為天空）
        # FN: 藍色（漏判的天空區域）
        fn_mask = gt_b & (~pred_b)  # False Negative
        fp_mask = pred_b & (~gt_b)  # False Positive
        tp_mask = gt_b & pred_b      # True Positive
        
        blue = np.array([0, 100, 255], dtype=np.float32)   # FN: 藍色
        red = np.array([255, 50, 50], dtype=np.float32)   # FP: 紅色
        green = np.array([50, 255, 50], dtype=np.float32) # TP: 綠色
        
        fn_3d = np.stack([fn_mask] * 3, axis=-1)
        fp_3d = np.stack([fp_mask] * 3, axis=-1)
        tp_3d = np.stack([tp_mask] * 3, axis=-1)
        
        # 先畫 TP（綠色，較淡）
        overlay = np.where(tp_3d, overlay * (1 - alpha * 0.3) + green * (alpha * 0.3), overlay)
        # 再畫 FN（藍色）
        overlay = np.where(fn_3d, overlay * (1 - alpha) + blue * alpha, overlay)
        # 最後畫 FP（紅色）
        overlay = np.where(fp_3d, overlay * (1 - alpha * 0.9) + red * (alpha * 0.9), overlay)
    else:
        # 沒有 GT：只顯示預測（綠色）
        pred_3d = np.stack([pred_b] * 3, axis=-1)
        green = np.array([50, 255, 50], dtype=np.float32)
        overlay = np.where(pred_3d, overlay * (1 - alpha * 0.5) + green * (alpha * 0.5), overlay)
    
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    overlay_img = Image.fromarray(overlay)
    
    return overlay_img


def evaluate_per_image_clipseg(processor, model, test_list, device='cpu', overlay_dir=None,
                                use_dynamic_threshold=True,
                                use_spatial_prior=True,
                                use_morphology_open=True,
                                use_multi_prompt_contrast=True,
                                morph_open_kernel_size=5):
    """
    使用 CLIPSeg 評估每張影像
    
    返回:
        List[Dict]: 每筆包含 image_path, camera_id, iou, fp_rate, fn_rate, has_gt, overlay_path
    """
    results = []
    
    for item in tqdm(test_list, desc='  Evaluating CLIPSeg'):
        if isinstance(item, dict):
            camera_id = item.get('camera_id', 'unknown')
            image_filename = item.get('image', '')
            relative_path = item.get('image_path', f"skyfinder_{camera_id}/images/{image_filename}")
            image_path = os.path.join("data", f"skyfinder_{camera_id}", "images", image_filename)
        else:
            relative_path = item
            camera_id = 'unknown'
            image_filename = os.path.basename(item)
            image_path = resolve_image_path(relative_path)
        
        if not image_path or not os.path.exists(image_path):
            print(f"  警告：找不到影像：{relative_path}")
            continue
        
        # CLIPSeg 推論
        try:
            pred_mask = predict_clipseg(
                processor, model, image_path,
                device=device,
                use_dynamic_threshold=use_dynamic_threshold,
                use_spatial_prior=use_spatial_prior,
                use_morphology_open=use_morphology_open,
                use_multi_prompt_contrast=use_multi_prompt_contrast,
                morph_open_kernel_size=morph_open_kernel_size,
            )
        except Exception as e:
            print(f"  錯誤：CLIPSeg 推論失敗 {image_path}: {e}")
            continue
        
        # 載入 GT mask（需傳入 camera_id 與 image_id，從檔名解析）
        image_filename = item.get('image', '') if isinstance(item, dict) else os.path.basename(relative_path)
        try:
            image_id = int(os.path.splitext(image_filename)[0])
        except ValueError:
            image_id = 0
        gt_mask, has_gt = load_gt_mask(image_path, camera_id, image_id, has_sky=True)
        
        # 計算 metrics（calculate_metrics_per_image 回傳 sam2_iou / sam2_fp_rate / sam2_fn_rate）
        if has_gt and gt_mask is not None:
            metrics = calculate_metrics_per_image(pred_mask, gt_mask, has_gt=True)
            iou = metrics.get('sam2_iou', metrics.get('iou', 0.0))
            fp_rate = metrics.get('sam2_fp_rate', metrics.get('fp_rate', 0.0))
            fn_rate = metrics.get('sam2_fn_rate', metrics.get('fn_rate', 0.0))
        else:
            # No-sky case
            metrics = calculate_metrics_per_image(pred_mask, None, has_gt=False)
            iou = None
            fp_rate = metrics.get('sam2_fp_rate', metrics.get('fp_rate', 0.0))
            fn_rate = None
        
        # 生成 overlay（如果指定）
        overlay_path = None
        if overlay_dir:
            try:
                overlay_img = create_overlay(image_path, pred_mask, gt_mask)
                overlay_filename = f"{camera_id}_{os.path.splitext(image_filename)[0]}_clipseg_overlay.png"
                overlay_path = os.path.join(overlay_dir, overlay_filename)
                os.makedirs(overlay_dir, exist_ok=True)
                overlay_img.save(overlay_path)
            except Exception as e:
                print(f"  警告：無法生成 overlay {image_path}: {e}")
        
        results.append({
            'image_path': relative_path,
            'camera_id': camera_id,
            'iou': iou,
            'fp_rate': fp_rate,
            'fn_rate': fn_rate,
            'has_gt': has_gt,
            'overlay_path': overlay_path
        })
    
    return results


def aggregate_per_camera(results):
    """
    依 camera_id 聚合 metrics（與 eval_dl_final.py 相同邏輯）
    """
    camera_stats = defaultdict(lambda: {
        'images': [],
        'num_images': 0,
        'iou_sum': 0.0,
        'fp_rate_sum': 0.0,
        'fn_rate_sum': 0.0,
        'num_with_sky': 0,
        'num_no_sky': 0,
        'fp_rate_nosky_sum': 0.0
    })
    
    for r in results:
        camera_id = r['camera_id']
        stats = camera_stats[camera_id]
        
        stats['images'].append(r)
        stats['num_images'] += 1
        stats['fp_rate_sum'] += r['fp_rate']
        
        if r['has_gt']:
            stats['num_with_sky'] += 1
            if r['iou'] is not None:
                stats['iou_sum'] += r['iou']
            if r['fn_rate'] is not None:
                stats['fn_rate_sum'] += r['fn_rate']
        else:
            stats['num_no_sky'] += 1
            stats['fp_rate_nosky_sum'] += r['fp_rate']
    
    aggregated = {}
    for camera_id, stats in camera_stats.items():
        num_images = stats['num_images']
        num_with_sky = stats['num_with_sky']
        num_no_sky = stats['num_no_sky']
        
        aggregated[camera_id] = {
            'num_images': num_images,
            'iou_mean': stats['iou_sum'] / num_with_sky if num_with_sky > 0 else None,
            'fp_mean': stats['fp_rate_sum'] / num_images,
            'fn_mean': stats['fn_rate_sum'] / num_with_sky if num_with_sky > 0 else None,
            'fp_rate_nosky': stats['fp_rate_nosky_sum'] / num_no_sky if num_no_sky > 0 else None
        }
    
    return aggregated


def main():
    parser = argparse.ArgumentParser(description='評估 CLIPSeg（使用固定 test_list.txt）')
    parser.add_argument('--test_list', type=str, default='outputs/test_list.txt',
                        help='test_list.txt 路徑')
    parser.add_argument('--model_id', type=str, default='CIDAS/clipseg-rd64-refined',
                        help='CLIPSeg 模型 ID')
    parser.add_argument('--device', type=str, default='cpu',
                        help='設備（cpu/cuda）')
    parser.add_argument('--output_csv', type=str, default='outputs/metrics_summary.csv',
                        help='輸出 CSV 路徑')
    parser.add_argument('--overlay_dir', type=str, default='outputs/clipseg_overlays',
                        help='overlay 輸出目錄（預設：outputs/clipseg_overlays）')
    # CLIPSeg 改進開關（預設全開）
    parser.add_argument('--no_dynamic_threshold', action='store_true',
                        help='關閉動態閾值（改回固定 0.5）')
    parser.add_argument('--no_spatial_prior', action='store_true',
                        help='關閉空間先驗加權')
    parser.add_argument('--no_morphology_open', action='store_true',
                        help='關閉形態學 OPEN 除雜訊')
    parser.add_argument('--no_multi_prompt_contrast', action='store_true',
                        help='關閉 sky vs building 多 prompt 對比')
    parser.add_argument('--morph_kernel', type=int, default=5,
                        help='形態學 OPEN 的 kernel 大小（預設 5）')
    args = parser.parse_args()
    
    test_list_file = args.test_list
    
    print("=" * 60)
    print("  評估 CLIPSeg（使用固定 test_list.txt）")
    print("=" * 60)
    print()
    
    # === 檢查 CLIPSeg 可用性 ===
    if not CLIPSEG_AVAILABLE:
        print("[錯誤] CLIPSeg 未安裝或不可用")
        print("請執行: pip install transformers timm")
        sys.exit(1)
    
    print(f"  Test list:    {test_list_file}")
    print(f"  Model ID:     {args.model_id}")
    print(f"  Device:       {args.device}")
    if args.overlay_dir:
        print(f"  Overlay 輸出目錄: {args.overlay_dir}")
    print()
    
    # === 載入 test list ===
    print(f"載入 test list: {test_list_file}...")
    if not os.path.exists(test_list_file):
        print(f"[錯誤] 找不到 test_list.txt: {test_list_file}")
        sys.exit(1)
    
    test_split_list = load_list_file(test_list_file)
    print(f"  共 {len(test_split_list)} 張測試影像")
    print()
    
    # === 載入 CLIPSeg 模型 ===
    print("載入 CLIPSeg 模型...")
    try:
        processor, model = load_clipseg_model(model_id=args.model_id, device=args.device)
        print(f"  CLIPSeg 模型載入成功")
    except Exception as e:
        print(f"[錯誤] 無法載入 CLIPSeg 模型: {e}")
        sys.exit(1)
    print()
    
    # === 建立 overlay 目錄 ===
    if args.overlay_dir:
        os.makedirs(args.overlay_dir, exist_ok=True)
    
    # === 評估 ===
    print("開始評估...")
    results = evaluate_per_image_clipseg(
        processor, model, test_split_list,
        device=args.device,
        overlay_dir=args.overlay_dir,
        use_dynamic_threshold=not args.no_dynamic_threshold,
        use_spatial_prior=not args.no_spatial_prior,
        use_morphology_open=not args.no_morphology_open,
        use_multi_prompt_contrast=not args.no_multi_prompt_contrast,
        morph_open_kernel_size=args.morph_kernel,
    )
    print(f"  完成，共評估 {len(results)} 張影像")
    if args.overlay_dir:
        overlay_count = sum(1 for r in results if r.get('overlay_path'))
        print(f"  生成 {overlay_count} 張 overlay")
    print()
    
    # === 聚合 per-camera ===
    print("聚合 per-camera metrics...")
    camera_metrics = aggregate_per_camera(results)
    print(f"  找到 {len(camera_metrics)} 個 camera")
    print()
    
    # === 輸出結果 ===
    print("輸出結果...")
    output_file = args.output_csv
    
    # 讀取現有的 CSV（如果存在）
    existing_rows = []
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            existing_rows = [row for row in reader if row.get('method') != 'CLIPSeg']
    
    # 準備新的 rows
    new_rows = []
    for camera_id in sorted(camera_metrics.keys()):
        metrics = camera_metrics[camera_id]
        row = {
            'method': 'CLIPSeg',
            'camera_id': camera_id,
            'num_images': str(metrics['num_images']),
            'IoU_mean': f"{metrics['iou_mean']:.6f}" if metrics['iou_mean'] is not None else '',
            'FP_mean': f"{metrics['fp_mean']:.6f}",
            'FN_mean': f"{metrics['fn_mean']:.6f}" if metrics['fn_mean'] is not None else '',
            'FP_rate_nosky': f"{metrics['fp_rate_nosky']:.6f}" if metrics['fp_rate_nosky'] is not None else ''
        }
        new_rows.append(row)
    
    # 合併並寫入
    all_rows = existing_rows + new_rows
    fieldnames = ['method', 'camera_id', 'num_images', 'IoU_mean', 'FP_mean', 'FN_mean', 'FP_rate_nosky']
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    
    print(f"  已寫入 {output_file}")
    print()
    
    # === 顯示摘要 ===
    print("=" * 60)
    print("  CLIPSeg 評估結果摘要")
    print("=" * 60)
    print()
    for camera_id in sorted(camera_metrics.keys()):
        metrics = camera_metrics[camera_id]
        print(f"Camera {camera_id}:")
        print(f"  影像數: {metrics['num_images']}")
        if metrics['iou_mean'] is not None:
            print(f"  IoU: {metrics['iou_mean']:.4f}")
            print(f"  FN Rate: {metrics['fn_mean']:.4f}")
        print(f"  FP Rate: {metrics['fp_mean']:.4f}")
        if metrics['fp_rate_nosky'] is not None:
            print(f"  FP Rate (no-sky): {metrics['fp_rate_nosky']:.4f}")
        print()
    
    print("=" * 60)
    print("  評估完成")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
