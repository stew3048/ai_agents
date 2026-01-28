"""
對 failure_cases_analysis.csv 中的 25 個樣本執行 SAM 2.0 inference

功能：
- 讀取 failure_cases_analysis.csv（25 個 top-5 worst 樣本）
- 載入 SAM 2.0 模型
- 對每個樣本執行 inference（支援多種提示策略）
- 計算指標（IoU, FP Rate, FN Rate）
- 生成 overlay 視覺化
- 輸出 sam2_failure_cases_metrics.csv

用法:
  python scripts/eval_sam2_failure_cases.py
  python scripts/eval_sam2_failure_cases.py --model_type sam2_hiera_large
"""

import os
import sys
import argparse
import csv
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm

# 添加專案根目錄到 sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

try:
    import sam2
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    SAM2_AVAILABLE = True
except ImportError:
    try:
        # 嘗試替代導入方式
        from sam2 import build_sam2, SAM2ImagePredictor
        SAM2_AVAILABLE = True
    except ImportError:
        SAM2_AVAILABLE = False
        print("[WARNING] SAM 2.0 not available. Please install: pip install sam2")


def parse_float(value):
    """解析浮點數"""
    if value == '' or value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def load_sam2_model(model_type='sam2_hiera_large', device='cuda'):
    """
    載入 SAM 2.0 模型
    
    參數:
        model_type: 模型類型（sam2_hiera_tiny, sam2_hiera_small, sam2_hiera_base, sam2_hiera_large）
        device: 計算設備
    
    返回:
        predictor: SAM2ImagePredictor
    """
    if not SAM2_AVAILABLE:
        raise ImportError("SAM 2.0 is not installed. Please install: pip install sam2")
    
    print(f"載入 SAM 2.0 模型: {model_type}")
    
    # 構建模型
    sam2_model = build_sam2(model_type, device=device)
    
    # 創建 predictor
    predictor = SAM2ImagePredictor(sam2_model)
    
    print(f"  ✓ 模型載入完成")
    
    return predictor


def get_prompt_strategy(subset_name, image_shape):
    """
    根據情境選擇提示策略
    
    參數:
        subset_name: 情境名稱（no-sky, sea-sky-confusable, heavy-occlusion, night, urban）
        image_shape: 圖片尺寸 (height, width)
    
    返回:
        prompt_type: 'box', 'point', 'none'
        prompt_data: 提示資料
    """
    h, w = image_shape
    
    if subset_name == 'no-sky':
        # no-sky: 不使用提示（讓 SAM 自動分割）
        return 'none', None
    
    elif subset_name == 'sea-sky-confusable':
        # sea-sky: box prompt（框選上方區域）
        # 框選上方 30-50% 區域
        y1 = int(h * 0.0)
        y2 = int(h * 0.5)
        x1 = int(w * 0.0)
        x2 = int(w * 1.0)
        return 'box', np.array([[x1, y1, x2, y2]])
    
    elif subset_name == 'heavy-occlusion':
        # heavy-occlusion: point prompt（多點提示）
        # 在圖片上方區域選多個點
        points = []
        for y_offset in [0.1, 0.2, 0.3]:
            for x_offset in [0.25, 0.5, 0.75]:
                x = int(w * x_offset)
                y = int(h * y_offset)
                points.append([x, y])
        points = np.array(points)
        labels = np.ones(len(points), dtype=int)  # 前景點
        return 'point', (points, labels)
    
    elif subset_name == 'night':
        # night: point prompt
        # 在圖片上方中心選點
        x = int(w * 0.5)
        y = int(h * 0.2)
        points = np.array([[x, y]])
        labels = np.array([1])  # 前景點
        return 'point', (points, labels)
    
    elif subset_name == 'urban':
        # urban: box prompt（框選上方區域）
        y1 = int(h * 0.0)
        y2 = int(h * 0.4)
        x1 = int(w * 0.0)
        x2 = int(w * 1.0)
        return 'box', np.array([[x1, y1, x2, y2]])
    
    else:
        # 預設：box prompt（上方區域）
        y1 = int(h * 0.0)
        y2 = int(h * 0.4)
        x1 = int(w * 0.0)
        x2 = int(w * 1.0)
        return 'box', np.array([[x1, y1, x2, y2]])


def predict_with_sam2(predictor, image_path, subset_name):
    """
    使用 SAM 2.0 進行預測
    
    參數:
        predictor: SAM2ImagePredictor
        image_path: 圖片路徑
        subset_name: 情境名稱
    
    返回:
        pred_mask: 預測 mask [H, W] (0/1)
    """
    # 讀取圖片
    image = Image.open(image_path).convert('RGB')
    image_np = np.array(image)
    
    # 設置圖片
    predictor.set_image(image_np)
    
    # 獲取提示策略
    prompt_type, prompt_data = get_prompt_strategy(subset_name, image_np.shape[:2])
    
    # 執行預測
    if prompt_type == 'box':
        masks, scores, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=prompt_data,
            multimask_output=False
        )
    elif prompt_type == 'point':
        points, labels = prompt_data
        masks, scores, _ = predictor.predict(
            point_coords=points,
            point_labels=labels,
            box=None,
            multimask_output=False
        )
    else:  # 'none'
        # 不使用提示，讓 SAM 自動分割（使用全圖預測）
        # 注意：SAM 2.0 需要至少一個提示，所以我們使用圖片中心點
        h, w = image_np.shape[:2]
        points = np.array([[w // 2, h // 2]])
        labels = np.array([1])
        masks, scores, _ = predictor.predict(
            point_coords=points,
            point_labels=labels,
            box=None,
            multimask_output=True
        )
        # 選擇分數最高的 mask
        best_idx = np.argmax(scores)
        masks = masks[best_idx:best_idx+1]
    
    # 取得第一個 mask（如果有多個）
    pred_mask = masks[0].astype(np.float32)  # [H, W] (0/1)
    
    return pred_mask


def load_gt_mask(image_path, camera_id, image_id, has_sky=True):
    """
    載入 GT mask（與 eval_diagnostic_set.py 相同邏輯）
    """
    if not has_sky:
        img = Image.open(image_path).convert('RGB')
        w, h = img.size
        return np.zeros((h, w), dtype=np.float32), True
    
    # 推導 mask 路徑
    path_parts = image_path.replace('\\', '/').split('/')
    camera_folder = None
    for part in path_parts:
        if part.startswith('skyfinder_'):
            camera_folder = part
            break
    
    if camera_folder is None:
        camera_folder = f'skyfinder_{camera_id}'
    
    mask_filename = f'{int(image_id):03d}.png'
    mask_path = os.path.join('data', camera_folder, 'masks', mask_filename)
    
    if not os.path.exists(mask_path):
        image_dir = os.path.dirname(image_path)
        if 'images' in image_dir:
            mask_dir = image_dir.replace('images', 'masks')
            mask_path = os.path.join(mask_dir, mask_filename)
    
    if not os.path.exists(mask_path):
        return None, False
    
    mask_img = Image.open(mask_path).convert('L')
    mask_np = np.array(mask_img, dtype=np.float32) / 255.0
    
    return mask_np, True


def calculate_metrics_per_image(pred_mask, gt_mask=None, has_gt=True, smooth=1e-6):
    """
    計算每張影像的指標（與 eval_diagnostic_set.py 相同邏輯）
    """
    pred_b = (pred_mask > 0.5).astype(np.float32)
    
    total_pixels = pred_mask.size
    pred_positive_ratio = pred_b.sum() / (total_pixels + smooth)
    
    metrics = {
        'sam2_pred_positive_ratio': float(pred_positive_ratio),
    }
    
    if not has_gt or gt_mask is None:
        metrics['sam2_iou'] = None
        metrics['sam2_fn_rate'] = None
        metrics['sam2_fp_rate'] = float(pred_positive_ratio)
        return metrics
    
    gt_b = (gt_mask > 0.5).astype(np.float32)
    
    # 確保尺寸一致
    if pred_b.shape != gt_b.shape:
        pred_img = Image.fromarray((pred_b * 255).astype(np.uint8))
        pred_img = pred_img.resize((gt_b.shape[1], gt_b.shape[0]), Image.NEAREST)
        pred_b = np.array(pred_img, dtype=np.float32) / 255.0
        pred_b = (pred_b > 0.5).astype(np.float32)
    
    tp = ((pred_b == 1) & (gt_b == 1)).sum()
    fp = ((pred_b == 1) & (gt_b == 0)).sum()
    fn = ((pred_b == 0) & (gt_b == 1)).sum()
    tn = ((pred_b == 0) & (gt_b == 0)).sum()
    
    intersection = tp
    union = tp + fp + fn
    if union == 0:
        iou = 1.0
    else:
        iou = intersection / union
    
    total_neg = tn + fp
    fp_rate = fp / (total_neg + smooth)
    
    total_pos = tp + fn
    fn_rate = fn / (total_pos + smooth)
    
    metrics['sam2_iou'] = float(iou)
    metrics['sam2_fp_rate'] = float(fp_rate)
    metrics['sam2_fn_rate'] = float(fn_rate)
    
    return metrics


def create_overlay_for_sam2(image_path, gt_mask, pred_mask, output_path, alpha=0.5):
    """
    創建 SAM 2.0 overlay 視覺化（與 DL 相同格式）
    """
    image = Image.open(image_path).convert('RGB')
    image_np = np.array(image, dtype=np.float32)
    
    if pred_mask.shape[:2] != image_np.shape[:2]:
        pred_img = Image.fromarray((pred_mask * 255).astype(np.uint8))
        pred_img = pred_img.resize((image_np.shape[1], image_np.shape[0]), Image.NEAREST)
        pred_mask = np.array(pred_img, dtype=np.float32) / 255.0
    
    pred_b = (pred_mask > 0.5)
    overlay = image_np.copy()
    
    if gt_mask is not None:
        if gt_mask.shape[:2] != image_np.shape[:2]:
            gt_img = Image.fromarray((gt_mask * 255).astype(np.uint8))
            gt_img = gt_img.resize((image_np.shape[1], image_np.shape[0]), Image.NEAREST)
            gt_mask = np.array(gt_img, dtype=np.float32) / 255.0
        
        gt_b = (gt_mask > 0.5)
        
        fn_mask = gt_b & (~pred_b)
        fp_mask = pred_b & (~gt_b)
        
        blue = np.array([0, 100, 255], dtype=np.float32)
        red = np.array([255, 50, 50], dtype=np.float32)
        
        fn_3d = np.stack([fn_mask] * 3, axis=-1)
        fp_3d = np.stack([fp_mask] * 3, axis=-1)
        
        overlay = np.where(fn_3d, overlay * (1 - alpha) + blue * alpha, overlay)
        overlay = np.where(fp_3d, overlay * (1 - alpha * 0.9) + red * (alpha * 0.9), overlay)
    else:
        # 沒有 GT：只顯示預測
        pred_3d = np.stack([pred_b] * 3, axis=-1)
        green = np.array([50, 255, 50], dtype=np.float32)
        overlay = np.where(pred_3d, overlay * (1 - alpha * 0.5) + green * (alpha * 0.5), overlay)
    
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    overlay_img = Image.fromarray(overlay)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    overlay_img.save(output_path)


def process_failure_cases(input_csv, output_csv, overlay_dir, model_type='sam2_hiera_large', device='cuda'):
    """
    處理 failure_cases_analysis.csv 中的所有樣本
    """
    print(f"讀取: {input_csv}")
    
    rows = []
    with open(input_csv, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    
    print(f"  讀取了 {len(rows)} 筆資料")
    
    # 載入 SAM 2.0 模型
    predictor = load_sam2_model(model_type=model_type, device=device)
    
    # 讀取 diagnostic_with_failure_modes.csv 以獲取 has_sky 和 path 資訊
    diagnostic_data = {}
    diagnostic_csv = 'outputs/diagnostic_with_failure_modes.csv'
    if os.path.exists(diagnostic_csv):
        with open(diagnostic_csv, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (row.get('camera_id', ''), row.get('image_id', ''))
                diagnostic_data[key] = row
    
    print(f"\n開始處理 {len(rows)} 個樣本...")
    print("=" * 60)
    
    results = []
    
    for i, row in enumerate(tqdm(rows, desc="Processing")):
        subset_name = row.get('subset_name', '')
        camera_id = row.get('camera_id', '')
        image_id = row.get('image_id', '')
        
        # 構建圖片路徑
        # 優先從 diagnostic_with_failure_modes.csv 獲取 path
        image_path = None
        if key in diagnostic_data:
            path_from_csv = diagnostic_data[key].get('path', '')
            if path_from_csv:
                # path 格式：skyfinder_4795\images\001.jpg
                image_path = os.path.join('data', path_from_csv.replace('\\', os.sep))
        
        # 如果找不到，嘗試構建路徑
        if not image_path or not os.path.exists(image_path):
            camera_folder = f'skyfinder_{camera_id}'
            image_id_int = int(image_id)
            image_filename = f'{image_id_int:03d}.jpg'
            image_path = os.path.join('data', camera_folder, 'images', image_filename)
        
        # 如果還是找不到，嘗試其他格式
        if not os.path.exists(image_path):
            camera_folder = f'skyfinder_{camera_id}'
            alt_path = os.path.join('data', camera_folder, 'images', f'{image_id}.jpg')
            if os.path.exists(alt_path):
                image_path = alt_path
        
        if not os.path.exists(image_path):
            print(f"\n[WARNING] 找不到圖片: camera_id={camera_id}, image_id={image_id}")
            print(f"  嘗試的路徑: {image_path}")
            continue
        
        # 獲取 has_sky 資訊
        key = (camera_id, image_id)
        has_sky = True
        if key in diagnostic_data:
            has_sky = diagnostic_data[key].get('has_sky', '').upper() == 'TRUE'
        
        # SAM 2.0 預測
        try:
            pred_mask = predict_with_sam2(predictor, image_path, subset_name)
        except Exception as e:
            print(f"\n[ERROR] 預測失敗 {camera_id}/{image_id}: {e}")
            continue
        
        # 載入 GT mask
        gt_mask, gt_exists = load_gt_mask(image_path, camera_id, image_id, has_sky)
        
        # 計算指標
        metrics = calculate_metrics_per_image(pred_mask, gt_mask, has_gt=gt_exists and has_sky)
        
        # 生成 overlay
        overlay_filename = f'camera_{camera_id}_image_{image_id}_sam2_overlay.png'
        overlay_path = os.path.join(overlay_dir, overlay_filename)
        create_overlay_for_sam2(image_path, gt_mask if gt_exists else None, pred_mask, overlay_path)
        
        # 組合結果
        result_row = row.copy()
        result_row.update(metrics)
        result_row['sam2_overlay_path'] = overlay_path
        results.append(result_row)
    
    # 寫入結果 CSV
    print(f"\n寫入結果: {output_csv}")
    if results:
        fieldnames = list(results[0].keys())
        with open(output_csv, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"  完成！寫入了 {len(results)} 筆結果")
    else:
        print("  沒有結果可寫入")


def main():
    parser = argparse.ArgumentParser(description='對 failure_cases_analysis.csv 執行 SAM 2.0 inference')
    parser.add_argument('--input_csv', type=str, default='outputs/failure_cases_analysis.csv',
                       help='輸入 CSV 路徑')
    parser.add_argument('--output_csv', type=str, default='outputs/sam2_failure_cases_metrics.csv',
                       help='輸出 CSV 路徑')
    parser.add_argument('--overlay_dir', type=str, default='outputs/sam2_overlays',
                       help='Overlay 輸出目錄')
    parser.add_argument('--model_type', type=str, default='sam2_hiera_large',
                       choices=['sam2_hiera_tiny', 'sam2_hiera_small', 'sam2_hiera_base', 'sam2_hiera_large'],
                       help='SAM 2.0 模型類型')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='計算設備')
    
    args = parser.parse_args()
    
    if not SAM2_AVAILABLE:
        print("[ERROR] SAM 2.0 is not installed.")
        print("Please install: pip install sam2")
        sys.exit(1)
    
    print("=" * 60)
    print("  SAM 2.0 Failure Cases Evaluation")
    print("=" * 60)
    print()
    print(f"  輸入 CSV:     {args.input_csv}")
    print(f"  輸出 CSV:     {args.output_csv}")
    print(f"  Overlay 目錄: {args.overlay_dir}")
    print(f"  模型類型:     {args.model_type}")
    print(f"  計算設備:     {args.device}")
    print()
    
    # 檢查設備
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("[WARNING] CUDA 不可用，切換到 CPU")
        args.device = 'cpu'
    
    process_failure_cases(
        args.input_csv,
        args.output_csv,
        args.overlay_dir,
        model_type=args.model_type,
        device=args.device
    )
    
    print("\n" + "=" * 60)
    print("  完成")
    print("=" * 60)


if __name__ == '__main__':
    main()
