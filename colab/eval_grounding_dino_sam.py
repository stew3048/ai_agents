"""
Grounding DINO + SAM 評估腳本（Colab 專屬版本，參數化）

使用 Grounding DINO 偵測 "sky" 區域，然後用 SAM 進行分割
所有路徑和參數都可通過 CLI 指定
支援 overlay 輸出

流程：
1. Grounding DINO 偵測標籤為 "sky" 的區域（輸出 bounding box）
2. 如果偵測到，將 bounding box 傳給 SAM 進行分割
3. 如果沒偵測到，輸出全黑 mask
4. 計算 metrics 並生成 overlay

輸出：
- {output_csv}：metrics 摘要 CSV
- {overlay_dir}/*.png：overlay 視覺化影像（如果指定）
"""

import os
import sys
import csv
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from PIL import Image

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# 添加專案根目錄到 Python 路徑
_script_dir = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_script_dir)
sys.path.insert(0, _root)
sys.path.insert(0, _script_dir)

from utils_colab import setup_device, detect_environment
from eval_utils import create_overlay, save_overlay

# 檢查依賴
try:
    import torch
    GROUNDING_DINO_AVAILABLE = False
    SAM_AVAILABLE = False
    
    # 檢查 Grounding DINO
    try:
        from transformers import AutoProcessor, GroundingDinoForObjectDetection
        GROUNDING_DINO_AVAILABLE = True
    except ImportError:
        print("[WARNING] Grounding DINO 未安裝。請執行: pip install transformers")
    
    # 檢查 SAM
    try:
        import sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        SAM_AVAILABLE = True
    except ImportError:
        print("[WARNING] SAM 2.0 未安裝。請執行: pip install sam2")
        
except ImportError:
    print("[WARNING] PyTorch 未安裝")
    GROUNDING_DINO_AVAILABLE = False
    SAM_AVAILABLE = False


# 對比式多目標 prompt：僅保留 label 為 "sky" 的 box（sky 為第一個 phrase，index=0）
CONTRASTIVE_TEXT_PROMPT = "sky . sea . ocean . building . mountain . tree . ground ."
SKY_LABEL_INDEX = 0
MIN_SKY_AREA_RATIO = 0.005   # 0.5%
BOTTOM_ZONE_RATIO = 0.8     # ymin >= height * BOTTOM_ZONE_RATIO 則捨棄

# SAM 模型映射
SAM_MODEL_TYPE_TO_HF_ID = {
    'sam2_hiera_tiny': 'facebook/sam2-hiera-tiny',
    'sam2_hiera_small': 'facebook/sam2-hiera-small',
    'sam2_hiera_base': 'facebook/sam2-hiera-base-plus',
    'sam2_hiera_large': 'facebook/sam2-hiera-large',
}


def load_list_file(list_file: str, base_data_dir: str = "data"):
    """從 list 檔案載入資料"""
    split_list = []
    
    if not os.path.exists(list_file):
        raise FileNotFoundError(f"找不到 list 檔案：{list_file}")
    
    with open(list_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.replace('\\', '/').split('/')
            
            if len(parts) < 3:
                print(f"  警告：無法解析路徑：{line}")
                continue
            
            camera_folder = parts[0]
            if not camera_folder.startswith('skyfinder_'):
                print(f"  警告：無效的 camera 資料夾名稱：{camera_folder}")
                continue
            
            camera_id = camera_folder.replace('skyfinder_', '')
            image_filename = '/'.join(parts[2:])
            
            # 找到對應的 mask
            img_base = os.path.splitext(image_filename)[0]
            possible_mask_names = [
                f"{img_base}.png",
                f"{img_base}.pgm",
                f"{img_base}.jpg",
                image_filename,
            ]
            
            mask_filename = None
            for mask_name in possible_mask_names:
                mask_path = os.path.join(base_data_dir, camera_folder, "masks", mask_name)
                if os.path.exists(mask_path):
                    mask_filename = mask_name
                    break
            
            if mask_filename is None:
                print(f"  警告：找不到 mask 檔案：{line}")
                continue
            
            split_list.append({
                'camera_id': camera_id,
                'image': image_filename,
                'mask': mask_filename,
                'image_path': line
            })
    
    return split_list


def load_gt_mask(image_path, mask_path):
    """
    載入 GT mask
    
    返回:
        gt_mask: numpy array (H, W) 0-1 範圍，或 None
        has_gt: bool，是否有 GT
    """
    if not os.path.exists(mask_path):
        return None, False
    
    try:
        mask_img = Image.open(mask_path).convert('L')  # 轉為灰階
        mask_array = np.array(mask_img)
        
        # 轉換為二值化 mask
        if len(mask_array.shape) == 3:
            mask_array = mask_array[:, :, 0]
        
        # 處理不同格式的 mask：
        # 1. 布林值（True/False）：True 代表天空
        # 2. 數值（0-255）：需要正規化到 0-1，> 0.5 視為天空
        if mask_array.dtype == bool:
            # 布林值：True = 天空
            gt_mask = mask_array.astype(np.float32)
        else:
            # 數值：正規化到 0-1，> 0.5 視為天空
            if mask_array.max() > 1.0:
                mask_array = mask_array.astype(np.float32) / 255.0
            else:
                mask_array = mask_array.astype(np.float32)
            gt_mask = (mask_array > 0.5).astype(np.float32)
        
        # 檢查是否有天空
        has_gt = (gt_mask.sum() > 0)
        
        return gt_mask, has_gt
    except Exception as e:
        print(f"  警告：無法載入 GT mask {mask_path}: {e}")
        import traceback
        traceback.print_exc()
        return None, False


def calculate_metrics_per_image(pred_mask, gt_mask=None, has_gt=True):
    """
    計算單張影像的 metrics
    
    參數:
        pred_mask: numpy array (H, W) 0-1 範圍
        gt_mask: numpy array (H, W) 0-1 範圍，或 None
        has_gt: bool，是否有 GT
    
    返回:
        Dict: {'iou': float, 'fp_rate': float, 'fn_rate': float}
    """
    smooth = 1e-6
    
    if not has_gt or gt_mask is None:
        # No-sky case：只計算 FP rate
        fp = pred_mask.sum()
        total = pred_mask.size
        fp_rate = fp / (total + smooth)
        return {
            'iou': None,
            'fp_rate': fp_rate,
            'fn_rate': None
        }
    
    # 確保尺寸一致
    if pred_mask.shape != gt_mask.shape:
        pred_img = Image.fromarray((pred_mask * 255).astype(np.uint8))
        pred_img = pred_img.resize((gt_mask.shape[1], gt_mask.shape[0]), Image.NEAREST)
        pred_mask = np.array(pred_img, dtype=np.float32) / 255.0
    
    # 二值化
    pred_b = (pred_mask > 0.5).astype(np.float32)
    gt_b = gt_mask.astype(np.float32)
    
    # 計算 metrics
    intersection = (pred_b * gt_b).sum()
    union = pred_b.sum() + gt_b.sum() - intersection
    iou = intersection / (union + smooth)
    
    # FP/FN rate
    fp = ((pred_b == 1) & (gt_b == 0)).sum()
    fn = ((pred_b == 0) & (gt_b == 1)).sum()
    tn = ((pred_b == 0) & (gt_b == 0)).sum()
    tp = ((pred_b == 1) & (gt_b == 1)).sum()
    
    # 避免 tn==0 時 fp/(tn+smooth) 爆掉（異常大 FP_rate）
    fp_rate = fp / (tn + smooth) if tn > 0 else (1.0 if fp > 0 else 0.0)
    fn_rate = fn / (tp + smooth) if tp > 0 else 0.0
    
    return {
        'iou': float(iou),
        'fp_rate': float(fp_rate),
        'fn_rate': float(fn_rate)
    }


def load_grounding_dino_model(model_id='IDEA-Research/grounding-dino-tiny', device='cpu'):
    """
    載入 Grounding DINO 模型與 processor
    
    參數:
        model_id: HuggingFace model ID
        device: 設備
    
    返回:
        processor, model
    """
    if not GROUNDING_DINO_AVAILABLE:
        raise ImportError("Grounding DINO 未安裝。請執行: pip install transformers")
    
    processor = AutoProcessor.from_pretrained(model_id)
    model = GroundingDinoForObjectDetection.from_pretrained(model_id)
    model.to(device)
    model.eval()
    return processor, model


def detect_sky_with_grounding_dino(processor, model, image_path, text_prompt=None,
                                   box_threshold=0.4, text_threshold=0.35, device='cpu',
                                   sky_label_index=SKY_LABEL_INDEX, bottom_zone_ratio=BOTTOM_ZONE_RATIO):
    """僅保留 label 為 sky 的 box；捨棄頂邊在圖片最底部 20% 的 box。"""
    if text_prompt is None:
        text_prompt = CONTRASTIVE_TEXT_PROMPT
    image = Image.open(image_path).convert('RGB')
    text = text_prompt if text_prompt.strip().endswith('.') else f"{text_prompt.strip()} ."
    inputs = processor(images=image, text=text, return_tensors="pt")
    inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    img_width, img_height = image.size
    y_bottom_zone = img_height * bottom_zone_ratio
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.image_processor.post_process_object_detection(
        outputs, threshold=box_threshold, target_sizes=target_sizes
    )[0]
    boxes = []
    scores = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        label_idx = label.item() if hasattr(label, 'item') else int(label)
        if label_idx != sky_label_index:
            continue
        if score.item() < text_threshold:
            continue
        x1, y1, x2, y2 = box.tolist()
        if y1 >= y_bottom_zone:
            continue
        x1 = max(0, min(x1, img_width))
        y1 = max(0, min(y1, img_height))
        x2 = max(0, min(x2, img_width))
        y2 = max(0, min(y2, img_height))
        if x2 > x1 and y2 > y1:
            boxes.append([x1, y1, x2, y2])
            scores.append(score.item())
    if len(boxes) == 0:
        return None, None
    return boxes, scores


def load_sam_model(model_type='sam2_hiera_small', device='cpu'):
    """
    載入 SAM 2.0 模型
    
    參數:
        model_type: SAM 模型類型
        device: 設備
    
    返回:
        predictor: SAM2ImagePredictor
    """
    if not SAM_AVAILABLE:
        raise ImportError("SAM 2.0 未安裝。請執行: pip install sam2")
    
    model_id = SAM_MODEL_TYPE_TO_HF_ID.get(model_type, 'facebook/sam2-hiera-small')
    print(f"載入 SAM 2.0 模型: {model_type} -> {model_id}")
    
    predictor = SAM2ImagePredictor.from_pretrained(model_id, device=device)
    
    return predictor


def segment_with_sam(predictor, image_path, boxes):
    """對每個 sky box 分別調用 SAM predict，再以 np.logical_or 合併所有 mask。"""
    image = Image.open(image_path).convert('RGB')
    image_np = np.array(image)
    h, w = image_np.shape[:2]
    predictor.set_image(image_np)
    merged_mask = np.zeros((h, w), dtype=np.float32)
    for box in boxes:
        x1 = max(0, min(float(box[0]), w - 1))
        y1 = max(0, min(float(box[1]), h - 1))
        x2 = max(0, min(float(box[2]), w))
        y2 = max(0, min(float(box[3]), h))
        if x2 <= x1 or y2 <= y1:
            continue
        sam_box = np.array([[x1, y1, x2, y2]], dtype=np.float32)
        masks, _, _ = predictor.predict(point_coords=None, point_labels=None, box=sam_box, multimask_output=False)
        one_mask = masks[0].astype(np.float32)
        merged_mask = np.logical_or(merged_mask > 0.5, one_mask > 0.5).astype(np.float32)
    return merged_mask


def predict_grounding_dino_sam(grounding_processor, grounding_model, sam_predictor,
                               image_path, text_prompt=None, device='cpu',
                               box_threshold=0.4, text_threshold=0.35,
                               min_sky_area_ratio=MIN_SKY_AREA_RATIO):
    """含面積過濾：合併 mask 面積佔圖 < min_sky_area_ratio 則輸出全黑。"""
    image = Image.open(image_path).convert('RGB')
    h, w = image.size[1], image.size[0]
    boxes, _ = detect_sky_with_grounding_dino(
        grounding_processor, grounding_model, image_path,
        text_prompt=text_prompt, box_threshold=box_threshold,
        text_threshold=text_threshold, device=device
    )
    if boxes is None or len(boxes) == 0:
        return np.zeros((h, w), dtype=np.float32)
    pred_mask = segment_with_sam(sam_predictor, image_path, boxes)
    area_ratio = pred_mask.sum() / (pred_mask.size + 1e-10)
    if area_ratio < min_sky_area_ratio:
        return np.zeros((h, w), dtype=np.float32)
    return pred_mask


def evaluate_per_image_grounding_dino_sam(grounding_processor, grounding_model, sam_predictor,
                                          test_list, device='cpu', overlay_dir=None, data_dir="data",
                                          text_prompt=None, box_threshold=0.4, text_threshold=0.35):
    """
    使用 Grounding DINO + SAM 評估每張影像
    
    返回:
        List[Dict]: 每筆包含 image_path, camera_id, iou, fp_rate, fn_rate, has_gt, overlay_path
    """
    results = []
    
    for item in tqdm(test_list, desc='  Evaluating Grounding DINO + SAM'):
        image_path_rel = item['image_path']
        camera_id = item['camera_id']
        image_filename = item['image']
        mask_filename = item['mask']
        
        # 構建完整路徑
        camera_folder = f"skyfinder_{camera_id}"
        full_image_path = os.path.join(data_dir, camera_folder, "images", image_filename)
        full_mask_path = os.path.join(data_dir, camera_folder, "masks", mask_filename)
        
        if not os.path.exists(full_image_path):
            print(f"  警告：找不到影像：{full_image_path}")
            continue
        
        # Grounding DINO + SAM 推論
        try:
            pred_mask = predict_grounding_dino_sam(
                grounding_processor, grounding_model, sam_predictor,
                full_image_path, text_prompt=text_prompt, device=device,
                box_threshold=box_threshold, text_threshold=text_threshold
            )
        except Exception as e:
            print(f"  錯誤：推論失敗 {full_image_path}: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # 載入 GT mask
        gt_mask, has_gt = load_gt_mask(full_image_path, full_mask_path)
        
        # 計算 metrics
        if has_gt and gt_mask is not None:
            metrics = calculate_metrics_per_image(pred_mask, gt_mask, has_gt=True)
            iou = metrics.get('iou')
            fp_rate = metrics.get('fp_rate', 0.0)
            fn_rate = metrics.get('fn_rate')
        else:
            # No-sky case
            metrics = calculate_metrics_per_image(pred_mask, None, has_gt=False)
            iou = None
            fp_rate = metrics.get('fp_rate', 0.0)
            fn_rate = None
        
        # 生成 overlay（如果指定）
        overlay_path = None
        if overlay_dir:
            try:
                overlay_img = create_overlay(full_image_path, pred_mask, gt_mask)
                overlay_filename = f"{camera_id}_{os.path.splitext(image_filename)[0]}_grounding_dino_sam_overlay.png"
                overlay_path = os.path.join(overlay_dir, overlay_filename)
                save_overlay(overlay_img, overlay_path)
            except Exception as e:
                print(f"  警告：無法生成 overlay {full_image_path}: {e}")
        
        results.append({
            'image_path': image_path_rel,
            'camera_id': camera_id,
            'iou': iou,
            'fp_rate': fp_rate,
            'fn_rate': fn_rate,
            'has_gt': has_gt,
            'overlay_path': overlay_path
        })
    
    return results


def aggregate_per_camera(results):
    """依 camera_id 聚合 metrics"""
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
    
    # 計算平均值
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
    parser = argparse.ArgumentParser(description='評估 Grounding DINO + SAM（Colab 專屬版本，參數化）')
    
    # 路徑參數
    parser.add_argument('--data_dir', type=str, default='data',
                        help='數據根目錄（預設：data）')
    parser.add_argument('--outputs_dir', type=str, default='outputs',
                        help='輸出目錄（預設：outputs）')
    parser.add_argument('--test_list', type=str, default=None,
                        help='test_list.txt 路徑（預設：{outputs_dir}/test_list.txt）')
    parser.add_argument('--output_csv', type=str, default=None,
                        help='輸出 CSV 路徑（預設：{outputs_dir}/metrics_summary.csv）')
    parser.add_argument('--overlay_dir', type=str, default=None,
                        help='overlay 輸出目錄（預設：不輸出 overlay；指定則輸出到該目錄）')
    
    # 模型參數
    parser.add_argument('--grounding_dino_model_id', type=str, default='IDEA-Research/grounding-dino-tiny',
                        help='Grounding DINO 模型 ID（預設：IDEA-Research/grounding-dino-tiny）')
    parser.add_argument('--sam_model_type', type=str, default='sam2_hiera_small',
                        help='SAM 模型類型（預設：sam2_hiera_small）')
    parser.add_argument('--text_prompt', type=str, default=None,
                        help='文字提示（預設：對比式多目標 prompt，僅保留 sky）')
    parser.add_argument('--box_threshold', type=float, default=0.4,
                        help='Grounding DINO box 置信度閾值（預設：0.4）')
    parser.add_argument('--text_threshold', type=float, default=0.35,
                        help='Grounding DINO 文字匹配閾值（預設：0.35）')
    
    # 評估參數
    parser.add_argument('--device', type=str, default=None,
                        help='設備（cpu/cuda/auto，預設：自動偵測環境）')
    
    args = parser.parse_args()
    
    # 環境偵測
    env = detect_environment()
    
    # 設定預設路徑
    if args.test_list is None:
        args.test_list = os.path.join(args.outputs_dir, 'test_list.txt')
    if args.output_csv is None:
        args.output_csv = os.path.join(args.outputs_dir, 'metrics_summary.csv')
    
    # 設定 device
    device, use_amp, env = setup_device(args.device, env)
    
    print("=" * 60)
    print("  評估 Grounding DINO + SAM（Colab 專屬版本，參數化）")
    print("=" * 60)
    print()
    print(f"環境: {env}")
    print(f"設備: {device}")
    print(f"Grounding DINO 模型: {args.grounding_dino_model_id}")
    print(f"SAM 模型: {args.sam_model_type}")
    print(f"文字提示: {args.text_prompt if args.text_prompt else CONTRASTIVE_TEXT_PROMPT}")
    print(f"Box 閾值: {args.box_threshold}")
    print(f"Text 閾值: {args.text_threshold}")
    if args.overlay_dir:
        print(f"Overlay 輸出目錄: {args.overlay_dir}")
    print()
    
    # === 檢查依賴 ===
    if not GROUNDING_DINO_AVAILABLE:
        print("[錯誤] Grounding DINO 未安裝或不可用")
        print("請執行: pip install transformers")
        sys.exit(1)
    
    if not SAM_AVAILABLE:
        print("[錯誤] SAM 2.0 未安裝或不可用")
        print("請執行: pip install sam2")
        sys.exit(1)
    
    # === 載入 test list ===
    print(f"載入 test list: {args.test_list}...")
    if not os.path.exists(args.test_list):
        print(f"[錯誤] 找不到 test_list.txt: {args.test_list}")
        sys.exit(1)
    
    test_split_list = load_list_file(args.test_list, base_data_dir=args.data_dir)
    print(f"  共 {len(test_split_list)} 張測試影像")
    print()
    
    # === 載入模型 ===
    print("載入 Grounding DINO 模型...")
    try:
        grounding_processor, grounding_model = load_grounding_dino_model(
            model_id=args.grounding_dino_model_id, device=device
        )
        print(f"  Grounding DINO 模型載入成功")
    except Exception as e:
        print(f"[錯誤] 無法載入 Grounding DINO 模型: {e}")
        sys.exit(1)
    
    print("載入 SAM 模型...")
    try:
        sam_predictor = load_sam_model(model_type=args.sam_model_type, device=device)
        print(f"  SAM 模型載入成功")
    except Exception as e:
        print(f"[錯誤] 無法載入 SAM 模型: {e}")
        sys.exit(1)
    print()
    
    # === 建立 overlay 目錄 ===
    if args.overlay_dir:
        os.makedirs(args.overlay_dir, exist_ok=True)
    
    # === 評估 ===
    print("開始評估...")
    results = evaluate_per_image_grounding_dino_sam(
        grounding_processor, grounding_model, sam_predictor,
        test_split_list,
        device=device,
        overlay_dir=args.overlay_dir,
        data_dir=args.data_dir,
        text_prompt=args.text_prompt,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold
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
            existing_rows = [row for row in reader if row.get('method') != 'GroundingDINO-SAM']
    
    # 準備新的 rows
    new_rows = []
    for camera_id in sorted(camera_metrics.keys()):
        metrics = camera_metrics[camera_id]
        row = {
            'method': 'GroundingDINO-SAM',
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
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    
    print(f"  已寫入 {output_file}")
    print()
    
    # === 顯示摘要 ===
    print("=" * 60)
    print("  Grounding DINO + SAM 評估結果摘要")
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
