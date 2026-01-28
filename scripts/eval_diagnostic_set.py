"""
對 diagnostic_candidates.csv 進行 DL model 推論與評估

功能：
- 讀取 diagnostic_candidates.csv
- 自動找最新的 checkpoint
- 逐張推論並計算指標（IoU, FP rate, FN rate, pred_positive_ratio）
- 生成 overlay 視覺化
- 輸出 diagnostic_with_dl_metrics.csv

用法:
  python scripts/eval_diagnostic_set.py
  python scripts/eval_diagnostic_set.py --checkpoint outputs/train_multi_camera_XXX/checkpoints/best.pth
"""

import os
import sys
import argparse
import csv
import json
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm

# 添加專案根目錄到 sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

from models import create_unet_model
import torchvision.transforms.functional as TF


def find_latest_checkpoint():
    """找最新的 train_multi_camera_* 下的 checkpoints/best.pth"""
    outputs_dir = Path('outputs')
    candidates = list(outputs_dir.glob('train_multi_camera_*/checkpoints/best.pth'))
    if not candidates:
        return None
    return str(max(candidates, key=os.path.getmtime))


def load_model(checkpoint_path, device='cpu'):
    """載入訓練好的模型"""
    model = create_unet_model(n_channels=3, n_classes=1)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    if 'epoch' in checkpoint:
        print(f"載入模型: Epoch {checkpoint['epoch']}")
    if 'val_iou' in checkpoint:
        print(f"Val IoU: {checkpoint['val_iou']:.4f}")
    
    return model


def preprocess_image(image_path, image_size=(160, 160)):
    """
    預處理輸入圖片
    
    參數:
        image_path: 圖片路徑
        image_size: 目標尺寸 (height, width)
    
    返回:
        tensor: 預處理後的張量 [1, 3, H, W]
        original_size: 原始圖片尺寸 (width, height)
    """
    image = Image.open(image_path).convert('RGB')
    original_size = image.size  # (width, height)
    
    # 調整尺寸
    image = image.resize((image_size[1], image_size[0]), Image.BILINEAR)
    
    # 轉換為張量 [0, 1]
    tensor = TF.to_tensor(image)
    
    # 添加 batch 維度
    tensor = tensor.unsqueeze(0)
    
    return tensor, original_size


def predict_single_image(model, image_tensor, device, threshold=0.5):
    """
    執行預測
    
    參數:
        model: 模型
        image_tensor: 輸入張量 [1, 3, H, W]
        device: 計算設備
        threshold: 二進制化閾值
    
    返回:
        pred_mask: binary mask [H, W] (0/1)
        pred_prob: probability map [H, W] (0-1)
    """
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        logits = model(image_tensor)
        prob = torch.sigmoid(logits)
        mask = (prob > threshold).float()
    
    # 轉換為 numpy
    prob_np = prob.squeeze().cpu().numpy()  # [H, W]
    mask_np = mask.squeeze().cpu().numpy()  # [H, W]
    
    return mask_np, prob_np


def load_gt_mask(image_path, camera_id, image_id, has_sky=True):
    """
    載入 GT mask
    
    參數:
        image_path: 圖片路徑（用於推導 mask 路徑）
        camera_id: camera ID
        image_id: image ID
        has_sky: 是否有天空（has_sky=FALSE 時返回全 0 mask）
    
    返回:
        gt_mask: mask tensor [H, W] (0-1) 或 None
        mask_exists: bool，mask 檔案是否存在
    """
    if not has_sky:
        # has_sky=FALSE：返回全 0 mask（需要知道尺寸）
        # 先讀取圖片取得尺寸
        img = Image.open(image_path).convert('RGB')
        w, h = img.size  # PIL size 是 (width, height)
        return np.zeros((h, w), dtype=np.float32), True
    
    # 推導 mask 路徑：skyfinder_XXX/images/XXX.jpg -> skyfinder_XXX/masks/XXX.png
    # CSV 中的 path 格式：skyfinder_10870\images\048.jpg
    path_parts = image_path.replace('\\', '/').split('/')
    camera_folder = None
    for part in path_parts:
        if part.startswith('skyfinder_'):
            camera_folder = part
            break
    
    if camera_folder is None:
        # 嘗試從 camera_id 推導
        camera_folder = f'skyfinder_{camera_id}'
    
    # 構建 mask 路徑
    mask_filename = f'{image_id:03d}.png'
    mask_path = os.path.join('data', camera_folder, 'masks', mask_filename)
    
    # 如果找不到，嘗試其他格式
    if not os.path.exists(mask_path):
        # 嘗試從 image_path 推導
        image_dir = os.path.dirname(image_path)
        if 'images' in image_dir:
            mask_dir = image_dir.replace('images', 'masks')
            mask_path = os.path.join(mask_dir, mask_filename)
    
    if not os.path.exists(mask_path):
        return None, False
    
    # 讀取 mask
    mask_img = Image.open(mask_path).convert('L')
    mask_np = np.array(mask_img, dtype=np.float32) / 255.0  # 正規化到 [0, 1]
    
    return mask_np, True


def calculate_metrics_per_image(pred_mask, gt_mask=None, has_gt=True, smooth=1e-6):
    """
    計算每張影像的指標
    
    參數:
        pred_mask: 預測 mask [H, W] (0/1)
        gt_mask: GT mask [H, W] (0-1) 或 None
        has_gt: 是否有 GT
        smooth: 平滑項
    
    返回:
        metrics: 字典，包含 dl_iou, dl_fp_rate, dl_fn_rate, dl_pred_positive_ratio
    """
    pred_b = (pred_mask > 0.5).astype(np.float32)
    
    # pred_positive_ratio：預測為天空的像素比例
    total_pixels = pred_mask.size
    pred_positive_ratio = pred_b.sum() / (total_pixels + smooth)
    
    metrics = {
        'dl_pred_positive_ratio': float(pred_positive_ratio),
    }
    
    if not has_gt or gt_mask is None:
        # 沒有 GT：IoU 和 FN rate 無法計算
        metrics['dl_iou'] = None
        metrics['dl_fn_rate'] = None
        
        # FP rate：預測為天空但沒有 GT（在 no-sky 情況下，所有預測為天空的都是 FP）
        # FP rate = FP / (TN + FP) = pred_positive / total_pixels
        metrics['dl_fp_rate'] = float(pred_positive_ratio)
        return metrics
    
    # 有 GT：計算完整指標
    gt_b = (gt_mask > 0.5).astype(np.float32)
    
    # 確保尺寸一致
    if pred_b.shape != gt_b.shape:
        # 調整 pred_mask 到 GT 尺寸（使用 PIL resize）
        pred_img = Image.fromarray((pred_b * 255).astype(np.uint8))
        pred_img = pred_img.resize((gt_b.shape[1], gt_b.shape[0]), Image.NEAREST)
        pred_b = np.array(pred_img, dtype=np.float32) / 255.0
        pred_b = (pred_b > 0.5).astype(np.float32)
    
    # 計算 TP, FP, FN, TN
    tp = ((pred_b == 1) & (gt_b == 1)).sum()
    fp = ((pred_b == 1) & (gt_b == 0)).sum()
    fn = ((pred_b == 0) & (gt_b == 1)).sum()
    tn = ((pred_b == 0) & (gt_b == 0)).sum()
    
    # IoU
    intersection = tp
    union = tp + fp + fn
    if union == 0:
        iou = 1.0  # 如果 union 為 0，表示兩者都是全 0，IoU = 1
    else:
        iou = intersection / union
    
    # FP rate: FP / (TN + FP) = FP / 總非天空像素
    total_neg = tn + fp
    fp_rate = fp / (total_neg + smooth)
    
    # FN rate: FN / (TP + FN) = FN / 總天空像素
    total_pos = tp + fn
    fn_rate = fn / (total_pos + smooth)
    
    metrics['dl_iou'] = float(iou)
    metrics['dl_fp_rate'] = float(fp_rate)
    metrics['dl_fn_rate'] = float(fn_rate)
    
    return metrics


def create_overlay_for_diagnostic(image_path, gt_mask, pred_mask, output_path, alpha=0.5):
    """
    創建 overlay 視覺化
    
    顏色意義：
      - 藍色：漏檢 (FN) — GT 有天空、預測沒有
      - 紅色：誤判 (FP) — 預測有天空、GT 沒有
    
    參數:
        image_path: 原始圖片路徑
        gt_mask: GT mask [H, W] (0-1) 或 None
        pred_mask: 預測 mask [H, W] (0-1)
        output_path: 輸出路徑
        alpha: 透明度
    """
    # 讀取原始圖片
    image = Image.open(image_path).convert('RGB')
    image_np = np.array(image, dtype=np.float32)  # [H, W, C]
    
    # 確保 pred_mask 與圖片尺寸一致
    if pred_mask.shape[:2] != image_np.shape[:2]:
        pred_img = Image.fromarray((pred_mask * 255).astype(np.uint8))
        pred_img = pred_img.resize((image_np.shape[1], image_np.shape[0]), Image.NEAREST)
        pred_mask = np.array(pred_img, dtype=np.float32) / 255.0
    
    pred_b = (pred_mask > 0.5)
    
    overlay = image_np.copy()
    
    if gt_mask is not None:
        # 確保 gt_mask 與圖片尺寸一致
        if gt_mask.shape[:2] != image_np.shape[:2]:
            from scipy.ndimage import zoom
            h_ratio = image_np.shape[0] / gt_mask.shape[0]
            w_ratio = image_np.shape[1] / gt_mask.shape[1]
            gt_mask = zoom(gt_mask, (h_ratio, w_ratio), order=0)
        
        gt_b = (gt_mask > 0.5)
        
        # 計算錯誤區域
        fn_mask = gt_b & (~pred_b)  # GT 有、預測沒有
        fp_mask = pred_b & (~gt_b)  # 預測有、GT 沒有
        
        # 標示錯誤
        blue = np.array([0, 100, 255], dtype=np.float32)   # FN
        red = np.array([255, 50, 50], dtype=np.float32)    # FP
        
        fn_3d = np.stack([fn_mask] * 3, axis=-1)
        fp_3d = np.stack([fp_mask] * 3, axis=-1)
        
        overlay = np.where(fn_3d, overlay * (1 - alpha) + blue * alpha, overlay)
        overlay = np.where(fp_3d, overlay * (1 - alpha * 0.9) + red * (alpha * 0.9), overlay)
    else:
        # 沒有 GT：只標示預測為天空的區域（用半透明紅色）
        red = np.array([255, 50, 50], dtype=np.float32)
        pred_3d = np.stack([pred_b] * 3, axis=-1)
        overlay = np.where(pred_3d, overlay * (1 - alpha * 0.5) + red * (alpha * 0.5), overlay)
    
    overlay = overlay.clip(0, 255).astype(np.uint8)
    overlay_img = Image.fromarray(overlay)
    overlay_img.save(output_path)


def process_diagnostic_set(csv_path, checkpoint_path, output_csv_path, overlay_dir, image_size=(160, 160), threshold=0.5):
    """
    主處理流程
    
    參數:
        csv_path: 輸入 CSV 路徑
        checkpoint_path: checkpoint 路徑
        output_csv_path: 輸出 CSV 路徑
        overlay_dir: overlay 輸出目錄
        image_size: 圖片尺寸 (height, width)
        threshold: 二進制化閾值
    """
    # 設置設備
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    print(f"使用設備: {device}")
    
    # 載入模型
    print(f"\n載入模型: {checkpoint_path}")
    model = load_model(checkpoint_path, device)
    
    # 創建 overlay 目錄
    os.makedirs(overlay_dir, exist_ok=True)
    
    # 讀取 CSV
    print(f"\n讀取 CSV: {csv_path}")
    rows = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)
    
    print(f"找到 {len(rows)} 張影像")
    
    # 準備輸出 CSV
    output_fieldnames = list(fieldnames) + ['dl_iou', 'dl_fp_rate', 'dl_fn_rate', 'dl_pred_positive_ratio', 'overlay_path']
    
    # 處理每張影像
    print("\n開始處理...")
    results = []
    errors = []
    
    for i, row in enumerate(tqdm(rows, desc='處理影像')):
        try:
            # 解析欄位
            camera_id = row['camera_id']
            image_id = int(row['image_id'])
            path = row['path']
            has_sky_str = row.get('has_sky', 'TRUE').upper()
            has_sky = has_sky_str == 'TRUE'
            
            # 構建完整圖片路徑
            image_path = os.path.join('data', path.replace('\\', '/'))
            if not os.path.exists(image_path):
                errors.append(f"找不到圖片: {image_path}")
                continue
            
            # 預處理圖片
            image_tensor, original_size = preprocess_image(image_path, image_size)
            
            # 推論
            pred_mask, pred_prob = predict_single_image(model, image_tensor, device, threshold)
            
            # 載入 GT mask
            gt_mask, gt_exists = load_gt_mask(image_path, camera_id, image_id, has_sky)
            
            # 計算指標
            metrics = calculate_metrics_per_image(
                pred_mask, 
                gt_mask if gt_exists else None, 
                has_gt=has_sky and gt_exists,
                smooth=1e-6
            )
            
            # 生成 overlay
            overlay_filename = f'camera_{camera_id}_image_{image_id}_overlay.png'
            overlay_path = os.path.join(overlay_dir, overlay_filename)
            
            # 為了 overlay，需要將 pred_mask 調整回原始尺寸
            # 先讀取原始圖片取得尺寸
            orig_img = Image.open(image_path).convert('RGB')
            orig_h, orig_w = orig_img.size[1], orig_img.size[0]
            
            # 調整 pred_mask 到原始尺寸
            if pred_mask.shape != (orig_h, orig_w):
                pred_img = Image.fromarray((pred_mask * 255).astype(np.uint8))
                pred_img = pred_img.resize((orig_w, orig_h), Image.NEAREST)
                pred_mask_orig = np.array(pred_img, dtype=np.float32) / 255.0
            else:
                pred_mask_orig = pred_mask
            
            # 調整 GT mask（如果有）
            gt_mask_orig = None
            if gt_exists and gt_mask is not None:
                if gt_mask.shape != (orig_h, orig_w):
                    gt_img = Image.fromarray((gt_mask * 255).astype(np.uint8))
                    gt_img = gt_img.resize((orig_w, orig_h), Image.NEAREST)
                    gt_mask_orig = np.array(gt_img, dtype=np.float32) / 255.0
                else:
                    gt_mask_orig = gt_mask
            
            create_overlay_for_diagnostic(
                image_path,
                gt_mask_orig,
                pred_mask_orig,
                overlay_path,
                alpha=0.5
            )
            
            # 構建新行
            new_row = row.copy()
            new_row['dl_iou'] = f"{metrics['dl_iou']:.6f}" if metrics['dl_iou'] is not None else ""
            new_row['dl_fp_rate'] = f"{metrics['dl_fp_rate']:.6f}"
            new_row['dl_fn_rate'] = f"{metrics['dl_fn_rate']:.6f}" if metrics['dl_fn_rate'] is not None else ""
            new_row['dl_pred_positive_ratio'] = f"{metrics['dl_pred_positive_ratio']:.6f}"
            new_row['overlay_path'] = overlay_path.replace('\\', '/')  # 統一使用 /
            
            results.append(new_row)
            
            # 清理記憶體
            del image_tensor, pred_mask, pred_prob
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        except Exception as e:
            error_msg = f"處理 {row.get('camera_id', '?')}/{row.get('image_id', '?')} 時發生錯誤: {str(e)}"
            errors.append(error_msg)
            print(f"\n[錯誤] {error_msg}")
            continue
    
    # 寫入輸出 CSV
    print(f"\n寫入輸出 CSV: {output_csv_path}")
    with open(output_csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=output_fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"完成！處理了 {len(results)} 張影像")
    if errors:
        print(f"\n發生 {len(errors)} 個錯誤：")
        for err in errors[:10]:  # 只顯示前 10 個
            print(f"  - {err}")
        if len(errors) > 10:
            print(f"  ... 還有 {len(errors) - 10} 個錯誤")


def main():
    parser = argparse.ArgumentParser(description='對 diagnostic_candidates.csv 進行 DL model 推論與評估')
    parser.add_argument('--input_csv', type=str, default='data/test_data/diagnostic_candidates.csv',
                       help='輸入 CSV 路徑（預設: data/test_data/diagnostic_candidates.csv）')
    parser.add_argument('--output_csv', type=str, default='outputs/diagnostic_with_dl_metrics.csv',
                       help='輸出 CSV 路徑（預設: outputs/diagnostic_with_dl_metrics.csv）')
    parser.add_argument('--overlay_dir', type=str, default='outputs/diagnostic_overlays',
                       help='Overlay 輸出目錄（預設: outputs/diagnostic_overlays）')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Checkpoint 路徑；未指定則自動找最新的 train_multi_camera_*/checkpoints/best.pth')
    parser.add_argument('--image_size', type=int, nargs=2, default=[160, 160],
                       help='圖片尺寸 [height width]（預設: 160 160）')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='二進制化閾值（預設: 0.5）')
    
    args = parser.parse_args()
    
    # 找 checkpoint
    if args.checkpoint:
        checkpoint_path = args.checkpoint
        if not os.path.isfile(checkpoint_path):
            print(f"[錯誤] 找不到 checkpoint: {checkpoint_path}")
            sys.exit(1)
    else:
        checkpoint_path = find_latest_checkpoint()
        if not checkpoint_path:
            print("[錯誤] 找不到任何 outputs/train_multi_camera_*/checkpoints/best.pth")
            print("       請先完成訓練，或使用 --checkpoint 指定路徑")
            sys.exit(1)
    
    print("=" * 60)
    print("  DL Model 診斷評估")
    print("=" * 60)
    print()
    print(f"  輸入 CSV:     {args.input_csv}")
    print(f"  輸出 CSV:     {args.output_csv}")
    print(f"  Overlay 目錄: {args.overlay_dir}")
    print(f"  Checkpoint:   {checkpoint_path}")
    print(f"  圖片尺寸:     {args.image_size}")
    print(f"  閾值:         {args.threshold}")
    print()
    
    # 執行處理
    process_diagnostic_set(
        csv_path=args.input_csv,
        checkpoint_path=checkpoint_path,
        output_csv_path=args.output_csv,
        overlay_dir=args.overlay_dir,
        image_size=tuple(args.image_size),
        threshold=args.threshold
    )
    
    print("\n" + "=" * 60)
    print("  完成")
    print("=" * 60)


if __name__ == '__main__':
    main()
