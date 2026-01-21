"""
比較 U-Net 與 CV Baseline 在 val set 上的 overlay

輸出：
- outputs/compare_val/036~040 的 U-Net overlay
- outputs/compare_val/036~040 的 CV overlay
- 每張圖的 IoU 比較
"""

import os
import sys
import json
import torch
import numpy as np
from PIL import Image
import cv2

# 加入專案路徑
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.unet import UNetSmall
from src.models.cv_baseline import predict_sky_mask_cv


# ============================================================
# 設定
# ============================================================
SPLITS_JSON = 'outputs/splits_10066.json'
IMAGES_DIR = 'data/skyfinder_sample/images'
MASKS_DIR = 'data/skyfinder_sample/masks'
UNET_MODEL = 'outputs/unet_10066_best.pt'
OUTPUT_DIR = 'outputs/compare_val'
IMAGE_SIZE = (256, 256)


# ============================================================
# Helper functions
# ============================================================
def create_overlay(image_np, pred_mask, gt_mask):
    """
    建立 overlay
    - 綠色：TP（正確預測天空）
    - 紅色：FP（錯誤預測為天空）
    - 藍色：FN（漏掉的天空）
    """
    result = image_np.astype(np.float32).copy()
    
    tp = (pred_mask == 1) & (gt_mask == 1)
    fp = (pred_mask == 1) & (gt_mask == 0)
    fn = (pred_mask == 0) & (gt_mask == 1)
    
    alpha = 0.5
    result[tp] = result[tp] * (1 - alpha) + np.array([0, 255, 0]) * alpha
    result[fp] = result[fp] * (1 - alpha) + np.array([255, 0, 0]) * alpha
    result[fn] = result[fn] * (1 - alpha) + np.array([0, 0, 255]) * alpha
    
    return result.clip(0, 255).astype(np.uint8)


def compute_iou(pred, gt):
    """計算 IoU"""
    smooth = 1e-6
    intersection = (pred * gt).sum()
    union = pred.sum() + gt.sum() - intersection
    return (intersection + smooth) / (union + smooth)


def load_and_preprocess(image_path, mask_path, size):
    """載入並預處理圖片和 mask"""
    # 載入原圖（用於視覺化）
    image_orig = Image.open(image_path).convert('RGB')
    mask_orig = Image.open(mask_path).convert('L')
    
    # Resize
    image_resized = image_orig.resize(size, Image.BILINEAR)
    mask_resized = mask_orig.resize(size, Image.NEAREST)
    
    # 轉換為 numpy
    image_np = np.array(image_resized)
    mask_np = np.array(mask_resized)
    
    # 二值化 mask
    mask_np = (mask_np > 127).astype(np.uint8)
    
    return image_np, mask_np


def predict_unet(model, image_np, device):
    """用 U-Net 預測"""
    # 轉換為 tensor
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
    image_tensor = image_tensor.unsqueeze(0).to(device)
    
    # 預測
    with torch.no_grad():
        logits = model(image_tensor)
        probs = torch.sigmoid(logits)
        pred = (probs > 0.5).float()
    
    return pred.squeeze().cpu().numpy().astype(np.uint8)


def predict_cv(image_np):
    """用 CV Baseline 預測"""
    pred = predict_sky_mask_cv(image_np)
    # predict_sky_mask_cv 已經回傳 {0, 1}，不需要再閾值化
    return pred.astype(np.uint8)


# ============================================================
# 主程式
# ============================================================
def main():
    print('=' * 60)
    print('  Compare U-Net vs CV Baseline on Val Set')
    print('=' * 60)
    print()
    
    # 建立輸出目錄
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 載入 splits
    with open(SPLITS_JSON, 'r') as f:
        splits = json.load(f)
    
    val_samples = splits['val'][:5]  # 只取前 5 張 (036-040)
    print(f'Val samples: {[s["image"] for s in val_samples]}')
    print()
    
    # 載入 U-Net 模型
    print('Loading U-Net model...')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = UNetSmall(in_channels=3, out_channels=1).to(device)
    checkpoint = torch.load(UNET_MODEL, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f'  Loaded from: {UNET_MODEL}')
    print(f'  Best epoch: {checkpoint["epoch"]}, Val IoU: {checkpoint["val_iou"]*100:.2f}%')
    print()
    
    # 處理每張圖
    print('Processing...')
    print('-' * 60)
    print(f'  {"Image":>8} | {"U-Net IoU":>10} | {"CV IoU":>10} | {"Diff":>8}')
    print('-' * 60)
    
    results = []
    
    for sample in val_samples:
        image_name = sample['image']
        mask_name = sample['mask']
        base_name = image_name.replace('.jpg', '')
        
        image_path = os.path.join(IMAGES_DIR, image_name)
        mask_path = os.path.join(MASKS_DIR, mask_name)
        
        # 載入圖片
        image_np, gt_np = load_and_preprocess(image_path, mask_path, IMAGE_SIZE)
        
        # U-Net 預測
        pred_unet = predict_unet(model, image_np, device)
        iou_unet = compute_iou(pred_unet, gt_np)
        
        # CV 預測
        pred_cv = predict_cv(image_np)
        iou_cv = compute_iou(pred_cv, gt_np)
        
        # 建立 overlay
        overlay_unet = create_overlay(image_np, pred_unet, gt_np)
        overlay_cv = create_overlay(image_np, pred_cv, gt_np)
        
        # 儲存
        # 原圖
        Image.fromarray(image_np).save(
            os.path.join(OUTPUT_DIR, f'{base_name}_1_image.png'))
        # GT mask
        Image.fromarray((gt_np * 255).astype(np.uint8)).save(
            os.path.join(OUTPUT_DIR, f'{base_name}_2_gt.png'))
        # U-Net overlay
        Image.fromarray(overlay_unet).save(
            os.path.join(OUTPUT_DIR, f'{base_name}_3_unet_overlay.png'))
        # CV overlay
        Image.fromarray(overlay_cv).save(
            os.path.join(OUTPUT_DIR, f'{base_name}_4_cv_overlay.png'))
        
        diff = (iou_unet - iou_cv) * 100
        print(f'  {base_name:>8} | {iou_unet*100:>9.2f}% | {iou_cv*100:>9.2f}% | {diff:>+7.2f}%')
        
        results.append({
            'image': base_name,
            'iou_unet': iou_unet,
            'iou_cv': iou_cv
        })
    
    print('-' * 60)
    
    # 平均
    avg_unet = np.mean([r['iou_unet'] for r in results])
    avg_cv = np.mean([r['iou_cv'] for r in results])
    
    print()
    print(f'  Average U-Net IoU: {avg_unet*100:.2f}%')
    print(f'  Average CV IoU:    {avg_cv*100:.2f}%')
    print(f'  Improvement:       {(avg_unet - avg_cv)*100:+.2f}%')
    print()
    print(f'Output saved to: {OUTPUT_DIR}/')
    print()
    print('Files per image:')
    print('  *_1_image.png       - Original image')
    print('  *_2_gt.png          - Ground truth mask')
    print('  *_3_unet_overlay.png - U-Net prediction overlay')
    print('  *_4_cv_overlay.png   - CV baseline prediction overlay')
    print()
    print('Overlay colors:')
    print('  Green = TP (correct sky)')
    print('  Red   = FP (false positive)')
    print('  Blue  = FN (missed sky)')
    print()
    print('=' * 60)


if __name__ == '__main__':
    main()
