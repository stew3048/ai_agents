"""
比較 CV Baseline 與 U-Net 在 Test Set 上的視覺化

輸出：
- outputs/compare_cv_unet/：每張 test 圖片的四宮格比較
  1) 原圖
  2) GT overlay
  3) CV pred overlay
  4) U-Net pred overlay
- outputs/compare_cv_unet/top5/：U-Net 改善最大的前 5 張
"""

import os
import sys
import json
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# 加入專案路徑
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.unet import UNetSmall
from src.models.cv_baseline import predict_sky_mask_cv


# ============================================================
# 設定
# ============================================================
CONFIG = {
    'splits_json': 'outputs/splits_10066.json',
    'images_dir': 'data/skyfinder_sample/images',
    'masks_dir': 'data/skyfinder_sample/masks',
    'unet_model': 'outputs/unet_10066_best.pt',
    'output_dir': 'outputs/compare_cv_unet',
    'top5_dir': 'outputs/compare_cv_unet/top5',
    'image_size': (256, 256),
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}

# Overlay 顏色設定（統一用於所有 overlay）
OVERLAY_ALPHA = 0.5
COLOR_TP = np.array([0, 255, 0])    # 綠色：正確預測天空
COLOR_FP = np.array([255, 0, 0])    # 紅色：誤判為天空
COLOR_FN = np.array([0, 0, 255])    # 藍色：漏掉的天空


# ============================================================
# Helper functions
# ============================================================
def create_overlay(image_np, pred_mask, gt_mask):
    """
    建立 overlay（統一顏色與透明度）
    - 綠色：TP（正確預測天空）
    - 紅色：FP（錯誤預測為天空）
    - 藍色：FN（漏掉的天空）
    """
    result = image_np.astype(np.float32).copy()
    
    tp = (pred_mask == 1) & (gt_mask == 1)
    fp = (pred_mask == 1) & (gt_mask == 0)
    fn = (pred_mask == 0) & (gt_mask == 1)
    
    result[tp] = result[tp] * (1 - OVERLAY_ALPHA) + COLOR_TP * OVERLAY_ALPHA
    result[fp] = result[fp] * (1 - OVERLAY_ALPHA) + COLOR_FP * OVERLAY_ALPHA
    result[fn] = result[fn] * (1 - OVERLAY_ALPHA) + COLOR_FN * OVERLAY_ALPHA
    
    return result.clip(0, 255).astype(np.uint8)


def create_gt_overlay(image_np, gt_mask):
    """
    建立 GT overlay（只顯示天空區域，用綠色）
    """
    result = image_np.astype(np.float32).copy()
    sky = (gt_mask == 1)
    result[sky] = result[sky] * (1 - OVERLAY_ALPHA) + COLOR_TP * OVERLAY_ALPHA
    return result.clip(0, 255).astype(np.uint8)


def compute_iou(pred, gt):
    """計算 IoU"""
    smooth = 1e-6
    intersection = (pred * gt).sum()
    union = pred.sum() + gt.sum() - intersection
    return (intersection + smooth) / (union + smooth)


def load_and_preprocess(image_path, mask_path, size):
    """載入並預處理圖片和 mask"""
    # 載入原圖
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
    return pred.astype(np.uint8)


def add_label_to_image(image_np, label, position='top'):
    """在圖片上加上文字標籤"""
    img = Image.fromarray(image_np)
    draw = ImageDraw.Draw(img)
    
    # 嘗試使用系統字型，如果失敗則用預設
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except:
        font = ImageFont.load_default()
    
    # 計算文字位置
    bbox = draw.textbbox((0, 0), label, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    if position == 'top':
        x = (img.width - text_width) // 2
        y = 5
    else:
        x = (img.width - text_width) // 2
        y = img.height - text_height - 5
    
    # 畫白色背景
    padding = 2
    draw.rectangle([x - padding, y - padding, x + text_width + padding, y + text_height + padding],
                   fill=(255, 255, 255))
    # 畫文字
    draw.text((x, y), label, fill=(0, 0, 0), font=font)
    
    return np.array(img)


def create_quad_grid(images, labels, spacing=5):
    """
    建立 2x2 四宮格圖片
    
    Args:
        images: list of 4 numpy arrays (RGB images)
        labels: list of 4 strings (labels for each image)
        spacing: 圖片間距
    
    Returns:
        combined: numpy array (RGB)
    """
    # 加上標籤
    labeled_images = []
    for img, label in zip(images, labels):
        labeled_img = add_label_to_image(img, label)
        labeled_images.append(labeled_img)
    
    h, w = labeled_images[0].shape[:2]
    
    # 建立空白畫布（白色背景）
    grid_h = h * 2 + spacing
    grid_w = w * 2 + spacing
    grid = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 255
    
    # 放置四張圖
    grid[0:h, 0:w] = labeled_images[0]                    # 左上
    grid[0:h, w+spacing:w*2+spacing] = labeled_images[1]  # 右上
    grid[h+spacing:h*2+spacing, 0:w] = labeled_images[2]  # 左下
    grid[h+spacing:h*2+spacing, w+spacing:w*2+spacing] = labeled_images[3]  # 右下
    
    return grid


# ============================================================
# 主程式
# ============================================================
def main():
    print('=' * 60)
    print('  Compare CV vs U-Net on Test Set')
    print('=' * 60)
    print()
    
    # 建立輸出目錄
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    os.makedirs(CONFIG['top5_dir'], exist_ok=True)
    
    # 載入 splits
    with open(CONFIG['splits_json'], 'r') as f:
        splits = json.load(f)
    
    test_samples = splits['test']
    print(f'Test samples: {[s["image"] for s in test_samples]}')
    print()
    
    # 載入 U-Net 模型
    print('[1] Loading U-Net model...')
    device = CONFIG['device']
    model = UNetSmall(in_channels=3, out_channels=1).to(device)
    checkpoint = torch.load(CONFIG['unet_model'], map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f'    Loaded from: {CONFIG["unet_model"]}')
    print(f'    Best epoch: {checkpoint["epoch"]}, Val IoU: {checkpoint["val_iou"]*100:.2f}%')
    print()
    
    # 處理每張圖
    print('[2] Processing test images...')
    print('-' * 75)
    print(f'  {"Image":>8} | {"U-Net IoU":>10} | {"CV IoU":>10} | {"Diff":>10} | {"Winner":>8}')
    print('-' * 75)
    
    results = []
    
    for sample in test_samples:
        image_name = sample['image']
        mask_name = sample['mask']
        base_name = image_name.replace('.jpg', '').replace('.png', '')
        
        image_path = os.path.join(CONFIG['images_dir'], image_name)
        mask_path = os.path.join(CONFIG['masks_dir'], mask_name)
        
        # 載入圖片
        image_np, gt_np = load_and_preprocess(image_path, mask_path, CONFIG['image_size'])
        
        # U-Net 預測
        pred_unet = predict_unet(model, image_np, device)
        iou_unet = compute_iou(pred_unet, gt_np)
        
        # CV 預測
        pred_cv = predict_cv(image_np)
        iou_cv = compute_iou(pred_cv, gt_np)
        
        # 建立 overlays
        gt_overlay = create_gt_overlay(image_np, gt_np)
        cv_overlay = create_overlay(image_np, pred_cv, gt_np)
        unet_overlay = create_overlay(image_np, pred_unet, gt_np)
        
        # 建立四宮格
        quad_images = [image_np, gt_overlay, cv_overlay, unet_overlay]
        quad_labels = [
            'Original',
            'GT (sky=green)',
            f'CV (IoU={iou_cv*100:.1f}%)',
            f'U-Net (IoU={iou_unet*100:.1f}%)'
        ]
        quad_grid = create_quad_grid(quad_images, quad_labels)
        
        # 儲存四宮格
        output_path = os.path.join(CONFIG['output_dir'], f'{base_name}_compare.png')
        Image.fromarray(quad_grid).save(output_path)
        
        # 計算改善
        diff = iou_unet - iou_cv
        winner = 'U-Net' if diff > 0 else 'CV' if diff < 0 else 'Tie'
        
        print(f'  {base_name:>8} | {iou_unet*100:>9.2f}% | {iou_cv*100:>9.2f}% | {diff*100:>+9.2f}% | {winner:>8}')
        
        results.append({
            'image': base_name,
            'iou_unet': iou_unet,
            'iou_cv': iou_cv,
            'diff': diff,
            'quad_grid': quad_grid,
        })
    
    print('-' * 75)
    
    # 統計
    avg_unet = np.mean([r['iou_unet'] for r in results])
    avg_cv = np.mean([r['iou_cv'] for r in results])
    
    print()
    print('[3] Summary:')
    print(f'    Average U-Net IoU: {avg_unet*100:.2f}%')
    print(f'    Average CV IoU:    {avg_cv*100:.2f}%')
    print(f'    Average Diff:      {(avg_unet - avg_cv)*100:+.2f}%')
    print()
    
    # 找出 U-Net 改善最大的前 5 張
    print('[4] Top 5 images with largest U-Net improvement:')
    sorted_results = sorted(results, key=lambda x: x['diff'], reverse=True)
    top5 = sorted_results[:5]
    
    print('-' * 50)
    for i, r in enumerate(top5, 1):
        print(f'    {i}. {r["image"]}: U-Net {r["iou_unet"]*100:.1f}% vs CV {r["iou_cv"]*100:.1f}% (diff: {r["diff"]*100:+.1f}%)')
        
        # 儲存到 top5 目錄
        top5_path = os.path.join(CONFIG['top5_dir'], f'top{i}_{r["image"]}_compare.png')
        Image.fromarray(r['quad_grid']).save(top5_path)
    
    print('-' * 50)
    print()
    
    # 輸出說明
    print('[5] Output files:')
    print(f'    All comparisons: {CONFIG["output_dir"]}/')
    print(f'    Top 5 improvements: {CONFIG["top5_dir"]}/')
    print()
    print('    Overlay colors:')
    print('      Green = TP (correct sky)')
    print('      Red   = FP (false positive)')
    print('      Blue  = FN (missed sky)')
    print()
    print('=' * 60)


if __name__ == '__main__':
    main()
