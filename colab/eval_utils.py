"""
評估工具函數（Colab 專屬版本）
包含 overlay 生成等共用功能
"""

import os
import numpy as np
from PIL import Image


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


def save_overlay(overlay_img, output_path):
    """保存 overlay 影像"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    overlay_img.save(output_path)
