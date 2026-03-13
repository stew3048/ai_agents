"""
評估指標計算
用於天空分割任務的評估指標
與 train_in_domain / eval_unet 同一套公式（IoU, Dice, FP rate, FN rate）
"""

import torch
import torch.nn.functional as F
import numpy as np


def calculate_iou(pred, target, threshold=0.5):
    """
    計算 Intersection over Union (IoU)
    
    參數:
        pred: 預測 logits [B, 1, H, W] 或機率 [B, 1, H, W]
        target: 真實 mask [B, 1, H, W]，值為 0 或 1
        threshold: 二進制化閾值，預設 0.5
    
    返回:
        iou: IoU 分數（標量）
    """
    # 如果 pred 是 logits，先轉換為機率
    if pred.max() > 1.0 or pred.min() < 0.0:
        pred = torch.sigmoid(pred)
    
    # 二進制化
    pred_binary = (pred > threshold).float()
    target_binary = (target > 0.5).float()
    
    # 計算交集和聯集
    intersection = (pred_binary * target_binary).sum()
    union = pred_binary.sum() + target_binary.sum() - intersection
    
    # 避免除以零
    if union == 0:
        return torch.tensor(1.0, device=pred.device)
    
    iou = intersection / union
    return iou


def calculate_dice_score(pred, target, threshold=0.5, smooth=1e-6):
    """
    計算 Dice Score (F1 Score)
    
    參數:
        pred: 預測 logits [B, 1, H, W] 或機率 [B, 1, H, W]
        target: 真實 mask [B, 1, H, W]，值為 0 或 1
        threshold: 二進制化閾值，預設 0.5
        smooth: 平滑項，避免除以零，預設 1e-6
    
    返回:
        dice: Dice 分數（標量）
    """
    # 如果 pred 是 logits，先轉換為機率
    if pred.max() > 1.0 or pred.min() < 0.0:
        pred = torch.sigmoid(pred)
    
    # 二進制化
    pred_binary = (pred > threshold).float()
    target_binary = (target > 0.5).float()
    
    # 計算交集和總和
    intersection = (pred_binary * target_binary).sum()
    dice = (2.0 * intersection + smooth) / (pred_binary.sum() + target_binary.sum() + smooth)
    
    return dice


def calculate_pixel_accuracy(pred, target, threshold=0.5):
    """
    計算像素準確率
    
    參數:
        pred: 預測 logits [B, 1, H, W] 或機率 [B, 1, H, W]
        target: 真實 mask [B, 1, H, W]，值為 0 或 1
        threshold: 二進制化閾值，預設 0.5
    
    返回:
        accuracy: 準確率（標量）
    """
    # 如果 pred 是 logits，先轉換為機率
    if pred.max() > 1.0 or pred.min() < 0.0:
        pred = torch.sigmoid(pred)
    
    # 二進制化
    pred_binary = (pred > threshold).float()
    target_binary = (target > 0.5).float()
    
    # 計算正確預測的像素數
    correct = (pred_binary == target_binary).float()
    accuracy = correct.mean()
    
    return accuracy


def calculate_metrics(pred, target, threshold=0.5):
    """
    計算所有評估指標
    
    參數:
        pred: 預測 logits [B, 1, H, W]
        target: 真實 mask [B, 1, H, W]
        threshold: 二進制化閾值，預設 0.5
    
    返回:
        metrics: 字典，包含所有指標
    """
    metrics = {
        'iou': calculate_iou(pred, target, threshold).item(),
        'dice': calculate_dice_score(pred, target, threshold).item(),
        'pixel_acc': calculate_pixel_accuracy(pred, target, threshold).item()
    }
    return metrics


def compute_metrics_numpy(pred_mask, gt_mask, smooth=1e-6):
    """
    Numpy 版：從 pred (H,W) 與 gt (H,W) 0~1 計算 iou, dice, fp_rate, fn_rate。
    與 train_in_domain / eval_unet 同一套定義，供 CLIPSeg、Grounding 等 mask 輸出方法使用。
    
    參數:
        pred_mask: 預測 mask，numpy (H,W)，0~1 或 0~255
        gt_mask: GT mask，numpy (H,W)，0~1 或 0~255
        smooth: 平滑項
    
    返回:
        (iou, dice, fp_rate, fn_rate) 四個 float
    """
    pred_b = (np.asarray(pred_mask, dtype=np.float32).squeeze() > 0.5).astype(np.float32)
    gt_b = (np.asarray(gt_mask, dtype=np.float32).squeeze() > 0.5).astype(np.float32)
    if pred_b.shape != gt_b.shape:
        from PIL import Image
        pred_b = np.array(
            Image.fromarray((pred_b * 255).astype(np.uint8)).resize(
                (gt_b.shape[1], gt_b.shape[0]), Image.NEAREST
            ), dtype=np.float32
        ) / 255.0
        pred_b = (pred_b > 0.5).astype(np.float32)
    tp = ((pred_b == 1) & (gt_b == 1)).sum()
    fp = ((pred_b == 1) & (gt_b == 0)).sum()
    fn = ((pred_b == 0) & (gt_b == 1)).sum()
    tn = ((pred_b == 0) & (gt_b == 0)).sum()
    union = tp + fp + fn
    iou = float((tp / (union + smooth)) if union > 0 else 1.0)
    dice = float((2 * tp + smooth) / (2 * tp + fp + fn + smooth)) if (tp + fp + fn) > 0 else 1.0
    total_neg = tn + fp
    total_pos = tp + fn
    fp_rate = float(fp / (total_neg + smooth))
    fn_rate = float(fn / (total_pos + smooth)) if total_pos > 0 else 0.0
    return iou, dice, fp_rate, fn_rate
