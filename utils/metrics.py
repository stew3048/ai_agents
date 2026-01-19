"""
評估指標計算
用於天空分割任務的評估指標
"""

import torch
import torch.nn.functional as F


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
