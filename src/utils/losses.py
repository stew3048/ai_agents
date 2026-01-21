"""
Loss Functions for Binary Segmentation

包含：
1. bce_with_logits_loss: 標準的 Binary Cross Entropy（帶 logits）
2. dice_loss: Dice Loss（關注整體重疊）
3. combined_loss: BCE + Dice（兩個角度都顧到）

為什麼用這些 loss？
- BCE: 每個像素獨立計算，是最基本的分類 loss
- Dice: 看預測區域和真實區域的重疊，對類別不平衡更穩健
- Combined: 結合兩者優點，通常效果更好
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def bce_with_logits_loss(logits, targets):
    """
    Binary Cross Entropy with Logits
    
    為什麼用 WithLogits 版本？
    - 數值穩定：內部用 log-sum-exp 技巧，避免 log(0) 的問題
    - 效率：不需要先做 sigmoid 再做 log
    
    Args:
        logits: 模型輸出，shape (B, 1, H, W)，未經 sigmoid
        targets: Ground truth，shape (B, 1, H, W)，值為 0 或 1
    
    Returns:
        loss: 標量
    """
    return F.binary_cross_entropy_with_logits(logits, targets)


def dice_loss(logits, targets, smooth=1e-6):
    """
    Dice Loss
    
    Dice 係數 = 2 * |A ∩ B| / (|A| + |B|)
    Dice Loss = 1 - Dice 係數
    
    為什麼用 Dice？
    - 直接優化 IoU/Dice 指標
    - 對類別不平衡更穩健（不會被多數類別主導）
    
    Args:
        logits: 模型輸出，shape (B, 1, H, W)，未經 sigmoid
        targets: Ground truth，shape (B, 1, H, W)，值為 0 或 1
        smooth: 平滑項，避免除以零
    
    Returns:
        loss: 標量
    """
    # 轉為機率
    probs = torch.sigmoid(logits)
    
    # Flatten（保留 batch 維度）
    probs_flat = probs.view(probs.size(0), -1)
    targets_flat = targets.view(targets.size(0), -1)
    
    # 計算 intersection 和 union
    intersection = (probs_flat * targets_flat).sum(dim=1)
    union = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)
    
    # Dice 係數（每個 batch 獨立計算）
    dice = (2. * intersection + smooth) / (union + smooth)
    
    # Dice Loss = 1 - Dice（取 batch 平均）
    return 1 - dice.mean()


def combined_loss(logits, targets, bce_weight=0.5, dice_weight=0.5):
    """
    Combined Loss = BCE + Dice
    
    為什麼要結合？
    - BCE: 每個像素獨立看，提供穩定的梯度
    - Dice: 整體區域看，關注形狀和重疊
    - 結合兩者，通常比單獨使用效果更好
    
    Args:
        logits: 模型輸出，shape (B, 1, H, W)，未經 sigmoid
        targets: Ground truth，shape (B, 1, H, W)，值為 0 或 1
        bce_weight: BCE loss 的權重
        dice_weight: Dice loss 的權重
    
    Returns:
        loss: 標量
    """
    bce = bce_with_logits_loss(logits, targets)
    dice = dice_loss(logits, targets)
    
    return bce_weight * bce + dice_weight * dice


class BCEWithLogitsLoss(nn.Module):
    """
    BCEWithLogitsLoss 的 Module 版本（方便放入 nn.Sequential 或存檔）
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, logits, targets):
        return bce_with_logits_loss(logits, targets)


class DiceLoss(nn.Module):
    """
    DiceLoss 的 Module 版本
    """
    
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, logits, targets):
        return dice_loss(logits, targets, self.smooth)


class CombinedLoss(nn.Module):
    """
    CombinedLoss (BCE + Dice) 的 Module 版本
    """
    
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
    
    def forward(self, logits, targets):
        return combined_loss(logits, targets, self.bce_weight, self.dice_weight)
