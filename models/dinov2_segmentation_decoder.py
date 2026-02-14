"""
DINOv2 多層特徵 + 多層卷積解碼器（分割用）

- Backbone: DINOv2（凍結），提取最後四層 patch tokens 並 Concatenate → 1536 維（ViT-S 每層 384）
- Decoder: conv_block 組成 4 個 Block → 輸出 Logit
- 訓練：僅訓練解碼器，Backbone 凍結
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

DINOV2_PATCH_SIZE = 14
# ViT-S/14 最後四層 concat：4 * 384 = 1536
DINOV2_LAST4_CONCAT_DIM = 1536


def conv_block(in_channels, out_channels, kernel_size):
    """
    輔助函數：Conv2d + BatchNorm2d + ReLU。
    kernel_size=3 時預設 padding=1 以保持空間尺寸。
    """
    if kernel_size == 1:
        padding = 0
    elif kernel_size == 3:
        padding = 1
    else:
        padding = kernel_size // 2
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


def _load_dinov2_backbone(arch_name='dinov2_vits14'):
    """從 torch.hub 載入 DINOv2 backbone 並凍結參數。"""
    model = torch.hub.load('facebookresearch/dinov2', arch_name, pretrained=True)
    for p in model.parameters():
        p.requires_grad = False
    return model


class DINOv2SegmentationModel(nn.Module):
    """
    多層特徵 + 多層卷積解碼器分割模型。
    
    - Feature Fusion: 提取 DINOv2 最後四層特徵並 Concatenate（1536 維）
    - 1D tokens (B, N, C) 依圖片比例 reshape 成 2D feature map (B, C, H/14, W/14)
    - Decoder: Block1 1x1 1536→512, Block2 3x3 512→512, Block3 3x3 512→256, Block4 1x1 256→1（Logit）
    - 最後 nn.Upsample bilinear 還原至原始圖片尺寸
    - Backbone 凍結，僅訓練解碼器
    """

    def __init__(self, backbone=None, patch_size=DINOV2_PATCH_SIZE):
        super().__init__()
        self.patch_size = patch_size

        if backbone is None:
            self.backbone = _load_dinov2_backbone('dinov2_vits14')
        else:
            self.backbone = backbone
            for p in self.backbone.parameters():
                p.requires_grad = False

        # 解碼器：僅此部分可訓練
        self.decoder = nn.Sequential(
            conv_block(1536, 512, kernel_size=1),   # Block 1: 1x1, 1536 → 512
            conv_block(512, 512, kernel_size=3),    # Block 2: 3x3, 512 → 512
            conv_block(512, 256, kernel_size=3),    # Block 3: 3x3, 512 → 256
            nn.Conv2d(256, 1, kernel_size=1),      # Block 4: 1x1, 256 → 1 (Output Logit)
        )

    def forward(self, images):
        """
        images: (B, 3, H, W)，若 H/W 非 14 的倍數會先縮放到 14 的倍數。
        返回: logits (B, 1, H, W)，未經 Sigmoid。
        """
        B, _, H, W = images.shape
        H2 = (H // self.patch_size) * self.patch_size
        W2 = (W // self.patch_size) * self.patch_size
        if H2 == 0:
            H2 = self.patch_size
        if W2 == 0:
            W2 = self.patch_size
        if H != H2 or W != W2:
            images = F.interpolate(images, size=(H2, W2), mode='bilinear', align_corners=False)
            orig_H, orig_W = H, W
        else:
            orig_H, orig_W = H, W
            H2, W2 = H, W

        h = H2 // self.patch_size
        w = W2 // self.patch_size

        # 提取最後四層 patch tokens（Backbone 凍結）
        with torch.no_grad():
            # get_intermediate_layers(x, n=4) 回傳 4 個 (B, N, C)，C=384
            outputs = self.backbone.get_intermediate_layers(images, n=4, norm=True)
        # 每個 output 為 (B, N, 384)，N = h*w
        # Concatenate on channel dim → (B, N, 1536)
        patch_cat = torch.cat(outputs, dim=-1)

        # Reshape: (B, N, 1536) → (B, 1536, h, w)
        patch_cat = patch_cat.transpose(1, 2).reshape(B, DINOV2_LAST4_CONCAT_DIM, h, w)

        # 解碼器
        logits = self.decoder(patch_cat)

        # 還原至 (H2, W2)：先 upsample 14 倍到 patch 對應的解析度（decoder 輸出為 h,w）
        logits = F.interpolate(logits, size=(H2, W2), mode='bilinear', align_corners=False)

        if orig_H != H2 or orig_W != W2:
            logits = F.interpolate(logits, size=(orig_H, orig_W), mode='bilinear', align_corners=False)
        return logits


def create_dinov2_segmentation_model(backbone=None, **kwargs):
    """工廠函數：建立 DINOv2SegmentationModel。"""
    return DINOv2SegmentationModel(backbone=backbone, **kwargs)
