"""
基於 DINOv2 的天空分割模型（多層解碼頭）

- Backbone: torch.hub DINOv2（參數凍結）
- Decoder Head:
  - Fusion: 1x1 Conv (embed_dim -> 256), BatchNorm, ReLU
  - Prediction: 1x1 Conv (256 -> 1)，輸出 Logit
- 輸出: (B, 1, H, W) Logit（不帶 Sigmoid，與訓練 loss BCEWithLogitsLoss 一致）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# DINOv2 ViT-S/14: patch_size=14, embed_dim=384
# 若改用 ViT-G 等則 embed_dim=1536，decoder 第一層輸入維度會自動對應
DINOV2_VITS14_EMBED_DIM = 384
DINOV2_PATCH_SIZE = 14
DECODER_FUSION_DIM = 256


def _load_dinov2_backbone(arch_name='dinov2_vits14'):
    """從 torch.hub 載入 DINOv2 backbone 並凍結參數。"""
    model = torch.hub.load('facebookresearch/dinov2', arch_name, pretrained=True)
    for p in model.parameters():
        p.requires_grad = False
    return model


class SkySegModel(nn.Module):
    """
    基於 DINOv2 的天空分割模型。
    
    - Backbone: DINOv2（凍結）
    - Decoder Head: Fusion (1x1 Conv + BN + ReLU) -> Prediction (1x1 Conv) -> Logit
    - 輸出: (B, 1, H, W) Logit（推論時需自行 sigmoid）
    """

    def __init__(self, backbone=None, embed_dim=DINOV2_VITS14_EMBED_DIM, patch_size=DINOV2_PATCH_SIZE, fusion_dim=DECODER_FUSION_DIM):
        """
        參數:
            backbone: 已載入且凍結的 DINOv2 模型；若為 None 則自動載入 dinov2_vits14
            embed_dim: backbone 特徵維度（ViT-S/14=384，ViT-G=1536 等）
            patch_size: patch 大小（14）
            fusion_dim: 解碼頭中間維度（256）
        """
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        if backbone is None:
            self.backbone = _load_dinov2_backbone('dinov2_vits14')
        else:
            self.backbone = backbone

        # 多層解碼頭：只訓練這兩層，Backbone 已凍結
        # 第一層 (Fusion): 1x1 卷積 embed_dim -> 256, BatchNorm, ReLU
        self.fusion = nn.Sequential(
            nn.Conv2d(embed_dim, fusion_dim, kernel_size=1),
            nn.BatchNorm2d(fusion_dim),
            nn.ReLU(inplace=True),
        )
        # 第二層 (Prediction): 1x1 卷積 256 -> 1，輸出 Logit
        self.prediction = nn.Conv2d(fusion_dim, 1, kernel_size=1)

    def forward(self, images):
        """
        前向傳播。
        
        參數:
            images: (B, 3, H, W)，RGB 影像。若 H/W 非 14 的倍數會先縮放到 14 的倍數再推理，輸出仍為 (B, 1, H, W)。
        
        返回:
            logits: (B, 1, H, W)，未經 Sigmoid 的 logit，可直接接 BCEWithLogitsLoss；推論時需自行 torch.sigmoid(logits)。
        """
        B, _, H, W = images.shape
        # DINOv2 patch_size=14，輸入須為 14 的倍數
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

        # Backbone: forward_features 回傳 dict，含 "x_norm_patchtokens" (B, N, C)
        with torch.no_grad():
            features = self.backbone.forward_features(images)
        if isinstance(features, dict):
            patch_tokens = features["x_norm_patchtokens"]  # (B, N, C)
        else:
            # 若為 tensor，(B, 1+N, C) -> 去掉 CLS
            patch_tokens = features[:, 1:, :]

        # 還原空間維度: h = H2/14, w = W2/14
        h = H2 // self.patch_size
        w = W2 // self.patch_size
        patch_tokens = patch_tokens.transpose(1, 2).reshape(B, self.embed_dim, h, w)  # (B, C, h, w)

        # Decoder Head: Fusion -> Prediction -> Logit（不帶 Sigmoid）
        x = self.fusion(patch_tokens)   # (B, 256, h, w)
        logits = self.prediction(x)     # (B, 1, h, w)

        # 上採樣回「目前」影像解析度 (B, 1, H2, W2)，再依需要插回原始 H,W
        logits = F.interpolate(
            logits,
            size=(H2, W2),
            mode='bilinear',
            align_corners=False
        )
        if orig_H != H2 or orig_W != W2:
            logits = F.interpolate(logits, size=(orig_H, orig_W), mode='bilinear', align_corners=False)
        return logits


def create_sky_seg_dinov2_linear(backbone=None, **kwargs):
    """工廠函數：建立 SkySegModel。"""
    return SkySegModel(backbone=backbone, **kwargs)
