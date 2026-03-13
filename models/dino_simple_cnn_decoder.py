"""
DINO → DL 方案1：簡單 CNN Decoder

架構：
DINO features (37×37×768)
    ↓
[ Reshape: (37×37×768) → (batch, 768, 37, 37) ]
    ↓
Conv2d(768→256, kernel=1) → 37×37×256
    ↓
Upsample (×2) → 74×74×256
    ↓
Conv2d(256→128, kernel=3, padding=1) → 74×74×128
    ↓
Upsample (×2) → 148×148×128
    ↓
Conv2d(128→64, kernel=3, padding=1) → 148×148×64
    ↓
Upsample (×2) → 296×296×64
    ↓
Conv2d(64→1, kernel=3, padding=1) → 296×296×1
    ↓
Resize → 256×256×1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DINOSimpleCNNDecoder(nn.Module):
    """
    DINO + 簡單 CNN Decoder
    
    輸入：DINO patch features (batch, 37, 37, 768) 或 (batch, 1369, 768)
    輸出：Mask (batch, 1, 256, 256)
    """
    
    def __init__(self, dino_feat_dim=768, dino_patch_size=37, target_size=256):
        """
        參數:
            dino_feat_dim: DINO feature 維度（768 for ViT-B/14）
            dino_patch_size: DINO patch 空間尺寸（37×37 for 518×518 input）
            target_size: 目標輸出尺寸（256）
        """
        super(DINOSimpleCNNDecoder, self).__init__()
        self.dino_feat_dim = dino_feat_dim
        self.dino_patch_size = dino_patch_size
        self.target_size = target_size
        
        # 初始降維：768 → 256
        self.conv1 = nn.Conv2d(dino_feat_dim, 256, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu1 = nn.ReLU(inplace=True)
        
        # 上採樣階段 1: 37×37 → 74×74
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(inplace=True)
        
        # 上採樣階段 2: 74×74 → 148×148
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU(inplace=True)
        
        # 上採樣階段 3: 148×148 → 296×296
        self.upsample3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv4 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, dino_features):
        """
        前向傳播
        
        參數:
            dino_features: DINO patch tokens
                - 如果是 (batch, N, D) 格式，N = 37*37 = 1369，D = 768
                - 需要 reshape 成 (batch, 768, 37, 37)
        
        返回:
            mask: (batch, 1, 256, 256)
        """
        # 處理輸入格式
        if dino_features.dim() == 3:
            # (batch, N, D) -> (batch, 37, 37, D) -> (batch, D, 37, 37)
            batch_size = dino_features.shape[0]
            dino_features = dino_features.reshape(batch_size, self.dino_patch_size, self.dino_patch_size, self.dino_feat_dim)
            dino_features = dino_features.permute(0, 3, 1, 2)  # (batch, 768, 37, 37)
        
        # 初始降維: (batch, 768, 37, 37) -> (batch, 256, 37, 37)
        x = self.conv1(dino_features)
        x = self.bn1(x)
        x = self.relu1(x)
        
        # 上採樣階段 1: (batch, 256, 37, 37) -> (batch, 128, 74, 74)
        x = self.upsample1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        
        # 上採樣階段 2: (batch, 128, 74, 74) -> (batch, 64, 148, 148)
        x = self.upsample2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        
        # 上採樣階段 3: (batch, 64, 148, 148) -> (batch, 1, 296, 296)
        x = self.upsample3(x)
        x = self.conv4(x)
        mask = self.sigmoid(x)  # (batch, 1, 296, 296)
        
        # Resize 到目標尺寸: 296×296 -> 256×256
        if mask.shape[2] != self.target_size or mask.shape[3] != self.target_size:
            mask = F.interpolate(
                mask, 
                size=(self.target_size, self.target_size), 
                mode='bilinear', 
                align_corners=False
            )
        
        return mask  # (batch, 1, 256, 256)


class DINOWithSimpleCNNDecoder(nn.Module):
    """
    完整的 DINO + 簡單 CNN Decoder 模型
    
    包含 DINO feature extractor（frozen）和簡單 CNN decoder（trainable）
    """
    
    def __init__(self, dino_model=None, device='cuda', target_size=256):
        """
        參數:
            dino_model: DINOv2 模型（如果 None，會自動載入）
            device: 設備
            target_size: 目標輸出尺寸（256）
        """
        super(DINOWithSimpleCNNDecoder, self).__init__()
        self.device = device
        self.target_size = target_size
        
        # 載入 DINO model（frozen）
        if dino_model is None:
            import torch.hub
            self.dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14', trust_repo=True)
            self.dino_model = self.dino_model.to(device)
        else:
            self.dino_model = dino_model
        
        # 凍結 DINO
        for param in self.dino_model.parameters():
            param.requires_grad = False
        self.dino_model.eval()
        
        # 簡單 CNN Decoder（trainable）
        self.decoder = DINOSimpleCNNDecoder(
            dino_feat_dim=768,
            dino_patch_size=37,
            target_size=target_size
        ).to(device)
    
    def get_dino_features(self, image_tensor):
        """
        從影像取得 DINO features
        
        參數:
            image_tensor: (batch, 3, H, W)，值域 [0, 1]，已 normalize（ImageNet mean/std）
        
        返回:
            patch_tokens: (batch, N, D)，N = 37*37 = 1369，D = 768
        """
        # DINO 需要 518×518 輸入
        if image_tensor.shape[2] != 518 or image_tensor.shape[3] != 518:
            image_tensor = F.interpolate(
                image_tensor, 
                size=(518, 518), 
                mode='bilinear', 
                align_corners=False
            )
        
        with torch.no_grad():
            out = self.dino_model.forward_features(image_tensor)
        
        # 提取 patch tokens（排除 CLS token）
        if isinstance(out, dict):
            tokens = out.get("x_norm_patchtokens")
            if tokens is None:
                tokens = out.get("x_prenorm")
                if tokens is not None:
                    tokens = tokens[:, 1:, :]  # 排除 CLS token
        else:
            tokens = out[:, 1:, :] if out.dim() == 3 else out
        
        return tokens  # (batch, 1369, 768)
    
    def forward(self, image_tensor):
        """
        前向傳播
        
        參數:
            image_tensor: (batch, 3, 256, 256)，值域 [0, 1]，需要 normalize
        
        返回:
            mask: (batch, 1, 256, 256)
        """
        # 取得 DINO features
        dino_features = self.get_dino_features(image_tensor)  # (batch, 1369, 768)
        
        # 通過 decoder
        mask = self.decoder(dino_features)  # (batch, 1, 256, 256)
        
        return mask
