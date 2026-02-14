"""
DINO → DL 方案6：混合方案（DINO + 原圖細節）

架構：
[ 影像 256×256 ]
    ↓
    ├─→ DINO → Patch features (37×37×768)  [ 全局語義 ]
    │
    └─→ 淺層 CNN → Features (256×256×64)  [ 像素細節 ]
         ↓
    [ DINO features 上採樣到 256×256×256 ]
         ↓
    [ Concat(DINO features, CNN features) ] → 256×256×320
         ↓
    [ 簡單 CNN Decoder ]
         ├─→ Conv(320→128) → 256×256×128
         ├─→ Conv(128→64) → 256×256×64
         └─→ Conv(64→1) → 256×256×1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DINOHybridDecoder(nn.Module):
    """
    DINO + 混合方案 Decoder
    
    結合 DINO 全局語義特徵和原圖的淺層 CNN 細節特徵
    """
    
    def __init__(self, dino_feat_dim=768, dino_patch_size=37, target_size=256):
        """
        參數:
            dino_feat_dim: DINO feature 維度（768 for ViT-B/14）
            dino_patch_size: DINO patch 空間尺寸（37×37 for 518×518 input）
            target_size: 目標輸出尺寸（256）
        """
        super(DINOHybridDecoder, self).__init__()
        self.dino_feat_dim = dino_feat_dim
        self.dino_patch_size = dino_patch_size
        self.target_size = target_size
        
        # 淺層 CNN：從原圖提取像素細節特徵
        self.shallow_cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )  # 輸出: (batch, 64, 256, 256)
        
        # DINO features 處理：降維並上採樣到 256×256
        self.dino_proj = nn.Sequential(
            nn.Linear(dino_feat_dim, 256),
            nn.LayerNorm(256),
            nn.GELU()
        )  # 輸出: (batch, 37*37, 256) -> reshape 成 (batch, 37, 37, 256)
        
        # 融合層：Concat 後通過簡單 CNN decoder
        self.fusion_decoder = nn.Sequential(
            # 320 = 256 (DINO) + 64 (CNN)
            nn.Conv2d(320, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, dino_features, original_image):
        """
        前向傳播
        
        參數:
            dino_features: DINO patch tokens
                - 如果是 (batch, N, D) 格式，N = 37*37 = 1369，D = 768
            original_image: 原始影像 (batch, 3, 256, 256)，值域 [0, 1]
        
        返回:
            mask: (batch, 1, 256, 256)
        """
        batch_size = original_image.shape[0]
        
        # 處理 DINO features
        if dino_features.dim() == 3:
            # (batch, N, D) -> (batch, N, 256)
            dino_proj = self.dino_proj(dino_features)  # (batch, 1369, 256)
            # Reshape 成空間格式: (batch, 37, 37, 256)
            dino_proj = dino_proj.view(batch_size, self.dino_patch_size, self.dino_patch_size, 256)
            # 轉換為 (batch, 256, 37, 37)
            dino_proj = dino_proj.permute(0, 3, 1, 2)
        else:
            # 如果已經是 (batch, 37, 37, 768) 格式
            dino_proj = dino_features.view(batch_size, self.dino_patch_size, self.dino_patch_size, self.dino_feat_dim)
            dino_proj = dino_proj.permute(0, 3, 1, 2)  # (batch, 768, 37, 37)
            # 需要先降維
            dino_proj = dino_proj.view(batch_size, self.dino_feat_dim, -1).permute(0, 2, 1)  # (batch, 1369, 768)
            dino_proj = self.dino_proj(dino_proj)  # (batch, 1369, 256)
            dino_proj = dino_proj.view(batch_size, self.dino_patch_size, self.dino_patch_size, 256)
            dino_proj = dino_proj.permute(0, 3, 1, 2)  # (batch, 256, 37, 37)
        
        # 上採樣 DINO features 到 256×256
        dino_up = F.interpolate(
            dino_proj, 
            size=(self.target_size, self.target_size), 
            mode='bilinear', 
            align_corners=False
        )  # (batch, 256, 256, 256)
        
        # 從原圖提取淺層 CNN 特徵
        cnn_features = self.shallow_cnn(original_image)  # (batch, 64, 256, 256)
        
        # Concat: (batch, 256, 256, 256) + (batch, 64, 256, 256) -> (batch, 320, 256, 256)
        fused = torch.cat([dino_up, cnn_features], dim=1)  # (batch, 320, 256, 256)
        
        # 通過融合 decoder
        mask = self.fusion_decoder(fused)  # (batch, 1, 256, 256)
        
        return mask


class DINOWithHybridDecoder(nn.Module):
    """
    完整的 DINO + 混合方案 Decoder 模型
    
    包含 DINO feature extractor（frozen）和混合 decoder（trainable）
    """
    
    def __init__(self, dino_model=None, device='cuda', target_size=256):
        """
        參數:
            dino_model: DINOv2 模型（如果 None，會自動載入）
            device: 設備
            target_size: 目標輸出尺寸（256）
        """
        super(DINOWithHybridDecoder, self).__init__()
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
        
        # 混合 Decoder（trainable）
        self.decoder = DINOHybridDecoder(
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
        # 保存原始影像（用於淺層 CNN）
        # 注意：image_tensor 可能已經 normalize，需要 denormalize 或直接使用
        # 為了簡化，我們假設 image_tensor 是 [0, 1] 範圍（未 normalize）
        # 如果需要 normalize，應該在外部處理
        
        # 取得 DINO features（需要 normalize 的版本）
        # 為了 DINO，我們需要 normalize
        from torchvision import transforms
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        image_normalized = normalize(image_tensor)
        
        dino_features = self.get_dino_features(image_normalized)  # (batch, 1369, 768)
        
        # 通過混合 decoder（使用原始 image_tensor 作為淺層 CNN 輸入）
        mask = self.decoder(dino_features, image_tensor)  # (batch, 1, 256, 256)
        
        return mask
