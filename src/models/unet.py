"""
U-Net for Binary Segmentation

架構：
- Encoder: 逐步縮小（下採樣），提取特徵
- Decoder: 逐步放大（上採樣），恢復解析度
- Skip connections: 把 encoder 的細節資訊傳給 decoder

Input: (B, 3, H, W) - RGB 圖片
Output: (B, 1, H, W) - logits（未經 sigmoid）
"""

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """
    U-Net 的基本單元：兩次 Conv + BatchNorm + ReLU
    
    (Conv -> BN -> ReLU) x 2
    """
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """
    Encoder 的下採樣單元：MaxPool + DoubleConv
    
    解析度減半，通道數增加
    """
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """
    Decoder 的上採樣單元：UpConv + Concat + DoubleConv
    
    解析度加倍，接收 skip connection
    """
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 上採樣（用轉置卷積）
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        # 合併後的卷積
        self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        """
        x1: 來自 decoder 的特徵（較小）
        x2: 來自 encoder 的 skip connection（較大）
        """
        x1 = self.up(x1)
        
        # 處理尺寸不匹配的情況（如果輸入不是 2 的倍數）
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        
        if diff_x > 0 or diff_y > 0:
            x1 = nn.functional.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                                        diff_y // 2, diff_y - diff_y // 2])
        
        # Concatenate skip connection
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net for Binary Segmentation
    
    Args:
        in_channels: 輸入通道數（RGB = 3）
        out_channels: 輸出通道數（binary = 1）
        features: 第一層的特徵數，後續每層翻倍
    
    Input: (B, 3, H, W)
    Output: (B, 1, H, W) - logits（需要 sigmoid 轉為機率）
    """
    
    def __init__(self, in_channels=3, out_channels=1, features=64):
        super().__init__()
        
        # Encoder (下採樣路徑)
        self.inc = DoubleConv(in_channels, features)           # 3 -> 64
        self.down1 = Down(features, features * 2)              # 64 -> 128
        self.down2 = Down(features * 2, features * 4)          # 128 -> 256
        self.down3 = Down(features * 4, features * 8)          # 256 -> 512
        self.down4 = Down(features * 8, features * 16)         # 512 -> 1024 (bottleneck)
        
        # Decoder (上採樣路徑)
        self.up1 = Up(features * 16, features * 8)             # 1024 -> 512
        self.up2 = Up(features * 8, features * 4)              # 512 -> 256
        self.up3 = Up(features * 4, features * 2)              # 256 -> 128
        self.up4 = Up(features * 2, features)                  # 128 -> 64
        
        # 最後的 1x1 卷積，輸出 logits
        self.outc = nn.Conv2d(features, out_channels, kernel_size=1)
    
    def forward(self, x):
        # Encoder
        x1 = self.inc(x)      # skip connection 1
        x2 = self.down1(x1)   # skip connection 2
        x3 = self.down2(x2)   # skip connection 3
        x4 = self.down3(x3)   # skip connection 4
        x5 = self.down4(x4)   # bottleneck
        
        # Decoder (使用 skip connections)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # 輸出 logits
        logits = self.outc(x)
        return logits


class UNetSmall(nn.Module):
    """
    小型 U-Net（適合小資料集或快速測試）
    
    比標準 U-Net 少一層，參數量約為 1/4
    
    Input: (B, 3, H, W)
    Output: (B, 1, H, W) - logits
    """
    
    def __init__(self, in_channels=3, out_channels=1, features=32):
        super().__init__()
        
        # Encoder (3 層而非 4 層)
        self.inc = DoubleConv(in_channels, features)           # 3 -> 32
        self.down1 = Down(features, features * 2)              # 32 -> 64
        self.down2 = Down(features * 2, features * 4)          # 64 -> 128
        self.down3 = Down(features * 4, features * 8)          # 128 -> 256 (bottleneck)
        
        # Decoder
        self.up1 = Up(features * 8, features * 4)              # 256 -> 128
        self.up2 = Up(features * 4, features * 2)              # 128 -> 64
        self.up3 = Up(features * 2, features)                  # 64 -> 32
        
        # 輸出
        self.outc = nn.Conv2d(features, out_channels, kernel_size=1)
    
    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        # Decoder
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        
        # 輸出 logits
        logits = self.outc(x)
        return logits


def get_unet(model_type='small', in_channels=3, out_channels=1):
    """
    建立 U-Net 模型
    
    Args:
        model_type: 'small' 或 'standard'
        in_channels: 輸入通道數
        out_channels: 輸出通道數
    
    Returns:
        U-Net 模型
    """
    if model_type == 'small':
        return UNetSmall(in_channels, out_channels, features=32)
    elif model_type == 'standard':
        return UNet(in_channels, out_channels, features=64)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
