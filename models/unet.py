"""
U-Net 模型定義
用於天空分割任務的語義分割模型

U-Net 架構特點：
- Encoder-Decoder 結構
- Skip connections（跳躍連接）保留細節資訊
- 對稱的編碼器與解碼器
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    U-Net 中的雙層卷積模組
    每個模組包含兩次 Conv2d + BatchNorm + ReLU 的組合
    
    參數:
        in_channels: 輸入通道數
        out_channels: 輸出通道數
    """
    
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        
        # 第一層卷積：3x3 卷積 + Batch Normalization + ReLU
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 
            kernel_size=3, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        
        # 第二層卷積：3x3 卷積 + Batch Normalization + ReLU
        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
    
    def forward(self, x):
        """
        前向傳播
        
        參數:
            x: 輸入張量 [B, C, H, W]
        
        返回:
            輸出張量 [B, out_channels, H, W]
        """
        # 第一層：卷積 -> BatchNorm -> ReLU
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        # 第二層：卷積 -> BatchNorm -> ReLU
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        
        return x


class UNet(nn.Module):
    """
    U-Net 模型主體
    
    架構說明：
    Encoder (下採樣路徑):
        - 輸入: [B, 3, 256, 256]
        - 經過 4 個下採樣階段，每個階段通道數翻倍，尺寸減半
        - 最終: [B, 512, 16, 16]
    
    Decoder (上採樣路徑):
        - 從 [B, 512, 16, 16] 開始
        - 經過 4 個上採樣階段，每個階段通道數減半，尺寸翻倍
        - 使用 skip connections 連接 encoder 的特徵圖
        - 最終輸出: [B, 1, 256, 256]
    
    參數:
        n_channels: 輸入圖片通道數（RGB 為 3）
        n_classes: 輸出類別數（二進制分割為 1）
    """
    
    def __init__(self, n_channels=3, n_classes=1):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        
        # ========== Encoder 路徑（下採樣） ==========
        
        # 第一層：輸入 [B, 3, 256, 256] -> [B, 64, 256, 256]
        self.inc = DoubleConv(n_channels, 64)
        
        # 第二層：下採樣 [B, 64, 256, 256] -> [B, 128, 128, 128]
        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),  # 最大池化，尺寸減半: 256 -> 128
            DoubleConv(64, 128)
        )
        
        # 第三層：下採樣 [B, 128, 128, 128] -> [B, 256, 64, 64]
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),  # 最大池化，尺寸減半: 128 -> 64
            DoubleConv(128, 256)
        )
        
        # 第四層：下採樣 [B, 256, 64, 64] -> [B, 512, 32, 32]
        self.down3 = nn.Sequential(
            nn.MaxPool2d(2),  # 最大池化，尺寸減半: 64 -> 32
            DoubleConv(256, 512)
        )
        
        # 第五層（最底層）：下採樣 [B, 512, 32, 32] -> [B, 1024, 16, 16]
        self.down4 = nn.Sequential(
            nn.MaxPool2d(2),  # 最大池化，尺寸減半: 32 -> 16
            DoubleConv(512, 1024)
        )
        
        # ========== Decoder 路徑（上採樣） ==========
        
        # 第一層上採樣： [B, 1024, 16, 16] -> [B, 512, 32, 32]
        # 使用轉置卷積進行上採樣，然後與 encoder 的對應層 concat
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(1024, 512)  # 1024 = 512(up1) + 512(skip from down3)
        
        # 第二層上採樣： [B, 512, 32, 32] -> [B, 256, 64, 64]
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(512, 256)  # 512 = 256(up2) + 256(skip from down2)
        
        # 第三層上採樣： [B, 256, 64, 64] -> [B, 128, 128, 128]
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(256, 128)  # 256 = 128(up3) + 128(skip from down1)
        
        # 第四層上採樣： [B, 128, 128, 128] -> [B, 64, 256, 256]
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(128, 64)  # 128 = 64(up4) + 64(skip from inc)
        
        # 輸出層： [B, 64, 256, 256] -> [B, 1, 256, 256]
        # 使用 1x1 卷積將特徵圖映射到輸出類別數
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)
    
    def forward(self, x):
        """
        前向傳播
        
        參數:
            x: 輸入圖片張量 [B, 3, H, W]，值範圍 [0, 1]
        
        返回:
            logits: 輸出 logits [B, 1, H, W]，未經過 sigmoid
                    （訓練時使用 BCEWithLogitsLoss 會自動處理 sigmoid）
        """
        # ========== Encoder 路徑 ==========
        
        # 第一層：輸入層
        x1 = self.inc(x)  # [B, 3, 256, 256] -> [B, 64, 256, 256]
        
        # 第二層：下採樣
        x2 = self.down1(x1)  # [B, 64, 256, 256] -> [B, 128, 128, 128]
        
        # 第三層：下採樣
        x3 = self.down2(x2)  # [B, 128, 128, 128] -> [B, 256, 64, 64]
        
        # 第四層：下採樣
        x4 = self.down3(x3)  # [B, 256, 64, 64] -> [B, 512, 32, 32]
        
        # 第五層：最底層（瓶頸層）
        x5 = self.down4(x4)  # [B, 512, 32, 32] -> [B, 1024, 16, 16]
        
        # ========== Decoder 路徑（帶 Skip Connections） ==========
        
        # 第一層上採樣：連接 x4 (encoder 第四層)
        x = self.up1(x5)  # [B, 1024, 16, 16] -> [B, 512, 32, 32]
        # 處理尺寸不匹配（如果有的話）
        if x.size() != x4.size():
            x = F.interpolate(x, size=x4.shape[2:], mode='bilinear', align_corners=False)
        # Skip connection: 將 encoder 的特徵圖與 decoder 的特徵圖拼接
        x = torch.cat([x4, x], dim=1)  # [B, 512+512, 32, 32] = [B, 1024, 32, 32]
        x = self.conv1(x)  # [B, 1024, 32, 32] -> [B, 512, 32, 32]
        
        # 第二層上採樣：連接 x3 (encoder 第三層)
        x = self.up2(x)  # [B, 512, 32, 32] -> [B, 256, 64, 64]
        if x.size() != x3.size():
            x = F.interpolate(x, size=x3.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x3, x], dim=1)  # [B, 256+256, 64, 64] = [B, 512, 64, 64]
        x = self.conv2(x)  # [B, 512, 64, 64] -> [B, 256, 64, 64]
        
        # 第三層上採樣：連接 x2 (encoder 第二層)
        x = self.up3(x)  # [B, 256, 64, 64] -> [B, 128, 128, 128]
        if x.size() != x2.size():
            x = F.interpolate(x, size=x2.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x2, x], dim=1)  # [B, 128+128, 128, 128] = [B, 256, 128, 128]
        x = self.conv3(x)  # [B, 256, 128, 128] -> [B, 128, 128, 128]
        
        # 第四層上採樣：連接 x1 (encoder 第一層)
        x = self.up4(x)  # [B, 128, 128, 128] -> [B, 64, 256, 256]
        if x.size() != x1.size():
            x = F.interpolate(x, size=x1.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x1, x], dim=1)  # [B, 64+64, 256, 256] = [B, 128, 256, 256]
        x = self.conv4(x)  # [B, 128, 256, 256] -> [B, 64, 256, 256]
        
        # 輸出層：1x1 卷積生成最終 mask
        logits = self.outc(x)  # [B, 64, 256, 256] -> [B, 1, 256, 256]
        
        return logits
    
    def predict_mask(self, x, threshold=0.5):
        """
        預測 mask（用於推理階段）
        
        參數:
            x: 輸入圖片張量 [B, 3, H, W]
            threshold: 二進制化閾值，預設 0.5
        
        返回:
            mask: 二進制 mask [B, 1, H, W]，值為 0 或 1
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            # 使用 sigmoid 將 logits 轉換為機率
            probs = torch.sigmoid(logits)
            # 二進制化
            mask = (probs > threshold).float()
        return mask


def create_unet_model(n_channels=3, n_classes=1):
    """
    創建 U-Net 模型的便利函數
    
    參數:
        n_channels: 輸入通道數（RGB 為 3）
        n_classes: 輸出類別數（二進制分割為 1）
    
    返回:
        UNet 模型實例
    """
    model = UNet(n_channels=n_channels, n_classes=n_classes)
    return model
