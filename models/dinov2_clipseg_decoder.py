"""
DINOv2 + CLIPSeg Prior + CNN Decoder（語意增強版）

架構說明
--------
- Backbone  : DINOv2 ViT-S/14（凍結），提取最後四層 patch tokens → concat → 1536 維
- Semantic Prior : CLIPSeg heatmap（單通道機率圖）resize 後接在 1536 維特徵後
  → 合併特徵維度 1537（1536 DINOv2 + 1 CLIPSeg）
- Decoder   : Block1 1x1 1537→512, Block2 3x3 512→512, Block3 3x3 512→256, Block4 1x1 256→1

與原 DINOv2SegmentationModel 的差異
-------------------------------------
1. forward() 接受額外參數 clipseg_heatmap (B, 1, H, W) 或 None
2. Decoder 第一層輸入從 1536 改為 1537
3. 提供 load_from_dino_checkpoint() 可從舊的 1536 版本遷移權重
   （新增的第 1537 個輸入 channel 以零初始化，等同「不看 heatmap」的中性初始值）

使用方式（推論）
----------------
model = create_dinov2_clipseg_model()
model.load_state_dict(torch.load("best.pth"))
logits = model(images, clipseg_heatmap)   # heatmap: (B,1,H,W), 值域 0~1

clipseg_heatmap 傳入 None 時，自動以全零 heatmap 填補（退化為純 DINOv2 模式）。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

DINOV2_PATCH_SIZE = 14
DINOV2_LAST4_CONCAT_DIM = 1536   # ViT-S/14：4 層 × 384 = 1536
FUSED_DIM = DINOV2_LAST4_CONCAT_DIM + 1  # + 1 CLIPSeg channel = 1537


def conv_block(in_channels, out_channels, kernel_size):
    padding = 0 if kernel_size == 1 else (1 if kernel_size == 3 else kernel_size // 2)
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


def _load_dinov2_backbone(arch_name='dinov2_vits14'):
    model = torch.hub.load('facebookresearch/dinov2', arch_name, pretrained=True)
    for p in model.parameters():
        p.requires_grad = False
    return model


class DINOv2CLIPSegModel(nn.Module):
    """
    DINOv2（凍結）+ CLIPSeg heatmap prior + CNN 解碼器。

    Parameters
    ----------
    backbone : nn.Module | None
        預先載入的 DINOv2 backbone；None 時自動從 torch.hub 下載。
    patch_size : int
        DINOv2 patch 大小，預設 14。
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

        # Decoder：第一層接收 1537 channels（1536 DINOv2 + 1 CLIPSeg）
        self.decoder = nn.Sequential(
            conv_block(FUSED_DIM, 512, kernel_size=1),  # Block 1: 1x1, 1537 → 512
            conv_block(512, 512,      kernel_size=3),   # Block 2: 3x3, 512  → 512
            conv_block(512, 256,      kernel_size=3),   # Block 3: 3x3, 512  → 256
            nn.Conv2d(256, 1,         kernel_size=1),   # Block 4: 1x1, 256  → 1 (Logit)
        )

    def forward(self, images, clipseg_heatmap=None):
        """
        Parameters
        ----------
        images : Tensor (B, 3, H, W)
            RGB 影像，值域 0~1（或 normalized）。
        clipseg_heatmap : Tensor (B, 1, H, W) | None
            CLIPSeg 輸出的機率圖，值域建議 0~1。
            傳入 None 時以全零填補（退化為純 DINOv2 模式）。

        Returns
        -------
        logits : Tensor (B, 1, H, W)，未經 Sigmoid。
        """
        B, _, H, W = images.shape

        # ── 1. 對齊到 patch_size 倍數 ─────────────────────────────
        H2 = max((H // self.patch_size) * self.patch_size, self.patch_size)
        W2 = max((W // self.patch_size) * self.patch_size, self.patch_size)
        if H != H2 or W != W2:
            images = F.interpolate(images, size=(H2, W2), mode='bilinear', align_corners=False)

        h = H2 // self.patch_size
        w = W2 // self.patch_size

        # ── 2. DINOv2 最後四層特徵（backbone 凍結）───────────────
        with torch.no_grad():
            outputs = self.backbone.get_intermediate_layers(images, n=4, norm=True)
        # outputs: list of (B, N, 384)，N = h*w
        patch_cat = torch.cat(outputs, dim=-1)                                  # (B, N, 1536)
        patch_cat = patch_cat.transpose(1, 2).reshape(B, DINOV2_LAST4_CONCAT_DIM, h, w)  # (B, 1536, h, w)

        # ── 3. CLIPSeg heatmap 前處理 ────────────────────────────
        if clipseg_heatmap is None:
            # 退化模式：全零 prior（等同忽略 CLIPSeg）
            heatmap = torch.zeros(B, 1, h, w, device=images.device, dtype=patch_cat.dtype)
        else:
            # 正規化到 [0, 1]
            hm = clipseg_heatmap.float()
            hm_min = hm.flatten(2).min(dim=-1)[0].unsqueeze(-1).unsqueeze(-1)
            hm_max = hm.flatten(2).max(dim=-1)[0].unsqueeze(-1).unsqueeze(-1)
            hm = (hm - hm_min) / (hm_max - hm_min + 1e-8)

            # Resize 到 DINOv2 feature map 的空間尺寸 (h, w)
            heatmap = F.interpolate(hm, size=(h, w), mode='bilinear', align_corners=False)

        # ── 4. Concatenate → (B, 1537, h, w) ───────────────────
        fused = torch.cat([patch_cat, heatmap], dim=1)

        # ── 5. Decoder ──────────────────────────────────────────
        logits = self.decoder(fused)

        # ── 6. 還原至原始影像尺寸 ────────────────────────────────
        logits = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=False)

        return logits

    @classmethod
    def load_from_dino_checkpoint(cls, checkpoint_path, backbone=None, **kwargs):
        """
        從原版 DINOv2SegmentationModel（1536 輸入）的 checkpoint 遷移權重。

        decoder 第一層原本是 Conv2d(1536, 512)，遷移後變成 Conv2d(1537, 512)。
        新增的第 1537 個輸入 channel 以「零權重」初始化：
        等同開訓練初期完全不看 CLIPSeg heatmap，讓模型自行學習其重要性。

        Parameters
        ----------
        checkpoint_path : str
            原版 best.pth 路徑。
        backbone : nn.Module | None
            選擇性傳入已載入的 backbone。

        Returns
        -------
        model : DINOv2CLIPSegModel（已載入權重）
        """
        model = cls(backbone=backbone, **kwargs)
        state = torch.load(checkpoint_path, map_location='cpu')

        # 相容 {'model': ...} 與直接 state_dict 兩種儲存格式
        if 'model' in state:
            state = state['model']

        new_state = model.state_dict()
        loaded, skipped = [], []

        for k, v in state.items():
            if k not in new_state:
                skipped.append(k)
                continue
            if new_state[k].shape == v.shape:
                new_state[k] = v
                loaded.append(k)
            elif k == 'decoder.0.0.weight':
                # Conv2d weight shape: (out_ch, in_ch, kH, kW)
                # 原 (512, 1536, 1, 1) → 新 (512, 1537, 1, 1)
                new_w = torch.zeros_like(new_state[k])  # (512, 1537, 1, 1)
                new_w[:, :1536, :, :] = v               # 複製舊的 1536 channels
                # new_w[:, 1536, :, :] 保持為 0（不看 heatmap 的中性初始值）
                new_state[k] = new_w
                loaded.append(f"{k} [zero-padded 1536→1537]")
            else:
                skipped.append(f"{k} (shape mismatch: {v.shape} vs {new_state[k].shape})")

        model.load_state_dict(new_state)
        print(f"[load_from_dino_checkpoint] loaded {len(loaded)} keys")
        if skipped:
            print(f"  skipped: {skipped}")
        return model


def create_dinov2_clipseg_model(backbone=None, **kwargs):
    """工廠函數：建立 DINOv2CLIPSegModel（1537 channels）。"""
    return DINOv2CLIPSegModel(backbone=backbone, **kwargs)
