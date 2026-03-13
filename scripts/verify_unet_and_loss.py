"""
驗證 U-Net 模型和 Loss 函數可以正常運作

檢查項目：
1. U-Net 模型可以建立
2. Forward pass 輸出正確的 shape
3. Loss 函數可以計算
4. Backward pass 可以執行（梯度計算）
"""

import os
import sys
import torch

# 加入專案路徑
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.unet import UNet, UNetSmall, get_unet
from src.utils.losses import bce_with_logits_loss, dice_loss, combined_loss


def count_parameters(model):
    """計算模型參數量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    print('=' * 60)
    print('  U-Net and Loss Function Verification')
    print('=' * 60)
    print()
    
    # 設定（使用較小的 batch size 和 CPU 以避免 OOM）
    batch_size = 2
    height, width = 256, 256
    
    # 先用 CPU 測試（避免 GPU OOM）
    device = 'cpu'
    
    # 檢查 GPU 是否可用
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f'  GPU available: Yes ({gpu_mem:.1f} GB)')
        print(f'  Note: Using CPU for testing to avoid OOM')
    else:
        print(f'  GPU available: No')
    
    print(f'  Device: {device}')
    print(f'  Input shape: ({batch_size}, 3, {height}, {width})')
    print()
    
    # 建立假資料
    dummy_input = torch.randn(batch_size, 3, height, width).to(device)
    dummy_target = torch.randint(0, 2, (batch_size, 1, height, width)).float().to(device)
    
    # ========================================
    # 測試 UNetSmall
    # ========================================
    print('[1] Testing UNetSmall')
    print('-' * 40)
    
    model_small = UNetSmall(in_channels=3, out_channels=1).to(device)
    params_small = count_parameters(model_small)
    print(f'  Parameters: {params_small:,}')
    
    # Forward pass
    model_small.eval()
    with torch.no_grad():
        output_small = model_small(dummy_input)
    
    print(f'  Input shape:  {dummy_input.shape}')
    print(f'  Output shape: {output_small.shape}')
    print(f'  Output range: [{output_small.min():.4f}, {output_small.max():.4f}]')
    
    # 驗證 shape
    expected_shape = torch.Size([batch_size, 1, height, width])
    shape_ok = output_small.shape == expected_shape
    print(f'  Shape correct: {shape_ok}')
    print()
    
    # ========================================
    # 測試 UNet (標準版)
    # ========================================
    print('[2] Testing UNet (standard)')
    print('-' * 40)
    
    model_standard = UNet(in_channels=3, out_channels=1).to(device)
    params_standard = count_parameters(model_standard)
    print(f'  Parameters: {params_standard:,}')
    
    # Forward pass
    model_standard.eval()
    with torch.no_grad():
        output_standard = model_standard(dummy_input)
    
    print(f'  Input shape:  {dummy_input.shape}')
    print(f'  Output shape: {output_standard.shape}')
    print(f'  Output range: [{output_standard.min():.4f}, {output_standard.max():.4f}]')
    
    shape_ok_std = output_standard.shape == expected_shape
    print(f'  Shape correct: {shape_ok_std}')
    print()
    
    # ========================================
    # 測試 Loss Functions
    # ========================================
    print('[3] Testing Loss Functions')
    print('-' * 40)
    
    # 用 small model 的輸出來測試
    model_small.train()
    output = model_small(dummy_input)
    
    # BCE Loss
    bce = bce_with_logits_loss(output, dummy_target)
    print(f'  BCE Loss: {bce.item():.4f}')
    
    # Dice Loss
    dice = dice_loss(output, dummy_target)
    print(f'  Dice Loss: {dice.item():.4f}')
    
    # Combined Loss
    combined = combined_loss(output, dummy_target)
    print(f'  Combined Loss: {combined.item():.4f}')
    
    # 驗證 loss 是合理的值
    loss_ok = (0 < bce.item() < 10) and (0 < dice.item() < 1) and (0 < combined.item() < 10)
    print(f'  Loss values reasonable: {loss_ok}')
    print()
    
    # ========================================
    # 測試 Backward Pass
    # ========================================
    print('[4] Testing Backward Pass')
    print('-' * 40)
    
    # 確認梯度可以計算
    optimizer = torch.optim.Adam(model_small.parameters(), lr=1e-4)
    optimizer.zero_grad()
    
    output = model_small(dummy_input)
    loss = combined_loss(output, dummy_target)
    loss.backward()
    
    # 檢查梯度
    has_grad = False
    for name, param in model_small.named_parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_grad = True
            break
    
    print(f'  Loss computed: {loss.item():.4f}')
    print(f'  Gradients exist: {has_grad}')
    
    # 模擬一步 optimizer step
    optimizer.step()
    print(f'  Optimizer step: OK')
    print()
    
    # ========================================
    # 測試 get_unet helper
    # ========================================
    print('[5] Testing get_unet helper')
    print('-' * 40)
    
    model_via_helper = get_unet(model_type='small')
    print(f'  get_unet("small"): {type(model_via_helper).__name__}')
    
    model_via_helper_std = get_unet(model_type='standard')
    print(f'  get_unet("standard"): {type(model_via_helper_std).__name__}')
    print()
    
    # ========================================
    # 測試與真實 DataLoader 的整合
    # ========================================
    print('[6] Testing with Real DataLoader')
    print('-' * 40)
    
    try:
        from src.datasets import get_skyfinder_dataloader
        
        dataloader = get_skyfinder_dataloader(
            splits_json='outputs/splits_10066.json',
            split='train',
            batch_size=2,  # 較小的 batch size
            shuffle=False,
            transform=False
        )
        
        # 取一個 batch
        images, masks = next(iter(dataloader))
        # 保持在 CPU 上測試
        
        print(f'  Images from DataLoader: {images.shape}')
        print(f'  Masks from DataLoader: {masks.shape}')
        
        # Forward pass（用新的 model instance 避免狀態問題）
        model_test = UNetSmall(in_channels=3, out_channels=1)
        model_test.eval()
        with torch.no_grad():
            output = model_test(images)
        
        print(f'  Model output: {output.shape}')
        
        # 計算 loss
        loss = combined_loss(output, masks)
        print(f'  Loss on real data: {loss.item():.4f}')
        
        dataloader_ok = True
    except Exception as e:
        print(f'  Error: {e}')
        dataloader_ok = False
    
    print()
    
    # ========================================
    # 總結
    # ========================================
    print('=' * 60)
    print('  Verification Summary')
    print('=' * 60)
    print()
    
    checks = [
        ('UNetSmall output shape', shape_ok),
        ('UNet output shape', shape_ok_std),
        ('Loss values reasonable', loss_ok),
        ('Gradients computed', has_grad),
        ('DataLoader integration', dataloader_ok),
    ]
    
    all_pass = True
    for name, passed in checks:
        status = 'PASS' if passed else 'FAIL'
        print(f'  [{status}] {name}')
        if not passed:
            all_pass = False
    
    print()
    print('  Model comparison:')
    print(f'    UNetSmall: {params_small:,} parameters')
    print(f'    UNet:      {params_standard:,} parameters')
    print(f'    Ratio:     {params_standard / params_small:.1f}x')
    print()
    
    if all_pass:
        print('  Result: ALL CHECKS PASSED!')
        print('  U-Net and Loss functions are ready for training.')
    else:
        print('  Result: SOME CHECKS FAILED!')
    
    print()
    print('=' * 60)


if __name__ == '__main__':
    main()
