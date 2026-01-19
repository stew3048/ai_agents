"""
測試 U-Net 模型
驗證模型結構、輸入輸出尺寸是否正確
"""

import torch
from models import create_unet_model
import torchsummary


def test_unet_model():
    """測試 U-Net 模型"""
    
    print("=" * 60)
    print("測試 U-Net 模型")
    print("=" * 60)
    print()
    
    # 創建模型
    print("1. 創建 U-Net 模型...")
    model = create_unet_model(n_channels=3, n_classes=1)
    print("✓ 模型創建成功")
    print()
    
    # 計算模型參數數量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"2. 模型參數統計:")
    print(f"   - 總參數數量: {total_params:,}")
    print(f"   - 可訓練參數: {trainable_params:,}")
    print()
    
    # 測試輸入輸出尺寸
    print("3. 測試輸入輸出尺寸...")
    batch_size = 2
    input_size = (256, 256)  # 符合 dataset loader 的預設尺寸
    
    # 創建模擬輸入 [B, 3, 256, 256]
    dummy_input = torch.randn(batch_size, 3, input_size[0], input_size[1])
    print(f"   輸入形狀: {dummy_input.shape}")
    
    # 前向傳播
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"   輸出形狀: {output.shape}")
    print(f"   輸出值範圍: [{output.min():.3f}, {output.max():.3f}]")
    print()
    
    # 驗證尺寸是否正確
    expected_output_shape = (batch_size, 1, input_size[0], input_size[1])
    if output.shape == expected_output_shape:
        print("✓ 輸出尺寸正確！")
    else:
        print(f"✗ 輸出尺寸錯誤！期望: {expected_output_shape}, 實際: {output.shape}")
    print()
    
    # 測試 predict_mask 方法
    print("4. 測試 predict_mask 方法...")
    mask = model.predict_mask(dummy_input)
    print(f"   Mask 形狀: {mask.shape}")
    print(f"   Mask 值範圍: [{mask.min():.3f}, {mask.max():.3f}]")
    print(f"   Mask 唯一值: {torch.unique(mask).tolist()}")
    
    if mask.min() == 0.0 and mask.max() == 1.0:
        print("✓ Mask 二進制化正確！")
    else:
        print("✗ Mask 二進制化可能有問題")
    print()
    
    # 測試模型摘要（如果 torchsummary 可用）
    print("5. 模型結構摘要:")
    try:
        torchsummary.summary(model, (3, 256, 256), device='cpu')
    except Exception as e:
        print(f"   無法顯示詳細摘要（需要 torchsummary 套件）")
        print(f"   錯誤: {e}")
        print()
        print("   模型主要結構:")
        print("   - Encoder: 5 個下採樣階段 (3->64->128->256->512->1024)")
        print("   - Decoder: 4 個上採樣階段 (1024->512->256->128->64->1)")
        print("   - Skip Connections: 連接 encoder 和 decoder 對應層")
    
    print()
    print("=" * 60)
    print("✓ U-Net 模型測試完成！")
    print("=" * 60)
    
    # 測試不同輸入尺寸（可選）
    print()
    print("6. 測試不同輸入尺寸的適應性...")
    test_sizes = [(128, 128), (512, 512)]
    for h, w in test_sizes:
        test_input = torch.randn(1, 3, h, w)
        try:
            with torch.no_grad():
                test_output = model(test_input)
            print(f"   ✓ 輸入 [{h}, {w}] -> 輸出 {test_output.shape}")
        except Exception as e:
            print(f"   ✗ 輸入 [{h}, {w}] 失敗: {e}")


if __name__ == '__main__':
    test_unet_model()
