"""
天空區域辨識 (Sky Segmentation) 主程序
使用 U-Net 或 DeepLabV3 作為 baseline 模型
"""

import argparse
import torch


def main():
    """主程序入口"""
    parser = argparse.ArgumentParser(description='天空區域辨識專案')
    parser.add_argument('--mode', type=str, default='train', 
                       choices=['train', 'test', 'predict'],
                       help='執行模式：train/test/predict')
    parser.add_argument('--model', type=str, default='unet',
                       choices=['unet', 'deeplabv3'],
                       help='選擇模型：unet 或 deeplabv3')
    
    args = parser.parse_args()
    
    print(f"執行模式: {args.mode}")
    print(f"選擇模型: {args.model}")
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 可用: {torch.cuda.is_available()}")
    
    # TODO: 後續步驟將在此處實現
    # - 資料載入
    # - 模型初始化
    # - 訓練/測試/預測流程


if __name__ == '__main__':
    main()
