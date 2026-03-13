"""
推理腳本
用於載入訓練好的模型並對圖片進行天空分割預測
"""

import os
import sys
import argparse
import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF

from models import create_unet_model


def load_model(checkpoint_path, device='cpu'):
    """
    載入訓練好的模型
    
    參數:
        checkpoint_path: 檢查點檔案路徑 (.pth)
        device: 計算設備
    
    返回:
        model: 載入權重的模型
    """
    # 創建模型
    model = create_unet_model(n_channels=3, n_classes=1)
    
    # 載入檢查點
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 移到設備並設為評估模式
    model = model.to(device)
    model.eval()
    
    # 顯示檢查點資訊
    if 'epoch' in checkpoint:
        print(f"載入模型: Epoch {checkpoint['epoch']}")
    if 'metrics' in checkpoint:
        metrics = checkpoint['metrics']
        print(f"模型指標: IoU={metrics.get('iou', 'N/A'):.4f}, Dice={metrics.get('dice', 'N/A'):.4f}")
    
    return model


def preprocess_image(image_path, image_size=(256, 256)):
    """
    預處理輸入圖片
    
    參數:
        image_path: 圖片路徑
        image_size: 目標尺寸 (height, width)
    
    返回:
        tensor: 預處理後的張量 [1, 3, H, W]
        original_size: 原始圖片尺寸 (width, height)
    """
    # 讀取圖片
    image = Image.open(image_path).convert('RGB')
    original_size = image.size  # (width, height)
    
    # 調整尺寸
    image = image.resize((image_size[1], image_size[0]), Image.BILINEAR)
    
    # 轉換為張量 [0, 1]
    tensor = TF.to_tensor(image)
    
    # 添加 batch 維度
    tensor = tensor.unsqueeze(0)
    
    return tensor, original_size


def predict(model, image_tensor, device='cpu', threshold=0.5):
    """
    執行預測
    
    參數:
        model: 模型
        image_tensor: 輸入張量 [1, 3, H, W]
        device: 計算設備
        threshold: 二進制化閾值
    
    返回:
        mask: 預測的 mask [H, W]，值為 0 或 255
        prob: 機率圖 [H, W]，值範圍 [0, 1]
    """
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        # 前向傳播
        logits = model(image_tensor)
        
        # 轉換為機率
        prob = torch.sigmoid(logits)
        
        # 二進制化
        mask = (prob > threshold).float()
    
    # 轉換為 numpy
    prob = prob.squeeze().cpu().numpy()
    mask = (mask.squeeze().cpu().numpy() * 255).astype(np.uint8)
    
    return mask, prob


def save_results(mask, prob, output_dir, image_name, original_size=None):
    """
    儲存預測結果
    
    參數:
        mask: 預測的 mask
        prob: 機率圖
        output_dir: 輸出目錄
        image_name: 圖片名稱
        original_size: 原始圖片尺寸 (width, height)，如果提供則調整回原始尺寸
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 轉換為 PIL Image
    mask_image = Image.fromarray(mask)
    prob_image = Image.fromarray((prob * 255).astype(np.uint8))
    
    # 如果需要，調整回原始尺寸
    if original_size is not None:
        mask_image = mask_image.resize(original_size, Image.NEAREST)
        prob_image = prob_image.resize(original_size, Image.BILINEAR)
    
    # 儲存
    base_name = os.path.splitext(image_name)[0]
    mask_path = os.path.join(output_dir, f'{base_name}_mask.png')
    prob_path = os.path.join(output_dir, f'{base_name}_prob.png')
    
    mask_image.save(mask_path)
    prob_image.save(prob_path)
    
    print(f"  Mask 儲存至: {mask_path}")
    print(f"  機率圖儲存至: {prob_path}")
    
    return mask_path, prob_path


def main():
    parser = argparse.ArgumentParser(description='天空分割推理')
    
    # 必需參數
    parser.add_argument('--input', type=str, required=True,
                       help='輸入圖片路徑或資料夾路徑')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='模型檢查點路徑 (.pth)')
    
    # 可選參數
    parser.add_argument('--output', type=str, default='outputs/predictions',
                       help='輸出資料夾路徑（預設: outputs/predictions）')
    parser.add_argument('--image_size', type=int, nargs=2, default=[256, 256],
                       help='推理時的圖片尺寸 [height, width]（預設: 256 256）')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='二進制化閾值（預設: 0.5）')
    parser.add_argument('--keep_original_size', action='store_true',
                       help='輸出 mask 保持原始圖片尺寸')
    parser.add_argument('--cpu_only', action='store_true',
                       help='強制使用 CPU')
    
    args = parser.parse_args()
    
    # 設置設備
    use_cuda = torch.cuda.is_available() and not args.cpu_only
    device = torch.device('cuda' if use_cuda else 'cpu')
    print(f"使用設備: {device}")
    
    # 載入模型
    print(f"\n載入模型: {args.checkpoint}")
    model = load_model(args.checkpoint, device)
    
    # 收集要處理的圖片
    if os.path.isfile(args.input):
        # 單一圖片
        image_paths = [args.input]
    elif os.path.isdir(args.input):
        # 資料夾
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
        image_paths = [
            os.path.join(args.input, f) 
            for f in os.listdir(args.input) 
            if os.path.splitext(f.lower())[1] in valid_extensions
        ]
    else:
        print(f"[ERROR] 找不到輸入: {args.input}")
        sys.exit(1)
    
    if len(image_paths) == 0:
        print("[ERROR] 找不到有效的圖片檔案")
        sys.exit(1)
    
    print(f"\n找到 {len(image_paths)} 張圖片")
    print("=" * 60)
    
    # 處理每張圖片
    for i, image_path in enumerate(image_paths):
        image_name = os.path.basename(image_path)
        print(f"\n[{i+1}/{len(image_paths)}] 處理: {image_name}")
        
        # 預處理
        image_tensor, original_size = preprocess_image(
            image_path, 
            image_size=tuple(args.image_size)
        )
        
        # 預測
        mask, prob = predict(
            model, 
            image_tensor, 
            device=device, 
            threshold=args.threshold
        )
        
        # 儲存結果
        save_results(
            mask, 
            prob, 
            args.output, 
            image_name,
            original_size=original_size if args.keep_original_size else None
        )
    
    print("\n" + "=" * 60)
    print(f"[OK] 完成！結果儲存在: {args.output}")


if __name__ == '__main__':
    main()
