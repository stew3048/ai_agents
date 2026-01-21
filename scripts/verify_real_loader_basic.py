"""
驗證 SkyFinder 10066 Dataset 的基本功能
- 抽 1 個 batch
- 印出 image/mask shape、dtype、unique values
"""

import os
import sys
import torch

# 加入專案路徑
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.datasets.skyfinder_10066_dataset import SkyFinder10066Dataset, get_skyfinder_dataloader


def main():
    print('=' * 60)
    print('  SkyFinder 10066 Dataset Verification')
    print('=' * 60)
    print()
    
    # 測試每個 split
    for split in ['train', 'val', 'test']:
        print(f'[{split.upper()}]')
        print('-' * 40)
        
        # 建立 Dataset
        dataset = SkyFinder10066Dataset(
            splits_json='outputs/splits_10066.json',
            split=split,
            image_size=(256, 256),
            transform=False  # 驗證時關閉 augmentation
        )
        
        print(f'  Dataset size: {len(dataset)} samples')
        print()
        
        # 取得單一樣本
        image, mask = dataset[0]
        
        print('  Single sample:')
        print(f'    Image shape: {image.shape}')
        print(f'    Image dtype: {image.dtype}')
        print(f'    Image min/max: {image.min():.4f} / {image.max():.4f}')
        print()
        print(f'    Mask shape: {mask.shape}')
        print(f'    Mask dtype: {mask.dtype}')
        print(f'    Mask unique values: {torch.unique(mask).tolist()}')
        print()
        
        # 建立 DataLoader
        batch_size = min(4, len(dataset))
        dataloader = get_skyfinder_dataloader(
            splits_json='outputs/splits_10066.json',
            split=split,
            batch_size=batch_size,
            shuffle=False
        )
        
        # 取得一個 batch
        images, masks = next(iter(dataloader))
        
        print(f'  Batch (size={batch_size}):')
        print(f'    Images shape: {images.shape}')
        print(f'    Images dtype: {images.dtype}')
        print(f'    Images min/max: {images.min():.4f} / {images.max():.4f}')
        print()
        print(f'    Masks shape: {masks.shape}')
        print(f'    Masks dtype: {masks.dtype}')
        print(f'    Masks unique values: {torch.unique(masks).tolist()}')
        print()
        
        # 印出 batch 中的檔名
        print('  Batch filenames:')
        for i in range(batch_size):
            print(f'    {i+1}. {dataset.get_filename(i)}')
        print()
    
    # 驗證摘要
    print('=' * 60)
    print('  Verification Summary')
    print('=' * 60)
    print()
    print('  Expected values:')
    print('    Image: float32, shape (B, 3, 256, 256), range [0, 1]')
    print('    Mask:  float32, shape (B, 1, 256, 256), values {0, 1}')
    print()
    
    # 驗證結果
    all_pass = True
    checks = [
        ('Image shape', images.shape == torch.Size([batch_size, 3, 256, 256])),
        ('Image dtype', images.dtype == torch.float32),
        ('Image range', images.min() >= 0 and images.max() <= 1),
        ('Mask shape', masks.shape == torch.Size([batch_size, 1, 256, 256])),
        ('Mask dtype', masks.dtype == torch.float32),
        ('Mask values', set(torch.unique(masks).tolist()).issubset({0.0, 1.0})),
    ]
    
    print('  Checks:')
    for name, passed in checks:
        status = 'PASS' if passed else 'FAIL'
        print(f'    [{status}] {name}')
        if not passed:
            all_pass = False
    
    print()
    if all_pass:
        print('  Result: ALL CHECKS PASSED!')
    else:
        print('  Result: SOME CHECKS FAILED!')
    
    print()
    print('=' * 60)


if __name__ == '__main__':
    main()
