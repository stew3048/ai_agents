"""
Data Loader Sanity Check
確認 Data Loader 輸出的格式正確
"""

import os
import numpy as np
from PIL import Image
import torch
from utils.dataset import get_dataloader

print('=' * 60)
print('  Data Loader Sanity Check')
print('=' * 60)

# ============================================================
# Part 1: Raw file check (before data loader)
# ============================================================
print()
print('-' * 60)
print('  Part 1: Raw File Check (Before Data Loader)')
print('-' * 60)

# Load single image
img_path = 'data/toy/images/001.jpg'
img = Image.open(img_path)
img_array = np.array(img)

print()
print(f'[Single Image] {img_path}')
print(f'  PIL mode:     {img.mode}')
print(f'  PIL size:     {img.size} (W x H)')
print(f'  Array shape:  {img_array.shape} (H x W x C)')
print(f'  Dtype:        {img_array.dtype}')
print(f'  Value range:  [{img_array.min()}, {img_array.max()}]')

# Load single mask
mask_path = 'data/toy/masks/001.png'
mask = Image.open(mask_path)
mask_array = np.array(mask)

print()
print(f'[Single Mask] {mask_path}')
print(f'  PIL mode:     {mask.mode}')
print(f'  PIL size:     {mask.size} (W x H)')
print(f'  Array shape:  {mask_array.shape} (H x W)')
print(f'  Dtype:        {mask_array.dtype}')
print(f'  Unique values: {set(np.unique(mask_array))}')
print(f'  Value range:  [{mask_array.min()}, {mask_array.max()}]')

# ============================================================
# Part 2: Data Loader output check
# ============================================================
print()
print('-' * 60)
print('  Part 2: Data Loader Output Check')
print('-' * 60)

# Create dataloader
dataloader = get_dataloader(
    images_dir='data/toy/images',
    masks_dir='data/toy/masks',
    batch_size=4,
    shuffle=False,
    transform=False  # Disable augmentation for consistent check
)

# Get one batch
batch_images, batch_masks = next(iter(dataloader))

print()
print('[Batch Images]')
print(f'  Shape:        {batch_images.shape}')
print(f'  Dtype:        {batch_images.dtype}')
print(f'  Value range:  [{batch_images.min():.4f}, {batch_images.max():.4f}]')

print()
print('[Batch Masks]')
print(f'  Shape:        {batch_masks.shape}')
print(f'  Dtype:        {batch_masks.dtype}')
print(f'  Unique values: {set(batch_masks.unique().numpy())}')
print(f'  Value range:  [{batch_masks.min():.4f}, {batch_masks.max():.4f}]')

# ============================================================
# Part 3: Summary
# ============================================================
print()
print('-' * 60)
print('  Part 3: Summary')
print('-' * 60)
print()

# Check expected values
checks = []

# Image shape check
img_shape_ok = batch_images.shape == torch.Size([4, 3, 256, 256])
checks.append(('Batch image shape [B,3,H,W]', img_shape_ok))

# Mask shape check
mask_shape_ok = batch_masks.shape == torch.Size([4, 1, 256, 256])
checks.append(('Batch mask shape [B,1,H,W]', mask_shape_ok))

# Image value range check
img_range_ok = batch_images.min() >= 0 and batch_images.max() <= 1
checks.append(('Image values in [0,1]', img_range_ok))

# Mask binary check
mask_values = set(batch_masks.unique().numpy())
mask_binary_ok = mask_values.issubset({0.0, 1.0})
checks.append(('Mask values binary {0,1}', mask_binary_ok))

for name, passed in checks:
    status = '[OK]' if passed else '[FAIL]'
    print(f'  {status} {name}')

print()
all_passed = all(p for _, p in checks)
if all_passed:
    print('[SUCCESS] All sanity checks passed!')
else:
    print('[WARNING] Some checks failed!')

print()
print('=' * 60)
