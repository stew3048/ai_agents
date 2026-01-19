"""
Toy Dataset Visual Verification
視覺化驗收 DataLoader 輸出是否正確

輸出：
1. Original image
2. GT mask (black/white)
3. GT overlay (mask 半透明疊在圖上)
4. Resized overlay (256x256, 確認 resize 後對齊)
"""

import os
import sys
import numpy as np
from PIL import Image
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.dataset import get_dataloader


def tensor_to_pil(tensor):
    """
    Convert image tensor [C, H, W] to PIL Image
    """
    # [C, H, W] -> [H, W, C]
    arr = tensor.permute(1, 2, 0).numpy()
    # [0, 1] -> [0, 255]
    arr = (arr * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def mask_tensor_to_pil(tensor):
    """
    Convert mask tensor [1, H, W] to PIL Image (grayscale)
    """
    # [1, H, W] -> [H, W]
    arr = tensor.squeeze(0).numpy()
    # [0, 1] -> [0, 255]
    arr = (arr * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode='L')


def create_overlay(image_pil, mask_pil, color=(0, 100, 255), alpha=0.5):
    """
    Create overlay: mask region with semi-transparent color on image
    
    Args:
        image_pil: PIL Image (RGB)
        mask_pil: PIL Image (L, grayscale)
        color: RGB tuple for mask region
        alpha: transparency (0=transparent, 1=opaque)
    
    Returns:
        PIL Image with overlay
    """
    # Convert to numpy
    image_arr = np.array(image_pil).astype(np.float32)
    mask_arr = np.array(mask_pil).astype(np.float32) / 255.0
    
    # Create color overlay
    overlay = np.zeros_like(image_arr)
    overlay[:, :, 0] = color[0]
    overlay[:, :, 1] = color[1]
    overlay[:, :, 2] = color[2]
    
    # Blend: only where mask == 1 (sky)
    mask_3ch = np.stack([mask_arr] * 3, axis=-1)
    result = image_arr * (1 - mask_3ch * alpha) + overlay * (mask_3ch * alpha)
    
    result = result.clip(0, 255).astype(np.uint8)
    return Image.fromarray(result)


def verify_resize(image_pil, mask_pil, target_size=(256, 256)):
    """
    Verify resize behavior
    - Image: bilinear interpolation
    - Mask: nearest interpolation (critical for segmentation!)
    
    Returns:
        resized_image, resized_mask, resized_overlay
    """
    # Resize image (bilinear)
    resized_image = image_pil.resize(target_size, Image.BILINEAR)
    
    # Resize mask (NEAREST - very important!)
    resized_mask = mask_pil.resize(target_size, Image.NEAREST)
    
    # Create overlay
    resized_overlay = create_overlay(resized_image, resized_mask)
    
    return resized_image, resized_mask, resized_overlay


def main():
    # Output directory
    output_dir = 'outputs/toy_loader_vis'
    os.makedirs(output_dir, exist_ok=True)
    
    print('=' * 60)
    print('  Toy Dataset Visual Verification')
    print('=' * 60)
    print()
    
    # Create dataloader (no augmentation for verification)
    print('[1] Loading DataLoader...')
    dataloader = get_dataloader(
        images_dir='data/toy/images',
        masks_dir='data/toy/masks',
        batch_size=5,
        shuffle=False,
        transform=False  # No augmentation
    )
    
    # Get 2 batches (10 samples)
    print('[2] Extracting 2 batches (10 samples)...')
    samples = []
    for i, (images, masks) in enumerate(dataloader):
        if i >= 2:
            break
        for j in range(images.shape[0]):
            samples.append((images[j], masks[j]))
    
    print(f'    Extracted {len(samples)} samples')
    print()
    
    # Process each sample
    print('[3] Generating visualizations...')
    print('-' * 60)
    
    for idx, (image_tensor, mask_tensor) in enumerate(samples):
        sample_id = f'{idx+1:03d}'
        
        # Convert tensors to PIL
        image_pil = tensor_to_pil(image_tensor)
        mask_pil = mask_tensor_to_pil(mask_tensor)
        
        # Create overlay
        overlay_pil = create_overlay(image_pil, mask_pil, color=(0, 100, 255), alpha=0.5)
        
        # Verify resize
        resized_img, resized_mask, resized_overlay = verify_resize(
            image_pil, mask_pil, target_size=(256, 256)
        )
        
        # Save outputs
        image_pil.save(os.path.join(output_dir, f'{sample_id}_1_original.png'))
        mask_pil.save(os.path.join(output_dir, f'{sample_id}_2_mask.png'))
        overlay_pil.save(os.path.join(output_dir, f'{sample_id}_3_overlay.png'))
        resized_overlay.save(os.path.join(output_dir, f'{sample_id}_4_resized_overlay.png'))
        
        print(f'    Sample {sample_id}: saved 4 images')
    
    print('-' * 60)
    print()
    print(f'[4] Output saved to: {output_dir}/')
    print()
    print('    Files per sample:')
    print('    - XXX_1_original.png      : Original image from DataLoader')
    print('    - XXX_2_mask.png          : GT mask (white=sky, black=non-sky)')
    print('    - XXX_3_overlay.png       : Mask overlaid on image (blue=sky)')
    print('    - XXX_4_resized_overlay.png : After resize to 256x256')
    print()
    print('=' * 60)
    print()
    
    # Print verification guide
    print('=' * 60)
    print('  HOW TO VERIFY ALIGNMENT')
    print('=' * 60)
    print()
    print('[CORRECT Alignment]')
    print('  - Blue overlay covers EXACTLY the sky region')
    print('  - Boundary between sky/ground is sharp and follows horizon')
    print('  - No blue color on ground, no missing blue on sky')
    print()
    print('[WRONG Alignment - 3 Common Cases]')
    print()
    print('  1. SHIFTED:')
    print('     - Blue region is offset from actual sky')
    print('     - Example: ground has blue, sky edge is not covered')
    print('     - Cause: image/mask filename mismatch')
    print()
    print('  2. BLURRY BOUNDARY:')
    print('     - Mask edge is soft/gradient instead of sharp')
    print('     - Gray pixels visible at sky/ground boundary')
    print('     - Cause: mask resized with BILINEAR instead of NEAREST')
    print()
    print('  3. INVERTED:')
    print('     - Ground is blue, sky is not covered')
    print('     - Mask values are flipped (0=sky, 1=ground)')
    print('     - Cause: mask interpretation reversed')
    print()
    print('=' * 60)


if __name__ == '__main__':
    main()
