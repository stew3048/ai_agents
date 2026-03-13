"""
Toy Dataset V2 Visual Verification
驗證 resize 前後的 overlay 是否正確對齊

輸出 5 張圖（每個 sample）：
1. XXX_1_original.png       - 原始 image (512x384)
2. XXX_2_mask.png           - 原始 mask (512x384)
3. XXX_3_overlay_original.png - 原始尺寸 overlay (512x384)
4. XXX_4_overlay_resized.png  - 手動 resize 後 overlay (256x256)
5. XXX_5_from_dataloader.png  - DataLoader 輸出的 overlay (256x256)

驗證重點：
- 比較 3 vs 4：resize 後邊界是否還是銳利、對齊
- 比較 4 vs 5：手動 resize 與 DataLoader 結果是否一致
"""

import os
import sys
import numpy as np
from PIL import Image
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.dataset import get_dataloader


def create_overlay(image_pil, mask_pil, color=(0, 100, 255), alpha=0.5):
    """
    Create overlay: mask region with semi-transparent color on image
    
    Args:
        image_pil: PIL Image (RGB)
        mask_pil: PIL Image (L, grayscale) - values 0 or 255
        color: RGB tuple for mask region (sky)
        alpha: transparency (0=transparent, 1=opaque)
    
    Returns:
        PIL Image with overlay
    """
    # Convert to numpy
    image_arr = np.array(image_pil).astype(np.float32)
    mask_arr = np.array(mask_pil).astype(np.float32) / 255.0  # normalize to 0-1
    
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


def create_overlay_from_tensor(image_tensor, mask_tensor, color=(0, 100, 255), alpha=0.5):
    """
    Create overlay from PyTorch tensors (DataLoader output)
    
    Args:
        image_tensor: [C, H, W], float32, range [0, 1]
        mask_tensor: [1, H, W], float32, range [0, 1]
    
    Returns:
        PIL Image with overlay
    """
    # Convert image tensor to numpy: [C, H, W] -> [H, W, C]
    image_arr = image_tensor.permute(1, 2, 0).numpy()
    image_arr = (image_arr * 255).clip(0, 255).astype(np.float32)
    
    # Convert mask tensor to numpy: [1, H, W] -> [H, W]
    mask_arr = mask_tensor.squeeze(0).numpy()  # already 0-1
    
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


def manual_resize(image_pil, mask_pil, target_size=(256, 256)):
    """
    Manually resize image and mask
    - Image: BILINEAR interpolation
    - Mask: NEAREST interpolation (critical!)
    
    Returns:
        resized_image, resized_mask
    """
    resized_image = image_pil.resize(target_size, Image.BILINEAR)
    resized_mask = mask_pil.resize(target_size, Image.NEAREST)
    return resized_image, resized_mask


def main():
    # Paths
    images_dir = 'data/toy_v2/images'
    masks_dir = 'data/toy_v2/masks'
    output_dir = 'outputs/toy_loader_vis_v2'
    
    os.makedirs(output_dir, exist_ok=True)
    
    print('=' * 60)
    print('  Toy Dataset V2 Visual Verification')
    print('=' * 60)
    print()
    print(f'Input:  {images_dir} (512x384)')
    print(f'Output: {output_dir}')
    print()
    
    # Get list of files
    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.jpg')])
    num_samples = min(10, len(image_files))  # Process up to 10 samples
    
    print(f'Processing {num_samples} samples...')
    print()
    
    # Create DataLoader for comparison
    print('[1] Loading DataLoader...')
    dataloader = get_dataloader(
        images_dir=images_dir,
        masks_dir=masks_dir,
        batch_size=num_samples,
        shuffle=False,
        transform=False  # No augmentation
    )
    
    # Get one batch from DataLoader
    dataloader_images, dataloader_masks = next(iter(dataloader))
    print(f'    DataLoader output shape: image={dataloader_images.shape}, mask={dataloader_masks.shape}')
    print()
    
    # Process each sample
    print('[2] Generating visualizations...')
    print('-' * 60)
    
    for idx in range(num_samples):
        sample_id = f'{idx+1:03d}'
        filename = image_files[idx]
        base_name = os.path.splitext(filename)[0]
        
        # === Read original files directly ===
        original_image = Image.open(os.path.join(images_dir, f'{base_name}.jpg')).convert('RGB')
        original_mask = Image.open(os.path.join(masks_dir, f'{base_name}.png')).convert('L')
        
        orig_size = original_image.size  # (W, H)
        
        # === 1. Save original image ===
        original_image.save(os.path.join(output_dir, f'{sample_id}_1_original.png'))
        
        # === 2. Save original mask ===
        original_mask.save(os.path.join(output_dir, f'{sample_id}_2_mask.png'))
        
        # === 3. Create overlay at original size ===
        overlay_original = create_overlay(original_image, original_mask)
        overlay_original.save(os.path.join(output_dir, f'{sample_id}_3_overlay_original.png'))
        
        # === 4. Manual resize + overlay ===
        resized_image, resized_mask = manual_resize(original_image, original_mask, (256, 256))
        overlay_resized = create_overlay(resized_image, resized_mask)
        overlay_resized.save(os.path.join(output_dir, f'{sample_id}_4_overlay_resized.png'))
        
        # === 5. DataLoader output overlay ===
        image_tensor = dataloader_images[idx]
        mask_tensor = dataloader_masks[idx]
        overlay_dataloader = create_overlay_from_tensor(image_tensor, mask_tensor)
        overlay_dataloader.save(os.path.join(output_dir, f'{sample_id}_5_from_dataloader.png'))
        
        print(f'    Sample {sample_id}: {orig_size[0]}x{orig_size[1]} -> 256x256, saved 5 images')
    
    print('-' * 60)
    print()
    
    # Summary
    print('[3] Output files:')
    print()
    print('    For each sample XXX:')
    print('    -------------------------------------------------------')
    print('    XXX_1_original.png        : Original image (512x384)')
    print('    XXX_2_mask.png            : Original mask (512x384)')
    print('    XXX_3_overlay_original.png: Overlay at original size')
    print('    XXX_4_overlay_resized.png : Overlay after manual resize')
    print('    XXX_5_from_dataloader.png : Overlay from DataLoader')
    print()
    print('=' * 60)
    print()
    
    # Verification guide
    print('=' * 60)
    print('  HOW TO VERIFY')
    print('=' * 60)
    print()
    print('[Compare XXX_3 vs XXX_4] Resize Quality')
    print('  - Both should have SHARP boundaries (no blur)')
    print('  - Blue overlay should cover the same sky region')
    print('  - If boundary is blurry in XXX_4: resize used wrong interpolation')
    print()
    print('[Compare XXX_4 vs XXX_5] DataLoader Correctness')
    print('  - Both should look IDENTICAL')
    print('  - If different: DataLoader has a bug')
    print()
    print('[Common Problems]')
    print('  1. BLURRY EDGE: Mask resize used BILINEAR instead of NEAREST')
    print('  2. SHIFTED: Image/mask mismatch in DataLoader')
    print('  3. DIFFERENT: DataLoader does unexpected processing')
    print()
    print('=' * 60)


if __name__ == '__main__':
    main()
