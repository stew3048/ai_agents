"""
檢查實際有效的圖片數量（排除受損的）
"""

import os
from PIL import Image

cameras = ['10066', '10870', '9112', '9291', '9483']
total_valid = 0
total_with_mask = 0

for cam in cameras:
    images_dir = f'data/skyfinder_{cam}/images'
    masks_dir = f'data/skyfinder_{cam}/masks'
    
    if not os.path.exists(images_dir):
        continue
    
    valid_images = []
    for filename in os.listdir(images_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(images_dir, filename)
            try:
                img = Image.open(img_path)
                img.verify()
                img.close()
                img = Image.open(img_path)
                img.load()
                img.close()
                valid_images.append(filename)
            except:
                pass
    
    # 檢查哪些有對應的 mask
    valid_pairs = 0
    for filename in valid_images:
        base_name = os.path.splitext(filename)[0]
        mask_found = False
        for ext in ['.png', '.jpg', '.jpeg']:
            mask_path = os.path.join(masks_dir, base_name + ext)
            if os.path.exists(mask_path):
                # 也檢查 mask 是否有效
                try:
                    mask = Image.open(mask_path)
                    mask.verify()
                    mask.close()
                    mask = Image.open(mask_path)
                    mask.load()
                    mask.close()
                    mask_found = True
                    break
                except:
                    pass
        
        if mask_found:
            valid_pairs += 1
    
    print(f"Camera {cam}: {len(valid_images)} valid images, {valid_pairs} valid pairs")
    total_valid += len(valid_images)
    total_with_mask += valid_pairs

print(f"\nTotal valid images: {total_valid}")
print(f"Total valid pairs (with mask): {total_with_mask}")
