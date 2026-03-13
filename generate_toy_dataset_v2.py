"""
Toy Dataset V2 產生器
產生較大尺寸 (512x384) 的模擬天空場景，用於驗證 resize 行為

格式：與現有 SkySegmentationDataset 相容
- 圖片：RGB .jpg 格式
- Mask：灰階 .png 格式（0=非天空, 255=天空）
- 檔名一致（例如 001.jpg 對應 001.png）
"""

import os
import random
import numpy as np
from PIL import Image


def generate_sky_image_and_mask(
    width=512, 
    height=384, 
    sky_ratio_range=(0.3, 0.7),
    curve_enabled=True
):
    """
    產生一張模擬天空場景的圖片與對應的 mask
    
    參數:
        width: 圖片寬度
        height: 圖片高度
        sky_ratio_range: 天空佔畫面的比例範圍 (min, max)
        curve_enabled: 是否使用曲線邊界（模擬山脈/建築輪廓）
    
    返回:
        image: PIL Image (RGB)
        mask: PIL Image (L, 灰階)
    """
    # 隨機決定天空佔比
    sky_ratio = random.uniform(*sky_ratio_range)
    sky_height = int(height * sky_ratio)
    
    # 產生天空顏色（藍色系，有變化）
    sky_r = random.randint(100, 180)
    sky_g = random.randint(150, 220)
    sky_b = random.randint(200, 255)
    
    # 產生地面顏色（綠色/棕色系）
    ground_type = random.choice(['green', 'brown', 'gray'])
    if ground_type == 'green':
        ground_r = random.randint(30, 80)
        ground_g = random.randint(80, 150)
        ground_b = random.randint(30, 80)
    elif ground_type == 'brown':
        ground_r = random.randint(100, 150)
        ground_g = random.randint(70, 120)
        ground_b = random.randint(40, 80)
    else:  # gray (城市)
        ground_r = random.randint(80, 130)
        ground_g = random.randint(80, 130)
        ground_b = random.randint(80, 130)
    
    # 建立圖片陣列
    image_array = np.zeros((height, width, 3), dtype=np.uint8)
    mask_array = np.zeros((height, width), dtype=np.uint8)
    
    # 產生邊界線（直線或曲線）
    if curve_enabled and random.random() > 0.3:
        # 使用曲線邊界（模擬山脈）
        boundary = _generate_curved_boundary(width, sky_height, height)
    else:
        # 使用直線邊界
        boundary = np.full(width, sky_height, dtype=np.int32)
    
    # 填充圖片和 mask
    for x in range(width):
        b = boundary[x]
        # 天空部分
        image_array[:b, x] = [sky_r, sky_g, sky_b]
        mask_array[:b, x] = 255  # 天空 = 255
        # 地面部分
        image_array[b:, x] = [ground_r, ground_g, ground_b]
        mask_array[b:, x] = 0  # 非天空 = 0
    
    # 加入一些雜訊讓圖片更自然
    noise = np.random.randint(-15, 15, (height, width, 3), dtype=np.int16)
    image_array = np.clip(image_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # 轉換為 PIL Image
    image = Image.fromarray(image_array)
    mask = Image.fromarray(mask_array)
    
    return image, mask


def _generate_curved_boundary(width, base_height, max_height):
    """
    產生曲線邊界，模擬山脈或建築輪廓
    
    參數:
        width: 圖片寬度
        base_height: 基準高度
        max_height: 圖片最大高度
    
    返回:
        boundary: 每個 x 座標對應的邊界 y 值
    """
    boundary = np.zeros(width, dtype=np.int32)
    
    # 使用多個 sin 波疊加產生自然曲線
    x = np.arange(width)
    curve = np.zeros(width, dtype=np.float32)
    
    # 加入 2-4 個不同頻率的波
    num_waves = random.randint(2, 4)
    for _ in range(num_waves):
        frequency = random.uniform(0.005, 0.03)
        amplitude = random.uniform(15, 60)  # 增加振幅讓邊界更明顯
        phase = random.uniform(0, 2 * np.pi)
        curve += amplitude * np.sin(frequency * x + phase)
    
    # 計算最終邊界
    boundary = (base_height + curve).astype(np.int32)
    
    # 確保邊界在有效範圍內
    boundary = np.clip(boundary, 20, max_height - 20)
    
    return boundary


def generate_toy_dataset_v2(
    output_dir,
    num_images=20,
    image_size=(512, 384),  # 較大尺寸 (W, H)
    seed=42
):
    """
    產生完整的 toy dataset v2
    
    參數:
        output_dir: 輸出資料夾路徑
        num_images: 產生的圖片數量
        image_size: 圖片尺寸 (width, height)
        seed: 隨機種子（確保可重現）
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # 建立資料夾結構
    images_dir = os.path.join(output_dir, 'images')
    masks_dir = os.path.join(output_dir, 'masks')
    
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)
    
    print('=' * 60)
    print('  Toy Dataset V2 Generator')
    print('=' * 60)
    print()
    print(f'Output path: {output_dir}')
    print(f'Image size:  {image_size[0]} x {image_size[1]} (W x H)')
    print(f'Num images:  {num_images}')
    print()
    print('-' * 60)
    
    for i in range(1, num_images + 1):
        # 產生圖片和 mask
        image, mask = generate_sky_image_and_mask(
            width=image_size[0],
            height=image_size[1]
        )
        
        # 檔名格式：三位數字
        filename = f"{i:03d}"
        
        # 儲存圖片 (.jpg)
        image_path = os.path.join(images_dir, f"{filename}.jpg")
        image.save(image_path, 'JPEG', quality=95)
        
        # 儲存 mask (.png)
        mask_path = os.path.join(masks_dir, f"{filename}.png")
        mask.save(mask_path, 'PNG')
        
        # 進度顯示
        if i % 5 == 0 or i == num_images:
            print(f'  Generated: {i}/{num_images}')
    
    print()
    print('-' * 60)
    print(f'Done! Generated {num_images} image-mask pairs')
    print(f'  Images: {images_dir}')
    print(f'  Masks:  {masks_dir}')
    print('=' * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate Toy Dataset V2 (larger size)')
    parser.add_argument('--output_dir', type=str, default='data/toy_v2',
                        help='Output directory (default: data/toy_v2)')
    parser.add_argument('--num_images', type=int, default=20,
                        help='Number of images to generate (default: 20)')
    parser.add_argument('--width', type=int, default=512,
                        help='Image width (default: 512)')
    parser.add_argument('--height', type=int, default=384,
                        help='Image height (default: 384)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    generate_toy_dataset_v2(
        output_dir=args.output_dir,
        num_images=args.num_images,
        image_size=(args.width, args.height),
        seed=args.seed
    )
