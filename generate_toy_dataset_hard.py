"""
Toy Dataset Hard Mode 產生器
產生各種「難搞」的天空場景，用來測試 CV baseline 的極限

破壞點：
1. 天空顏色變化：灰(雨天)、白、淡藍(陰天)、夜晚灰、黃昏橘色
2. 亮度變化：偏暗、偏亮
3. 地面顏色：可能接近天空顏色

目的：觀察 CV baseline 在哪些情況會失敗
"""

import os
import random
import numpy as np
from PIL import Image


# === 定義各種天空類型 ===
SKY_TYPES = {
    'blue_normal': {
        'name': '正常藍天',
        'sky_color': (135, 180, 230),   # 標準藍色
        'variation': 20,
    },
    'blue_light': {
        'name': '淡藍天（陰天）',
        'sky_color': (180, 200, 220),   # 淡藍偏灰
        'variation': 15,
    },
    'gray_rainy': {
        'name': '灰天（雨天）',
        'sky_color': (150, 150, 155),   # 灰色
        'variation': 10,
    },
    'white_overcast': {
        'name': '白天（多雲）',
        'sky_color': (220, 220, 225),   # 接近白色
        'variation': 10,
    },
    'orange_sunset': {
        'name': '黃昏橘色',
        'sky_color': (230, 150, 100),   # 橘色
        'variation': 25,
    },
    'pink_sunset': {
        'name': '黃昏粉紅',
        'sky_color': (220, 140, 160),   # 粉紅色
        'variation': 20,
    },
    'dark_night': {
        'name': '夜晚深藍',
        'sky_color': (30, 40, 70),      # 深藍近黑
        'variation': 10,
    },
    'gray_night': {
        'name': '夜晚灰',
        'sky_color': (50, 50, 55),      # 灰黑
        'variation': 10,
    },
    'blue_dark': {
        'name': '陰暗藍天',
        'sky_color': (80, 100, 140),    # 較暗的藍色
        'variation': 15,
    },
    'yellow_haze': {
        'name': '霧霾黃',
        'sky_color': (200, 190, 150),   # 霧霾感
        'variation': 15,
    },
}

# === 定義各種地面類型 ===
GROUND_TYPES = {
    'green_grass': {
        'name': '綠色草地',
        'color': (50, 120, 50),
        'variation': 20,
    },
    'brown_earth': {
        'name': '棕色土地',
        'color': (120, 90, 60),
        'variation': 15,
    },
    'gray_city': {
        'name': '灰色城市',
        'color': (100, 100, 105),
        'variation': 15,
    },
    'dark_forest': {
        'name': '深色森林',
        'color': (30, 60, 30),
        'variation': 15,
    },
    'blue_water': {
        'name': '藍色水面',  # 這會讓 CV baseline 誤判！
        'color': (70, 130, 180),
        'variation': 20,
    },
}


def generate_hard_image_and_mask(
    width=256,
    height=256,
    sky_type=None,
    ground_type=None,
    sky_ratio_range=(0.3, 0.7),
    curve_enabled=True
):
    """
    產生一張「難搞」的天空場景
    
    Args:
        width, height: 圖片尺寸
        sky_type: 天空類型（None 則隨機選）
        ground_type: 地面類型（None 則隨機選）
        sky_ratio_range: 天空佔比範圍
        curve_enabled: 是否使用曲線邊界
    
    Returns:
        image: PIL Image (RGB)
        mask: PIL Image (L)
        info: dict，包含這張圖的資訊
    """
    # 隨機選擇天空和地面類型
    if sky_type is None:
        sky_type = random.choice(list(SKY_TYPES.keys()))
    if ground_type is None:
        ground_type = random.choice(list(GROUND_TYPES.keys()))
    
    sky_config = SKY_TYPES[sky_type]
    ground_config = GROUND_TYPES[ground_type]
    
    # 取得基礎顏色並加入隨機變化
    sky_color = np.array(sky_config['sky_color'])
    sky_var = sky_config['variation']
    sky_color = sky_color + np.random.randint(-sky_var, sky_var+1, 3)
    sky_color = np.clip(sky_color, 0, 255)
    
    ground_color = np.array(ground_config['color'])
    ground_var = ground_config['variation']
    ground_color = ground_color + np.random.randint(-ground_var, ground_var+1, 3)
    ground_color = np.clip(ground_color, 0, 255)
    
    # 決定天空佔比
    sky_ratio = random.uniform(*sky_ratio_range)
    sky_height = int(height * sky_ratio)
    
    # 建立圖片陣列
    image_array = np.zeros((height, width, 3), dtype=np.uint8)
    mask_array = np.zeros((height, width), dtype=np.uint8)
    
    # 產生邊界線
    if curve_enabled and random.random() > 0.3:
        boundary = _generate_curved_boundary(width, sky_height, height)
    else:
        boundary = np.full(width, sky_height, dtype=np.int32)
    
    # 填充圖片和 mask
    for x in range(width):
        b = boundary[x]
        # 天空部分
        image_array[:b, x] = sky_color
        mask_array[:b, x] = 255
        # 地面部分
        image_array[b:, x] = ground_color
        mask_array[b:, x] = 0
    
    # 加入雜訊
    noise = np.random.randint(-10, 10, (height, width, 3), dtype=np.int16)
    image_array = np.clip(image_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # 轉換為 PIL Image
    image = Image.fromarray(image_array)
    mask = Image.fromarray(mask_array)
    
    info = {
        'sky_type': sky_type,
        'sky_name': sky_config['name'],
        'ground_type': ground_type,
        'ground_name': ground_config['name'],
        'sky_ratio': sky_ratio,
    }
    
    return image, mask, info


def _generate_curved_boundary(width, base_height, max_height):
    """產生曲線邊界"""
    x = np.arange(width)
    curve = np.zeros(width, dtype=np.float32)
    
    num_waves = random.randint(2, 4)
    for _ in range(num_waves):
        frequency = random.uniform(0.005, 0.03)
        amplitude = random.uniform(10, 40)
        phase = random.uniform(0, 2 * np.pi)
        curve += amplitude * np.sin(frequency * x + phase)
    
    boundary = (base_height + curve).astype(np.int32)
    boundary = np.clip(boundary, 10, max_height - 10)
    
    return boundary


def generate_toy_dataset_hard(
    output_dir,
    num_images=20,
    image_size=(256, 256),
    seed=123
):
    """
    產生 hard mode 的 toy dataset
    
    確保每種天空類型至少出現一次
    """
    random.seed(seed)
    np.random.seed(seed)
    
    images_dir = os.path.join(output_dir, 'images')
    masks_dir = os.path.join(output_dir, 'masks')
    
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)
    
    print('=' * 60)
    print('  Toy Dataset HARD MODE Generator')
    print('=' * 60)
    print()
    print(f'Output: {output_dir}')
    print(f'Size:   {image_size[0]} x {image_size[1]}')
    print(f'Count:  {num_images}')
    print()
    
    # 確保每種天空類型至少出現一次
    sky_types_list = list(SKY_TYPES.keys())
    ground_types_list = list(GROUND_TYPES.keys())
    
    # 先生成每種天空類型各一張
    assigned_sky_types = sky_types_list.copy()
    # 剩餘的隨機分配
    remaining = num_images - len(assigned_sky_types)
    if remaining > 0:
        assigned_sky_types += random.choices(sky_types_list, k=remaining)
    random.shuffle(assigned_sky_types)
    
    print('-' * 60)
    print('Generating images...')
    print('-' * 60)
    
    all_info = []
    
    for i in range(num_images):
        sky_type = assigned_sky_types[i] if i < len(assigned_sky_types) else random.choice(sky_types_list)
        ground_type = random.choice(ground_types_list)
        
        image, mask, info = generate_hard_image_and_mask(
            width=image_size[0],
            height=image_size[1],
            sky_type=sky_type,
            ground_type=ground_type
        )
        
        filename = f"{i+1:03d}"
        image.save(os.path.join(images_dir, f"{filename}.jpg"), 'JPEG', quality=95)
        mask.save(os.path.join(masks_dir, f"{filename}.png"), 'PNG')
        
        info['sample_id'] = filename
        all_info.append(info)
        
        print(f"  {filename}: {info['sky_name']} + {info['ground_name']}")
    
    print('-' * 60)
    print()
    
    # 儲存資訊到 CSV
    info_csv = os.path.join(output_dir, 'dataset_info.csv')
    import csv
    with open(info_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['sample_id', 'sky_type', 'sky_name', 'ground_type', 'ground_name', 'sky_ratio'])
        writer.writeheader()
        for info in all_info:
            writer.writerow({
                'sample_id': info['sample_id'],
                'sky_type': info['sky_type'],
                'sky_name': info['sky_name'],
                'ground_type': info['ground_type'],
                'ground_name': info['ground_name'],
                'sky_ratio': f"{info['sky_ratio']:.2f}"
            })
    
    print(f'Dataset info saved to: {info_csv}')
    print()
    
    # 統計
    print('Sky type distribution:')
    from collections import Counter
    sky_counts = Counter([info['sky_type'] for info in all_info])
    for sky_type, count in sky_counts.items():
        print(f"  {SKY_TYPES[sky_type]['name']}: {count}")
    
    print()
    print('=' * 60)
    print('Done!')
    print('=' * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate Toy Dataset HARD MODE')
    parser.add_argument('--output_dir', type=str, default='data/toy_hard',
                        help='Output directory (default: data/toy_hard)')
    parser.add_argument('--num_images', type=int, default=20,
                        help='Number of images (default: 20)')
    parser.add_argument('--seed', type=int, default=123,
                        help='Random seed (default: 123)')
    
    args = parser.parse_args()
    
    generate_toy_dataset_hard(
        output_dir=args.output_dir,
        num_images=args.num_images,
        seed=args.seed
    )
