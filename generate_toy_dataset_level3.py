"""
Toy Dataset Level 3：複雜邊界測試
天空區域形狀變複雜：山脈、樹梢、建築物輪廓

破壞點：
1. 山脈輪廓（多個山峰）
2. 樹梢剪影（鋸齒狀）
3. 建築物天際線
4. 波浪狀邊界
5. 混合輪廓

目的：測試 CV baseline 能否正確處理非水平的天空邊界
（理論上應該沒問題，因為還是靠顏色判斷）
"""

import os
import random
import numpy as np
from PIL import Image


# === 天空類型（用正常藍天） ===
SKY_TYPES = {
    'blue_normal': {
        'name': '正常藍天',
        'sky_color': (135, 180, 230),
        'variation': 20,
    },
    'blue_light': {
        'name': '淡藍天',
        'sky_color': (160, 195, 225),
        'variation': 15,
    },
}

# === 地面類型 ===
GROUND_TYPES = {
    'green_forest': {
        'name': '綠色森林',
        'color': (40, 90, 40),
        'variation': 20,
    },
    'dark_mountain': {
        'name': '深色山脈',
        'color': (60, 70, 60),
        'variation': 15,
    },
    'gray_city': {
        'name': '灰色城市',
        'color': (90, 90, 95),
        'variation': 15,
    },
}

# === 邊界類型 ===
BOUNDARY_TYPES = {
    'mountain_single': {
        'name': '單峰山脈',
        'description': '一座山峰',
    },
    'mountain_multi': {
        'name': '多峰山脈',
        'description': '多座山峰連綿',
    },
    'tree_silhouette': {
        'name': '樹梢剪影',
        'description': '鋸齒狀樹梢',
    },
    'building_skyline': {
        'name': '建築天際線',
        'description': '方形建築輪廓',
    },
    'wave': {
        'name': '波浪邊界',
        'description': '平滑波浪',
    },
    'jagged': {
        'name': '鋸齒邊界',
        'description': '不規則鋸齒',
    },
    'mixed': {
        'name': '混合輪廓',
        'description': '山+樹+建築',
    },
}


def generate_mountain_single(width, base_height, height):
    """產生單峰山脈邊界"""
    x = np.arange(width)
    
    # 山峰位置和高度
    peak_x = width * random.uniform(0.3, 0.7)
    peak_height = random.randint(50, 100)
    spread = width * random.uniform(0.3, 0.5)
    
    # 高斯形狀的山峰
    mountain = peak_height * np.exp(-((x - peak_x) ** 2) / (2 * spread ** 2))
    
    boundary = (base_height - mountain).astype(np.int32)
    return np.clip(boundary, 20, height - 20)


def generate_mountain_multi(width, base_height, height):
    """產生多峰山脈邊界"""
    x = np.arange(width)
    boundary = np.full(width, base_height, dtype=np.float32)
    
    # 產生 2-4 個山峰
    num_peaks = random.randint(2, 4)
    for i in range(num_peaks):
        peak_x = width * (0.1 + 0.8 * i / num_peaks) + random.randint(-20, 20)
        peak_height = random.randint(30, 80)
        spread = width * random.uniform(0.1, 0.2)
        
        mountain = peak_height * np.exp(-((x - peak_x) ** 2) / (2 * spread ** 2))
        boundary = boundary - mountain
    
    return np.clip(boundary.astype(np.int32), 20, height - 20)


def generate_tree_silhouette(width, base_height, height):
    """產生樹梢剪影（鋸齒狀）"""
    boundary = np.full(width, base_height, dtype=np.float32)
    
    # 產生多棵樹的輪廓
    x = 0
    while x < width:
        # 樹的寬度和高度
        tree_width = random.randint(15, 40)
        tree_height = random.randint(20, 60)
        
        # 三角形樹冠
        for i in range(tree_width):
            if x + i < width:
                # 三角形：中間最高
                progress = abs(i - tree_width / 2) / (tree_width / 2)
                h = tree_height * (1 - progress)
                boundary[x + i] = min(boundary[x + i], base_height - h)
        
        # 隨機間隔
        x += tree_width + random.randint(5, 20)
    
    # 加入小幅度隨機鋸齒
    noise = np.random.randint(-5, 5, width)
    boundary = boundary + noise
    
    return np.clip(boundary.astype(np.int32), 20, height - 20)


def generate_building_skyline(width, base_height, height):
    """產生建築天際線"""
    boundary = np.full(width, base_height, dtype=np.float32)
    
    x = 0
    while x < width:
        # 建築的寬度和高度
        building_width = random.randint(20, 50)
        building_height = random.randint(30, 90)
        
        # 矩形建築
        end_x = min(x + building_width, width)
        boundary[x:end_x] = base_height - building_height
        
        # 間隔（可能有或沒有）
        gap = random.randint(0, 15)
        x = end_x + gap
    
    return np.clip(boundary.astype(np.int32), 20, height - 20)


def generate_wave_boundary(width, base_height, height):
    """產生波浪邊界"""
    x = np.arange(width)
    
    # 多個正弦波疊加
    wave = np.zeros(width, dtype=np.float32)
    for _ in range(random.randint(2, 4)):
        freq = random.uniform(0.01, 0.04)
        amp = random.uniform(15, 40)
        phase = random.uniform(0, 2 * np.pi)
        wave += amp * np.sin(freq * x + phase)
    
    boundary = base_height - wave
    return np.clip(boundary.astype(np.int32), 20, height - 20)


def generate_jagged_boundary(width, base_height, height):
    """產生鋸齒邊界"""
    boundary = np.full(width, base_height, dtype=np.float32)
    
    # 產生隨機鋸齒
    x = 0
    while x < width:
        segment_width = random.randint(5, 20)
        segment_height = random.randint(-40, 40)
        
        end_x = min(x + segment_width, width)
        boundary[x:end_x] = base_height - segment_height
        
        x = end_x
    
    # 平滑化
    kernel_size = 3
    kernel = np.ones(kernel_size) / kernel_size
    boundary = np.convolve(boundary, kernel, mode='same')
    
    return np.clip(boundary.astype(np.int32), 20, height - 20)


def generate_mixed_boundary(width, base_height, height):
    """產生混合輪廓（山+樹+建築）"""
    boundary = np.full(width, base_height, dtype=np.float32)
    
    # 左邊：山脈
    left_third = width // 3
    x = np.arange(left_third)
    peak_x = left_third * 0.5
    peak_height = random.randint(40, 70)
    spread = left_third * 0.4
    mountain = peak_height * np.exp(-((x - peak_x) ** 2) / (2 * spread ** 2))
    boundary[:left_third] = base_height - mountain
    
    # 中間：樹
    mid_start = left_third
    mid_end = 2 * width // 3
    x = mid_start
    while x < mid_end:
        tree_width = random.randint(10, 25)
        tree_height = random.randint(20, 50)
        for i in range(tree_width):
            if x + i < mid_end:
                progress = abs(i - tree_width / 2) / (tree_width / 2)
                h = tree_height * (1 - progress)
                boundary[x + i] = min(boundary[x + i], base_height - h)
        x += tree_width + random.randint(3, 10)
    
    # 右邊：建築
    right_start = 2 * width // 3
    x = right_start
    while x < width:
        building_width = random.randint(15, 35)
        building_height = random.randint(30, 70)
        end_x = min(x + building_width, width)
        boundary[x:end_x] = base_height - building_height
        x = end_x + random.randint(0, 10)
    
    return np.clip(boundary.astype(np.int32), 20, height - 20)


BOUNDARY_GENERATORS = {
    'mountain_single': generate_mountain_single,
    'mountain_multi': generate_mountain_multi,
    'tree_silhouette': generate_tree_silhouette,
    'building_skyline': generate_building_skyline,
    'wave': generate_wave_boundary,
    'jagged': generate_jagged_boundary,
    'mixed': generate_mixed_boundary,
}


def generate_level3_image_and_mask(
    width=256,
    height=256,
    sky_type=None,
    ground_type=None,
    boundary_type=None,
    base_sky_ratio=0.5,
):
    """產生 Level 3 測試圖片"""
    
    if sky_type is None:
        sky_type = random.choice(list(SKY_TYPES.keys()))
    if ground_type is None:
        ground_type = random.choice(list(GROUND_TYPES.keys()))
    if boundary_type is None:
        boundary_type = random.choice(list(BOUNDARY_TYPES.keys()))
    
    sky_config = SKY_TYPES[sky_type]
    ground_config = GROUND_TYPES[ground_type]
    boundary_config = BOUNDARY_TYPES[boundary_type]
    
    # 顏色
    sky_color = np.array(sky_config['sky_color'])
    sky_color = sky_color + np.random.randint(-sky_config['variation'], sky_config['variation']+1, 3)
    sky_color = np.clip(sky_color, 0, 255)
    
    ground_color = np.array(ground_config['color'])
    ground_color = ground_color + np.random.randint(-ground_config['variation'], ground_config['variation']+1, 3)
    ground_color = np.clip(ground_color, 0, 255)
    
    # 產生邊界
    base_height = int(height * base_sky_ratio)
    boundary_generator = BOUNDARY_GENERATORS[boundary_type]
    boundary = boundary_generator(width, base_height, height)
    
    # 建立圖片和 mask
    image_array = np.zeros((height, width, 3), dtype=np.uint8)
    mask_array = np.zeros((height, width), dtype=np.uint8)
    
    for x in range(width):
        b = boundary[x]
        # 天空
        image_array[:b, x] = sky_color
        mask_array[:b, x] = 255
        # 地面
        image_array[b:, x] = ground_color
        mask_array[b:, x] = 0
    
    # 加入雜訊
    noise = np.random.randint(-8, 8, (height, width, 3), dtype=np.int16)
    image_array = np.clip(image_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    image = Image.fromarray(image_array)
    mask = Image.fromarray(mask_array)
    
    # 計算實際天空比例
    actual_sky_ratio = mask_array.sum() / 255 / (width * height)
    
    info = {
        'sky_type': sky_type,
        'sky_name': sky_config['name'],
        'ground_type': ground_type,
        'ground_name': ground_config['name'],
        'boundary_type': boundary_type,
        'boundary_name': boundary_config['name'],
        'boundary_description': boundary_config['description'],
        'sky_ratio': actual_sky_ratio,
    }
    
    return image, mask, info


def generate_toy_dataset_level3(
    output_dir,
    num_images=50,
    image_size=(256, 256),
    seed=789
):
    """產生 Level 3 dataset"""
    random.seed(seed)
    np.random.seed(seed)
    
    images_dir = os.path.join(output_dir, 'images')
    masks_dir = os.path.join(output_dir, 'masks')
    
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)
    
    print('=' * 60)
    print('  Toy Dataset LEVEL 3: Complex Boundaries')
    print('=' * 60)
    print()
    print(f'Output: {output_dir}')
    print(f'Size:   {image_size[0]} x {image_size[1]}')
    print(f'Count:  {num_images}')
    print()
    print('Boundary types:')
    for bt, config in BOUNDARY_TYPES.items():
        print(f"  - {config['name']}: {config['description']}")
    print()
    
    print('-' * 60)
    print('Generating images...')
    print('-' * 60)
    
    all_info = []
    boundary_types_list = list(BOUNDARY_TYPES.keys())
    
    for i in range(num_images):
        # 確保每種邊界類型都有出現
        if i < len(boundary_types_list):
            boundary_type = boundary_types_list[i]
        else:
            boundary_type = random.choice(boundary_types_list)
        
        image, mask, info = generate_level3_image_and_mask(
            width=image_size[0],
            height=image_size[1],
            boundary_type=boundary_type,
            base_sky_ratio=random.uniform(0.4, 0.6),
        )
        
        filename = f"{i+1:03d}"
        image.save(os.path.join(images_dir, f"{filename}.jpg"), 'JPEG', quality=95)
        mask.save(os.path.join(masks_dir, f"{filename}.png"), 'PNG')
        
        info['sample_id'] = filename
        all_info.append(info)
        
        print(f"  {filename}: {info['boundary_name']} | sky={info['sky_ratio']*100:.1f}%")
    
    print('-' * 60)
    print()
    
    # 儲存 CSV
    import csv
    info_csv = os.path.join(output_dir, 'dataset_info.csv')
    with open(info_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'sample_id', 'sky_type', 'sky_name', 'ground_type', 'ground_name',
            'boundary_type', 'boundary_name', 'sky_ratio'
        ])
        writer.writeheader()
        for info in all_info:
            writer.writerow({
                'sample_id': info['sample_id'],
                'sky_type': info['sky_type'],
                'sky_name': info['sky_name'],
                'ground_type': info['ground_type'],
                'ground_name': info['ground_name'],
                'boundary_type': info['boundary_type'],
                'boundary_name': info['boundary_name'],
                'sky_ratio': f"{info['sky_ratio']:.2f}",
            })
    
    print(f'Dataset info saved to: {info_csv}')
    print()
    
    # 統計
    from collections import Counter
    boundary_counts = Counter([info['boundary_type'] for info in all_info])
    print('Boundary type distribution:')
    for bt, count in boundary_counts.most_common():
        print(f"  {BOUNDARY_TYPES[bt]['name']}: {count}")
    
    print()
    print('=' * 60)
    print('Done!')
    print('=' * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate Toy Dataset Level 3')
    parser.add_argument('--output_dir', type=str, default='data/toy_level3',
                        help='Output directory')
    parser.add_argument('--num_images', type=int, default=50,
                        help='Number of images')
    parser.add_argument('--seed', type=int, default=789,
                        help='Random seed')
    
    args = parser.parse_args()
    
    generate_toy_dataset_level3(
        output_dir=args.output_dir,
        num_images=args.num_images,
        seed=args.seed
    )
