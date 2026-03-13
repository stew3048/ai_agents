"""
Toy Dataset Level 4：多層地面測試
天空下方可能是：雪地、森林、沙漠、海洋等不同顏色的地面

破壞點：
1. 白色雪地（可能與白色天空混淆）
2. 黃色沙漠（可能與黃昏天空混淆）
3. 藍色海洋/湖泊（可能與藍色天空混淆）
4. 多層地面（森林+湖泊、山+雪地）
5. 漸層地面

目的：測試 CV baseline 對「顏色相近的天空與地面」的分辨能力
"""

import os
import random
import numpy as np
from PIL import Image


# === 天空類型 ===
SKY_TYPES = {
    'blue_normal': {
        'name': '正常藍天',
        'sky_color': (135, 180, 230),
        'variation': 20,
    },
    'white_cloudy': {
        'name': '白色多雲',
        'sky_color': (220, 225, 230),
        'variation': 10,
    },
    'orange_sunset': {
        'name': '黃昏橘色',
        'sky_color': (230, 160, 120),
        'variation': 20,
    },
    'gray_overcast': {
        'name': '灰色陰天',
        'sky_color': (160, 165, 170),
        'variation': 15,
    },
}

# === 地面場景類型 ===
GROUND_SCENES = {
    'snow_field': {
        'name': '雪地',
        'description': '白色雪地，可能與白色天空混淆',
        'layers': [
            {'name': '雪', 'color': (240, 245, 250), 'variation': 10, 'ratio': 1.0},
        ],
        'confuse_with': 'white_cloudy',
    },
    'desert': {
        'name': '沙漠',
        'description': '黃色沙漠，可能與黃昏天空混淆',
        'layers': [
            {'name': '沙', 'color': (220, 180, 130), 'variation': 20, 'ratio': 1.0},
        ],
        'confuse_with': 'orange_sunset',
    },
    'ocean': {
        'name': '海洋',
        'description': '藍色海洋，可能與藍色天空混淆',
        'layers': [
            {'name': '海', 'color': (70, 130, 180), 'variation': 20, 'ratio': 1.0},
        ],
        'confuse_with': 'blue_normal',
    },
    'forest_lake': {
        'name': '森林湖泊',
        'description': '綠色森林 + 藍色湖泊',
        'layers': [
            {'name': '森林', 'color': (40, 80, 40), 'variation': 15, 'ratio': 0.6},
            {'name': '湖泊', 'color': (60, 120, 160), 'variation': 15, 'ratio': 0.4},
        ],
        'confuse_with': 'blue_normal',
    },
    'mountain_snow': {
        'name': '雪山',
        'description': '深色山體 + 白色雪頂',
        'layers': [
            {'name': '山體', 'color': (80, 70, 60), 'variation': 15, 'ratio': 0.5},
            {'name': '雪頂', 'color': (235, 240, 245), 'variation': 10, 'ratio': 0.5},
        ],
        'confuse_with': 'white_cloudy',
    },
    'beach': {
        'name': '海灘',
        'description': '黃色沙灘 + 藍色海水',
        'layers': [
            {'name': '沙灘', 'color': (230, 210, 170), 'variation': 15, 'ratio': 0.4},
            {'name': '海水', 'color': (80, 150, 190), 'variation': 20, 'ratio': 0.6},
        ],
        'confuse_with': 'blue_normal',
    },
    'glacier': {
        'name': '冰川',
        'description': '淡藍色冰川，可能與天空混淆',
        'layers': [
            {'name': '冰川', 'color': (200, 220, 240), 'variation': 15, 'ratio': 1.0},
        ],
        'confuse_with': 'blue_normal',
    },
    'gray_city': {
        'name': '灰色城市',
        'description': '灰色建築，可能與灰色天空混淆',
        'layers': [
            {'name': '城市', 'color': (130, 130, 135), 'variation': 20, 'ratio': 1.0},
        ],
        'confuse_with': 'gray_overcast',
    },
}


def generate_level4_image_and_mask(
    width=256,
    height=256,
    sky_type=None,
    ground_scene=None,
    sky_ratio_range=(0.35, 0.55),
    use_confusing_sky=True,
):
    """
    產生 Level 4 測試圖片
    
    Args:
        use_confusing_sky: 如果 True，選擇與地面顏色相近的天空
    """
    
    # 選擇地面場景
    if ground_scene is None:
        ground_scene = random.choice(list(GROUND_SCENES.keys()))
    
    scene_config = GROUND_SCENES[ground_scene]
    
    # 選擇天空類型
    if sky_type is None:
        if use_confusing_sky and 'confuse_with' in scene_config:
            # 50% 機率選擇會混淆的天空
            if random.random() < 0.5:
                sky_type = scene_config['confuse_with']
            else:
                sky_type = random.choice(list(SKY_TYPES.keys()))
        else:
            sky_type = random.choice(list(SKY_TYPES.keys()))
    
    sky_config = SKY_TYPES[sky_type]
    
    # 天空顏色
    sky_color = np.array(sky_config['sky_color'])
    sky_color = sky_color + np.random.randint(-sky_config['variation'], sky_config['variation']+1, 3)
    sky_color = np.clip(sky_color, 0, 255)
    
    # 決定天空高度
    sky_ratio = random.uniform(*sky_ratio_range)
    sky_height = int(height * sky_ratio)
    
    # 建立圖片和 mask
    image_array = np.zeros((height, width, 3), dtype=np.uint8)
    mask_array = np.zeros((height, width), dtype=np.uint8)
    
    # 填充天空
    image_array[:sky_height, :] = sky_color
    mask_array[:sky_height, :] = 255
    
    # 填充地面（可能有多層）
    ground_height = height - sky_height
    layers = scene_config['layers']
    
    if len(layers) == 1:
        # 單層地面
        layer = layers[0]
        ground_color = np.array(layer['color'])
        ground_color = ground_color + np.random.randint(-layer['variation'], layer['variation']+1, 3)
        ground_color = np.clip(ground_color, 0, 255)
        image_array[sky_height:, :] = ground_color
    else:
        # 多層地面
        current_y = sky_height
        for i, layer in enumerate(layers):
            layer_height = int(ground_height * layer['ratio'])
            if i == len(layers) - 1:
                # 最後一層填滿剩餘空間
                layer_height = height - current_y
            
            ground_color = np.array(layer['color'])
            ground_color = ground_color + np.random.randint(-layer['variation'], layer['variation']+1, 3)
            ground_color = np.clip(ground_color, 0, 255)
            
            end_y = min(current_y + layer_height, height)
            image_array[current_y:end_y, :] = ground_color
            current_y = end_y
    
    # 地面的 mask 都是 0（非天空）
    mask_array[sky_height:, :] = 0
    
    # 加入雜訊
    noise = np.random.randint(-8, 8, (height, width, 3), dtype=np.int16)
    image_array = np.clip(image_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    image = Image.fromarray(image_array)
    mask = Image.fromarray(mask_array)
    
    # 計算顏色相似度（用於分析）
    # 取地面第一層顏色與天空顏色的歐氏距離
    first_layer_color = np.array(layers[0]['color'])
    color_distance = np.sqrt(np.sum((sky_color - first_layer_color) ** 2))
    
    info = {
        'sky_type': sky_type,
        'sky_name': sky_config['name'],
        'ground_scene': ground_scene,
        'ground_name': scene_config['name'],
        'ground_description': scene_config['description'],
        'sky_ratio': sky_ratio,
        'color_distance': color_distance,
        'is_confusing': sky_type == scene_config.get('confuse_with', ''),
    }
    
    return image, mask, info


def generate_toy_dataset_level4(
    output_dir,
    num_images=50,
    image_size=(256, 256),
    seed=999
):
    """產生 Level 4 dataset"""
    random.seed(seed)
    np.random.seed(seed)
    
    images_dir = os.path.join(output_dir, 'images')
    masks_dir = os.path.join(output_dir, 'masks')
    
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)
    
    print('=' * 60)
    print('  Toy Dataset LEVEL 4: Multi-terrain Ground')
    print('=' * 60)
    print()
    print(f'Output: {output_dir}')
    print(f'Size:   {image_size[0]} x {image_size[1]}')
    print(f'Count:  {num_images}')
    print()
    print('Ground scenes (with confusing sky):')
    for scene, config in GROUND_SCENES.items():
        confuse = config.get('confuse_with', 'N/A')
        print(f"  - {config['name']}: {config['description']}")
        print(f"    -> Confuses with: {SKY_TYPES.get(confuse, {}).get('name', 'N/A')}")
    print()
    
    print('-' * 60)
    print('Generating images...')
    print('-' * 60)
    
    all_info = []
    scene_types_list = list(GROUND_SCENES.keys())
    
    for i in range(num_images):
        # 確保每種場景都有出現
        if i < len(scene_types_list):
            ground_scene = scene_types_list[i]
        else:
            ground_scene = random.choice(scene_types_list)
        
        image, mask, info = generate_level4_image_and_mask(
            width=image_size[0],
            height=image_size[1],
            ground_scene=ground_scene,
            use_confusing_sky=True,
        )
        
        filename = f"{i+1:03d}"
        image.save(os.path.join(images_dir, f"{filename}.jpg"), 'JPEG', quality=95)
        mask.save(os.path.join(masks_dir, f"{filename}.png"), 'PNG')
        
        info['sample_id'] = filename
        all_info.append(info)
        
        confuse_mark = '*' if info['is_confusing'] else ' '
        print(f"  {filename}: {info['sky_name']:10} + {info['ground_name']:10} | dist={info['color_distance']:5.1f} {confuse_mark}")
    
    print('-' * 60)
    print('  (* = sky-ground color confusion pair)')
    print()
    
    # 儲存 CSV
    import csv
    info_csv = os.path.join(output_dir, 'dataset_info.csv')
    with open(info_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'sample_id', 'sky_type', 'sky_name', 'ground_scene', 'ground_name',
            'sky_ratio', 'color_distance', 'is_confusing'
        ])
        writer.writeheader()
        for info in all_info:
            writer.writerow({
                'sample_id': info['sample_id'],
                'sky_type': info['sky_type'],
                'sky_name': info['sky_name'],
                'ground_scene': info['ground_scene'],
                'ground_name': info['ground_name'],
                'sky_ratio': f"{info['sky_ratio']:.2f}",
                'color_distance': f"{info['color_distance']:.1f}",
                'is_confusing': info['is_confusing'],
            })
    
    print(f'Dataset info saved to: {info_csv}')
    print()
    
    # 統計
    from collections import Counter
    scene_counts = Counter([info['ground_scene'] for info in all_info])
    confusing_count = sum(1 for info in all_info if info['is_confusing'])
    
    print('Ground scene distribution:')
    for scene, count in scene_counts.most_common():
        print(f"  {GROUND_SCENES[scene]['name']}: {count}")
    
    print()
    print(f'Confusing pairs: {confusing_count} / {num_images} ({confusing_count/num_images*100:.1f}%)')
    
    # 顏色距離統計
    distances = [info['color_distance'] for info in all_info]
    print(f'Color distance: min={min(distances):.1f}, max={max(distances):.1f}, avg={np.mean(distances):.1f}')
    
    print()
    print('=' * 60)
    print('Done!')
    print('=' * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate Toy Dataset Level 4')
    parser.add_argument('--output_dir', type=str, default='data/toy_level4',
                        help='Output directory')
    parser.add_argument('--num_images', type=int, default=50,
                        help='Number of images')
    parser.add_argument('--seed', type=int, default=999,
                        help='Random seed')
    
    args = parser.parse_args()
    
    generate_toy_dataset_level4(
        output_dir=args.output_dir,
        num_images=args.num_images,
        seed=args.seed
    )
