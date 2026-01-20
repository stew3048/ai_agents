"""
Toy Dataset Level 2：地面遮蔽物測試
在地面區域加入藍色物體（矩形、紋理），測試 CV baseline 的誤判問題

破壞點：
1. 藍色矩形建築物
2. 藍色水池/湖泊
3. 藍色車輛
4. 藍色廣告牌
5. 藍色紋理區域

目的：CV baseline 會把這些藍色物體誤判為天空
"""

import os
import random
import numpy as np
from PIL import Image


# === 天空類型（這次用正常藍天，確保天空能被正確偵測） ===
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
}

# === 藍色遮蔽物類型（這些會讓 CV baseline 誤判！） ===
BLUE_OBJECTS = {
    'blue_building': {
        'name': '藍色建築',
        'color': (70, 130, 200),
        'variation': 20,
        'shape': 'rectangle',
        'size_range': (0.1, 0.25),  # 相對於圖片的大小比例
    },
    'blue_pool': {
        'name': '藍色水池',
        'color': (80, 150, 210),
        'variation': 15,
        'shape': 'ellipse',
        'size_range': (0.1, 0.2),
    },
    'blue_car': {
        'name': '藍色車輛',
        'color': (50, 100, 180),
        'variation': 15,
        'shape': 'rectangle',
        'size_range': (0.05, 0.12),
    },
    'blue_billboard': {
        'name': '藍色廣告牌',
        'color': (60, 120, 200),
        'variation': 20,
        'shape': 'rectangle',
        'size_range': (0.08, 0.18),
    },
    'blue_texture': {
        'name': '藍色紋理區',
        'color': (90, 140, 190),
        'variation': 25,
        'shape': 'irregular',
        'size_range': (0.1, 0.2),
    },
    'cyan_object': {
        'name': '青色物體',
        'color': (100, 180, 200),
        'variation': 20,
        'shape': 'rectangle',
        'size_range': (0.08, 0.15),
    },
    'light_blue_tent': {
        'name': '淺藍帳篷',
        'color': (150, 190, 230),
        'variation': 15,
        'shape': 'triangle',
        'size_range': (0.1, 0.2),
    },
}


def draw_rectangle(array, x, y, w, h, color):
    """繪製矩形"""
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(array.shape[1], x + w), min(array.shape[0], y + h)
    array[y1:y2, x1:x2] = color


def draw_ellipse(array, cx, cy, rx, ry, color):
    """繪製橢圓"""
    for y in range(max(0, cy - ry), min(array.shape[0], cy + ry)):
        for x in range(max(0, cx - rx), min(array.shape[1], cx + rx)):
            if ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1:
                array[y, x] = color


def draw_triangle(array, x, y, w, h, color):
    """繪製三角形（帳篷形狀）"""
    for row in range(h):
        # 計算這一行的寬度
        progress = row / h
        left = int(x + w * 0.5 * (1 - progress))
        right = int(x + w * 0.5 * (1 + progress))
        y_pos = y + row
        if 0 <= y_pos < array.shape[0]:
            left = max(0, left)
            right = min(array.shape[1], right)
            array[y_pos, left:right] = color


def draw_irregular(array, x, y, w, h, color, seed=None):
    """繪製不規則形狀（用多個重疊圓形）"""
    if seed is not None:
        np.random.seed(seed)
    
    num_circles = random.randint(5, 10)
    for _ in range(num_circles):
        cx = x + random.randint(0, w)
        cy = y + random.randint(0, h)
        r = random.randint(min(w, h) // 6, min(w, h) // 3)
        draw_ellipse(array, cx, cy, r, r, color)


def generate_level2_image_and_mask(
    width=256,
    height=256,
    sky_type=None,
    ground_type=None,
    blue_objects_list=None,
    num_objects_range=(1, 3),
    sky_ratio_range=(0.3, 0.5),
):
    """
    產生 Level 2 測試圖片
    
    Args:
        blue_objects_list: 要放置的藍色物體類型列表（None 則隨機）
        num_objects_range: 物體數量範圍
    
    Returns:
        image, mask, info
    """
    # 選擇天空和地面
    if sky_type is None:
        sky_type = random.choice(list(SKY_TYPES.keys()))
    if ground_type is None:
        ground_type = random.choice(list(GROUND_TYPES.keys()))
    
    sky_config = SKY_TYPES[sky_type]
    ground_config = GROUND_TYPES[ground_type]
    
    # 取得顏色
    sky_color = np.array(sky_config['sky_color'])
    sky_color = sky_color + np.random.randint(-sky_config['variation'], sky_config['variation']+1, 3)
    sky_color = np.clip(sky_color, 0, 255)
    
    ground_color = np.array(ground_config['color'])
    ground_color = ground_color + np.random.randint(-ground_config['variation'], ground_config['variation']+1, 3)
    ground_color = np.clip(ground_color, 0, 255)
    
    # 決定天空高度
    sky_ratio = random.uniform(*sky_ratio_range)
    sky_height = int(height * sky_ratio)
    
    # 建立基礎圖片和 mask
    image_array = np.zeros((height, width, 3), dtype=np.uint8)
    mask_array = np.zeros((height, width), dtype=np.uint8)
    
    # 填充天空
    image_array[:sky_height, :] = sky_color
    mask_array[:sky_height, :] = 255  # 天空 = 255
    
    # 填充地面
    image_array[sky_height:, :] = ground_color
    mask_array[sky_height:, :] = 0  # 地面 = 0
    
    # 決定要放置的藍色物體
    num_objects = random.randint(*num_objects_range)
    if blue_objects_list is None:
        blue_objects_list = random.choices(list(BLUE_OBJECTS.keys()), k=num_objects)
    
    placed_objects = []
    
    for obj_type in blue_objects_list:
        obj_config = BLUE_OBJECTS[obj_type]
        
        # 計算物體大小
        size_ratio = random.uniform(*obj_config['size_range'])
        obj_w = int(width * size_ratio)
        obj_h = int(height * size_ratio * random.uniform(0.5, 1.5))
        
        # 決定物體位置（必須在地面區域）
        ground_height = height - sky_height
        max_y = height - obj_h - 5
        min_y = sky_height + 5
        
        if max_y <= min_y:
            continue
            
        obj_x = random.randint(5, width - obj_w - 5)
        obj_y = random.randint(min_y, max_y)
        
        # 取得物體顏色
        obj_color = np.array(obj_config['color'])
        obj_color = obj_color + np.random.randint(-obj_config['variation'], obj_config['variation']+1, 3)
        obj_color = np.clip(obj_color, 0, 255)
        
        # 繪製物體（只在 image 上，mask 保持為地面=0）
        shape = obj_config['shape']
        if shape == 'rectangle':
            draw_rectangle(image_array, obj_x, obj_y, obj_w, obj_h, obj_color)
        elif shape == 'ellipse':
            draw_ellipse(image_array, obj_x + obj_w//2, obj_y + obj_h//2, obj_w//2, obj_h//2, obj_color)
        elif shape == 'triangle':
            draw_triangle(image_array, obj_x, obj_y, obj_w, obj_h, obj_color)
        elif shape == 'irregular':
            draw_irregular(image_array, obj_x, obj_y, obj_w, obj_h, obj_color)
        
        placed_objects.append({
            'type': obj_type,
            'name': obj_config['name'],
            'x': obj_x,
            'y': obj_y,
            'w': obj_w,
            'h': obj_h,
        })
    
    # 加入雜訊
    noise = np.random.randint(-8, 8, (height, width, 3), dtype=np.int16)
    image_array = np.clip(image_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # 轉換為 PIL
    image = Image.fromarray(image_array)
    mask = Image.fromarray(mask_array)
    
    info = {
        'sky_type': sky_type,
        'sky_name': sky_config['name'],
        'ground_type': ground_type,
        'ground_name': ground_config['name'],
        'sky_ratio': sky_ratio,
        'blue_objects': placed_objects,
        'num_blue_objects': len(placed_objects),
    }
    
    return image, mask, info


def generate_toy_dataset_level2(
    output_dir,
    num_images=50,
    image_size=(256, 256),
    seed=456
):
    """產生 Level 2 dataset"""
    random.seed(seed)
    np.random.seed(seed)
    
    images_dir = os.path.join(output_dir, 'images')
    masks_dir = os.path.join(output_dir, 'masks')
    
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)
    
    print('=' * 60)
    print('  Toy Dataset LEVEL 2: Blue Objects on Ground')
    print('=' * 60)
    print()
    print(f'Output: {output_dir}')
    print(f'Size:   {image_size[0]} x {image_size[1]}')
    print(f'Count:  {num_images}')
    print()
    print('Blue object types:')
    for obj_type, config in BLUE_OBJECTS.items():
        print(f"  - {config['name']} ({obj_type})")
    print()
    
    print('-' * 60)
    print('Generating images...')
    print('-' * 60)
    
    all_info = []
    object_type_list = list(BLUE_OBJECTS.keys())
    
    for i in range(num_images):
        # 確保每種藍色物體都有出現
        if i < len(object_type_list):
            # 前幾張確保每種物體至少出現一次
            blue_objects_list = [object_type_list[i % len(object_type_list)]]
        else:
            # 之後隨機選擇 1-3 個物體
            blue_objects_list = None
        
        image, mask, info = generate_level2_image_and_mask(
            width=image_size[0],
            height=image_size[1],
            blue_objects_list=blue_objects_list,
            num_objects_range=(1, 3),
        )
        
        filename = f"{i+1:03d}"
        image.save(os.path.join(images_dir, f"{filename}.jpg"), 'JPEG', quality=95)
        mask.save(os.path.join(masks_dir, f"{filename}.png"), 'PNG')
        
        info['sample_id'] = filename
        all_info.append(info)
        
        obj_names = [obj['name'] for obj in info['blue_objects']]
        print(f"  {filename}: {info['sky_name']} | Objects: {', '.join(obj_names)}")
    
    print('-' * 60)
    print()
    
    # 儲存 CSV
    import csv
    info_csv = os.path.join(output_dir, 'dataset_info.csv')
    with open(info_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'sample_id', 'sky_type', 'sky_name', 'ground_type', 'ground_name',
            'sky_ratio', 'num_blue_objects', 'blue_object_types'
        ])
        writer.writeheader()
        for info in all_info:
            obj_types = ';'.join([obj['type'] for obj in info['blue_objects']])
            writer.writerow({
                'sample_id': info['sample_id'],
                'sky_type': info['sky_type'],
                'sky_name': info['sky_name'],
                'ground_type': info['ground_type'],
                'ground_name': info['ground_name'],
                'sky_ratio': f"{info['sky_ratio']:.2f}",
                'num_blue_objects': info['num_blue_objects'],
                'blue_object_types': obj_types,
            })
    
    print(f'Dataset info saved to: {info_csv}')
    print()
    
    # 統計
    from collections import Counter
    all_objects = []
    for info in all_info:
        for obj in info['blue_objects']:
            all_objects.append(obj['name'])
    
    print('Blue object distribution:')
    for name, count in Counter(all_objects).most_common():
        print(f"  {name}: {count}")
    
    print()
    print('=' * 60)
    print('Done!')
    print('=' * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate Toy Dataset Level 2')
    parser.add_argument('--output_dir', type=str, default='data/toy_level2',
                        help='Output directory')
    parser.add_argument('--num_images', type=int, default=50,
                        help='Number of images')
    parser.add_argument('--seed', type=int, default=456,
                        help='Random seed')
    
    args = parser.parse_args()
    
    generate_toy_dataset_level2(
        output_dir=args.output_dir,
        num_images=args.num_images,
        seed=args.seed
    )
