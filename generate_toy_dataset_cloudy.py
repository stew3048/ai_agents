"""
產生有雲的藍天測試資料集
測試重點：天空有雲（白色/灰色），看 CV baseline 會不會把雲漏掉

雲的特性：
- 白色或灰色
- 不規則形狀
- 可能部分遮住藍天
- 但仍然是「天空」的一部分，mask 應該是 1
"""

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

# 輸出目錄
OUTPUT_DIR = "data/toy_cloudy"

# 圖片尺寸
WIDTH = 256
HEIGHT = 256

# === 天空類型（都是藍天，但有不同程度的雲） ===
SKY_TYPES = {
    'clear_blue': {
        'name': '晴朗藍天（無雲）',
        'sky_color': (135, 180, 230),
        'cloud_coverage': 0.0,  # 雲覆蓋率
    },
    'few_clouds': {
        'name': '少量雲（10-20%）',
        'sky_color': (130, 175, 225),
        'cloud_coverage': 0.15,
    },
    'scattered_clouds': {
        'name': '散佈雲（30-40%）',
        'sky_color': (125, 170, 220),
        'cloud_coverage': 0.35,
    },
    'broken_clouds': {
        'name': '多雲（50-60%）',
        'sky_color': (120, 165, 215),
        'cloud_coverage': 0.55,
    },
    'overcast_partial': {
        'name': '大部分多雲（70-80%）',
        'sky_color': (115, 160, 210),
        'cloud_coverage': 0.75,
    },
}

# === 雲的類型 ===
CLOUD_TYPES = {
    'cumulus': {
        'name': '積雲（蓬鬆白雲）',
        'color': (255, 255, 255),
        'variation': 10,
        'shape': 'puffy',
    },
    'stratus': {
        'name': '層雲（薄層雲）',
        'color': (240, 240, 245),
        'variation': 15,
        'shape': 'layer',
    },
    'cirrus': {
        'name': '卷雲（絲狀雲）',
        'color': (250, 250, 255),
        'variation': 5,
        'shape': 'wispy',
    },
    'gray_cloud': {
        'name': '灰色雲',
        'color': (200, 200, 205),
        'variation': 20,
        'shape': 'puffy',
    },
    'dark_cloud': {
        'name': '深灰雲（可能下雨）',
        'color': (160, 160, 165),
        'variation': 15,
        'shape': 'puffy',
    },
}

# === 地面類型（非藍色） ===
GROUND_TYPES = {
    'grass': {'name': '草地', 'color': (80, 140, 70), 'variation': 20},
    'earth': {'name': '土地', 'color': (140, 100, 60), 'variation': 15},
    'forest': {'name': '森林', 'color': (40, 90, 40), 'variation': 15},
    'city': {'name': '城市', 'color': (130, 130, 135), 'variation': 20},
}


def add_variation(color, variation):
    """為顏色加入隨機變化"""
    r = np.clip(color[0] + np.random.randint(-variation, variation+1), 0, 255)
    g = np.clip(color[1] + np.random.randint(-variation, variation+1), 0, 255)
    b = np.clip(color[2] + np.random.randint(-variation, variation+1), 0, 255)
    return (int(r), int(g), int(b))


def draw_puffy_cloud(draw, cx, cy, size, color):
    """繪製蓬鬆的積雲（多個重疊的橢圓）"""
    num_blobs = np.random.randint(4, 8)
    
    for _ in range(num_blobs):
        # 隨機偏移
        offset_x = np.random.randint(-size//2, size//2)
        offset_y = np.random.randint(-size//3, size//3)
        
        # 隨機大小
        blob_w = np.random.randint(size//3, size)
        blob_h = np.random.randint(size//4, size//2)
        
        x1 = cx + offset_x - blob_w//2
        y1 = cy + offset_y - blob_h//2
        x2 = x1 + blob_w
        y2 = y1 + blob_h
        
        # 稍微變化顏色
        blob_color = add_variation(color, 10)
        draw.ellipse([x1, y1, x2, y2], fill=blob_color)


def draw_layer_cloud(draw, cx, cy, width, height, color):
    """繪製層雲（水平拉長的橢圓）"""
    # 主體
    x1 = cx - width//2
    y1 = cy - height//2
    x2 = cx + width//2
    y2 = cy + height//2
    draw.ellipse([x1, y1, x2, y2], fill=color)
    
    # 添加一些小的變化
    for _ in range(3):
        offset_x = np.random.randint(-width//3, width//3)
        offset_y = np.random.randint(-height//4, height//4)
        small_w = np.random.randint(width//3, width//2)
        small_h = np.random.randint(height//3, height//2)
        
        sx1 = cx + offset_x - small_w//2
        sy1 = cy + offset_y - small_h//2
        sx2 = sx1 + small_w
        sy2 = sy1 + small_h
        
        blob_color = add_variation(color, 8)
        draw.ellipse([sx1, sy1, sx2, sy2], fill=blob_color)


def draw_wispy_cloud(draw, cx, cy, length, color):
    """繪製卷雲（細長的絲狀）"""
    # 繪製多條細線
    num_wisps = np.random.randint(3, 6)
    
    for _ in range(num_wisps):
        start_x = cx + np.random.randint(-length//3, length//3)
        start_y = cy + np.random.randint(-20, 20)
        
        # 曲線方向
        angle = np.random.uniform(-0.3, 0.3)
        
        points = []
        for i in range(20):
            x = start_x + i * length // 20
            y = start_y + int(np.sin(i * 0.3 + angle) * 10)
            points.append((x, y))
        
        # 繪製漸變的線
        for i in range(len(points) - 1):
            thickness = max(1, 5 - i // 4)
            wisp_color = add_variation(color, 5)
            draw.line([points[i], points[i+1]], fill=wisp_color, width=thickness)


def generate_cloudy_image_and_mask(
    sky_type=None,
    cloud_type=None,
    ground_type=None,
    sky_ratio=0.5
):
    """產生有雲的天空圖片和對應的 mask"""
    
    # 隨機選擇類型
    if sky_type is None:
        sky_type = np.random.choice(list(SKY_TYPES.keys()))
    if cloud_type is None:
        cloud_type = np.random.choice(list(CLOUD_TYPES.keys()))
    if ground_type is None:
        ground_type = np.random.choice(list(GROUND_TYPES.keys()))
    
    sky_info = SKY_TYPES[sky_type]
    cloud_info = CLOUD_TYPES[cloud_type]
    ground_info = GROUND_TYPES[ground_type]
    
    # 建立圖片
    image = Image.new('RGB', (WIDTH, HEIGHT))
    draw = ImageDraw.Draw(image)
    
    # 天空/地面分界線
    boundary_y = int(HEIGHT * sky_ratio)
    
    # 繪製天空背景（藍色）
    sky_color = add_variation(sky_info['sky_color'], 10)
    draw.rectangle([0, 0, WIDTH, boundary_y], fill=sky_color)
    
    # 繪製地面
    ground_color = add_variation(ground_info['color'], ground_info['variation'])
    draw.rectangle([0, boundary_y, WIDTH, HEIGHT], fill=ground_color)
    
    # 添加雲
    cloud_coverage = sky_info['cloud_coverage']
    if cloud_coverage > 0:
        # 計算要繪製多少雲
        num_clouds = int(cloud_coverage * 10) + np.random.randint(0, 3)
        
        for _ in range(num_clouds):
            # 雲的位置（只在天空區域）
            cx = np.random.randint(20, WIDTH - 20)
            cy = np.random.randint(20, boundary_y - 20)
            
            # 雲的大小
            cloud_size = np.random.randint(30, 80)
            
            # 雲的顏色
            cloud_color = add_variation(cloud_info['color'], cloud_info['variation'])
            
            # 根據雲的類型繪製
            if cloud_info['shape'] == 'puffy':
                draw_puffy_cloud(draw, cx, cy, cloud_size, cloud_color)
            elif cloud_info['shape'] == 'layer':
                draw_layer_cloud(draw, cx, cy, cloud_size * 2, cloud_size // 2, cloud_color)
            elif cloud_info['shape'] == 'wispy':
                draw_wispy_cloud(draw, cx, cy, cloud_size * 2, cloud_color)
    
    # 稍微模糊化讓雲更自然
    if cloud_coverage > 0:
        image = image.filter(ImageFilter.GaussianBlur(radius=1))
    
    # 建立 mask（天空=1, 地面=0）
    # 注意：雲也是天空的一部分！
    mask = Image.new('L', (WIDTH, HEIGHT), 0)
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.rectangle([0, 0, WIDTH, boundary_y], fill=255)  # 整個天空區域（包含雲）
    
    return image, mask, {
        'sky_type': sky_type,
        'sky_name': sky_info['name'],
        'cloud_type': cloud_type,
        'cloud_name': cloud_info['name'],
        'cloud_coverage': cloud_coverage,
        'ground_type': ground_type,
        'ground_name': ground_info['name'],
        'boundary_y': boundary_y,
    }


def generate_toy_dataset_cloudy(num_images=50):
    """產生有雲的測試資料集"""
    
    # 建立目錄
    images_dir = os.path.join(OUTPUT_DIR, "images")
    masks_dir = os.path.join(OUTPUT_DIR, "masks")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)
    
    print("=" * 60)
    print("  Generating Cloudy Sky Dataset")
    print("=" * 60)
    print()
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Total images: {num_images}")
    print()
    
    # 確保每種天空類型都有足夠的樣本
    sky_types = list(SKY_TYPES.keys())
    cloud_types = list(CLOUD_TYPES.keys())
    
    # 記錄生成資訊
    info_records = []
    
    for i in range(num_images):
        # 輪流使用不同的天空類型
        sky_type = sky_types[i % len(sky_types)]
        # 隨機雲的類型
        cloud_type = np.random.choice(cloud_types)
        
        image, mask, info = generate_cloudy_image_and_mask(
            sky_type=sky_type,
            cloud_type=cloud_type,
        )
        
        # 儲存
        image_path = os.path.join(images_dir, f"{i+1:03d}.jpg")
        mask_path = os.path.join(masks_dir, f"{i+1:03d}.png")
        
        image.save(image_path, quality=95)
        mask.save(mask_path)
        
        info['image_id'] = f"{i+1:03d}"
        info_records.append(info)
        
        print(f"  [{i+1:03d}] {info['sky_name']} + {info['cloud_name']} (coverage: {info['cloud_coverage']*100:.0f}%)")
    
    # 儲存資訊 CSV
    import csv
    csv_path = os.path.join(OUTPUT_DIR, "dataset_info.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['image_id', 'sky_type', 'sky_name', 'cloud_type', 'cloud_name', 
                      'cloud_coverage', 'ground_type', 'ground_name']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in info_records:
            writer.writerow({k: record[k] for k in fieldnames})
    
    print()
    print("=" * 60)
    print("  Dataset Generated!")
    print("=" * 60)
    print()
    print(f"  Images: {images_dir}/ ({num_images} files)")
    print(f"  Masks:  {masks_dir}/ ({num_images} files)")
    print(f"  Info:   {csv_path}")
    print()
    
    # 統計
    print("  Sky type distribution:")
    for st in sky_types:
        count = sum(1 for r in info_records if r['sky_type'] == st)
        coverage = SKY_TYPES[st]['cloud_coverage']
        print(f"    {SKY_TYPES[st]['name']}: {count} images (cloud: {coverage*100:.0f}%)")
    print()


if __name__ == "__main__":
    generate_toy_dataset_cloudy(50)
