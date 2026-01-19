"""
生成假的天空測試資料
用於驗證 pipeline 是否正常運作
"""

import os
import numpy as np
from PIL import Image, ImageDraw
import random


def generate_sky_image(width=256, height=256):
    """
    生成一張模擬天空場景的圖片
    
    返回:
        image: PIL Image (RGB)
        mask: PIL Image (L, 灰階)
    """
    # 隨機決定地平線高度 (40% - 60% 的位置)
    horizon_ratio = random.uniform(0.4, 0.6)
    horizon_y = int(height * horizon_ratio)
    
    # 創建圖片陣列
    img_array = np.zeros((height, width, 3), dtype=np.uint8)
    mask_array = np.zeros((height, width), dtype=np.uint8)
    
    # ========== 天空區域 ==========
    # 基礎天空顏色 (淺藍到深藍的漸層)
    sky_top = np.array([random.randint(100, 150), random.randint(180, 220), random.randint(230, 255)])  # 淺藍
    sky_bottom = np.array([random.randint(135, 180), random.randint(200, 230), random.randint(235, 255)])  # 較深藍
    
    for y in range(horizon_y):
        # 線性漸層
        ratio = y / max(horizon_y - 1, 1)
        color = (sky_top * (1 - ratio) + sky_bottom * ratio).astype(np.uint8)
        img_array[y, :] = color
    
    # 天空區域的 mask = 255 (白色)
    mask_array[:horizon_y, :] = 255
    
    # ========== 地面區域 ==========
    # 隨機選擇地面類型
    ground_type = random.choice(['green', 'brown', 'gray'])
    if ground_type == 'green':
        ground_color = np.array([random.randint(30, 60), random.randint(120, 160), random.randint(30, 60)])
    elif ground_type == 'brown':
        ground_color = np.array([random.randint(100, 140), random.randint(80, 110), random.randint(50, 80)])
    else:
        ground_color = np.array([random.randint(100, 140), random.randint(100, 140), random.randint(100, 140)])
    
    # 填充地面
    for y in range(horizon_y, height):
        # 添加一些變化
        noise = np.random.randint(-10, 10, 3)
        color = np.clip(ground_color + noise, 0, 255).astype(np.uint8)
        img_array[y, :] = color
    
    # 地面區域的 mask = 0 (黑色) - 已經是預設值
    
    # ========== 添加雲朵 ==========
    image = Image.fromarray(img_array)
    draw = ImageDraw.Draw(image)
    
    num_clouds = random.randint(1, 4)
    for _ in range(num_clouds):
        # 雲朵位置（只在天空區域）
        cloud_x = random.randint(0, width)
        cloud_y = random.randint(10, horizon_y - 30)
        
        # 雲朵大小
        cloud_w = random.randint(30, 80)
        cloud_h = random.randint(15, 35)
        
        # 繪製橢圓形雲朵（白色，半透明效果）
        cloud_color = (255, 255, 255)
        draw.ellipse([cloud_x - cloud_w//2, cloud_y - cloud_h//2, 
                      cloud_x + cloud_w//2, cloud_y + cloud_h//2], 
                     fill=cloud_color)
    
    # 轉換回陣列並創建 mask
    mask = Image.fromarray(mask_array)
    
    return image, mask


def generate_dataset(output_dir, num_train=8, num_val=2, image_size=(256, 256)):
    """
    生成完整的訓練和驗證資料集
    
    參數:
        output_dir: 輸出目錄 (應該是 data/ 資料夾)
        num_train: 訓練集圖片數量
        num_val: 驗證集圖片數量
        image_size: 圖片尺寸 (width, height)
    """
    # 定義路徑
    train_images_dir = os.path.join(output_dir, 'train', 'images')
    train_masks_dir = os.path.join(output_dir, 'train', 'masks')
    val_images_dir = os.path.join(output_dir, 'val', 'images')
    val_masks_dir = os.path.join(output_dir, 'val', 'masks')
    
    # 確保目錄存在
    for dir_path in [train_images_dir, train_masks_dir, val_images_dir, val_masks_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    print("=" * 60)
    print("生成假的天空測試資料")
    print("=" * 60)
    
    # 生成訓練集
    print(f"\n生成訓練集 ({num_train} 張)...")
    for i in range(num_train):
        image, mask = generate_sky_image(image_size[0], image_size[1])
        
        # 儲存圖片
        image_path = os.path.join(train_images_dir, f'sky_{i:03d}.png')
        mask_path = os.path.join(train_masks_dir, f'sky_{i:03d}.png')
        
        image.save(image_path)
        mask.save(mask_path)
        
        print(f"  [OK] {image_path}")
    
    # 生成驗證集
    print(f"\n生成驗證集 ({num_val} 張)...")
    for i in range(num_val):
        image, mask = generate_sky_image(image_size[0], image_size[1])
        
        # 儲存圖片
        image_path = os.path.join(val_images_dir, f'sky_{i:03d}.png')
        mask_path = os.path.join(val_masks_dir, f'sky_{i:03d}.png')
        
        image.save(image_path)
        mask.save(mask_path)
        
        print(f"  [OK] {image_path}")
    
    print("\n" + "=" * 60)
    print("[OK] 資料集生成完成！")
    print("=" * 60)
    print(f"\n資料集統計:")
    print(f"  訓練集: {num_train} 張")
    print(f"  驗證集: {num_val} 張")
    print(f"  圖片尺寸: {image_size[0]}x{image_size[1]}")
    print(f"\n資料位置:")
    print(f"  訓練圖片: {train_images_dir}")
    print(f"  訓練 mask: {train_masks_dir}")
    print(f"  驗證圖片: {val_images_dir}")
    print(f"  驗證 mask: {val_masks_dir}")


if __name__ == '__main__':
    # 生成資料集
    generate_dataset(
        output_dir='data',
        num_train=8,
        num_val=2,
        image_size=(256, 256)
    )
