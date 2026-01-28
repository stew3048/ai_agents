"""
新增圖片到 diagnostic_candidates.csv

小心保留用戶已標記的欄位，只新增指定的圖片
"""

import os
import sys
import csv
import glob
from pathlib import Path
from PIL import Image
import torch
import torchvision.transforms.functional as TF
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

from utils.image_utils import mean_luma_linear


def process_image_for_csv(image_path, mask_path, camera_id, image_id, 
                          light_override=None, weather_override=None):
    """
    處理單張圖片，計算自動標籤
    如果提供 light_override 或 weather_override，則使用該值
    """
    try:
        # 讀取圖片
        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        
        # 轉換為 tensor 用於計算
        image_tensor = TF.to_tensor(image)
        
        # 檢查 has_sky
        mask_array = np.array(mask)
        has_sky = bool(np.any(mask_array > 128))
        
        # 計算 pred_sky_area_ratio
        sky_pixels = np.sum(mask_array > 128)
        total_pixels = mask_array.size
        pred_sky_area_ratio = float(sky_pixels / total_pixels) if total_pixels > 0 else 0.0
        
        # 構建相對路徑
        rel_path = os.path.relpath(image_path, 'data')
        
        # 使用 override 或自動計算
        if light_override:
            light = light_override
        else:
            luma = mean_luma_linear(image_tensor)
            if luma < 0.15:
                light = 'night'
            elif luma < 0.30:
                light = 'dusk'
            else:
                light = 'day'
        
        if weather_override:
            weather = weather_override
        else:
            weather = 'unknown'  # 預設為 unknown，讓用戶之後補
        
        return {
            'camera_id': camera_id,
            'image_id': str(image_id),  # 轉為字串，與 CSV 格式一致
            'path': rel_path,
            'has_sky': 'TRUE' if has_sky else 'FALSE',
            'sea_sky_confusable': '0',  # 預設為 0
            'scene(sea|urban|forest|other)': 'unknown',  # 預設，讓用戶補
            'occlusion(none|partial|heavy)': 'unknown',  # 預設，讓用戶補
            'weather(clear|cloudy|rain|fog|snow|unknown)': weather,
            'light (day|dusk|night)': light,
            'pred_sky_area_ratio': f"{pred_sky_area_ratio:.6f}",
            'notes': ''
        }
    except Exception as e:
        print(f"  [錯誤] 處理 {image_path} 失敗: {e}")
        return None


def find_image_and_mask(camera_id, image_id):
    """尋找指定 camera 和 image_id 的圖片和 mask"""
    base_dir = 'data'
    camera_dir = os.path.join(base_dir, f'skyfinder_{camera_id}')
    images_dir = os.path.join(camera_dir, 'images')
    masks_dir = os.path.join(camera_dir, 'masks')
    
    # 嘗試不同的檔名格式
    image_id_str = str(image_id).zfill(3)  # 補零到 3 位數
    
    # 尋找圖片
    image_path = None
    for ext in ['.jpg', '.jpeg', '.png']:
        potential_image = os.path.join(images_dir, f"{image_id_str}{ext}")
        if os.path.exists(potential_image):
            image_path = potential_image
            break
    
    # 尋找 mask
    mask_path = None
    for ext in ['.png', '.jpg', '.jpeg']:
        potential_mask = os.path.join(masks_dir, f"{image_id_str}{ext}")
        if os.path.exists(potential_mask):
            mask_path = potential_mask
            break
    
    return image_path, mask_path


def main():
    csv_file = 'data/test_data/diagnostic_candidates.csv'
    
    # 讀取現有的 CSV
    print("讀取現有 CSV...")
    existing_rows = []
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            existing_rows.append(row)
    
    print(f"  現有 {len(existing_rows)} 筆資料")
    
    # 檢查要新增的圖片是否已存在
    existing_keys = set()
    for row in existing_rows:
        key = (row['camera_id'], row['image_id'])
        existing_keys.add(key)
    
    # 要新增的圖片
    new_images = [
        ('10066', '41', 'dusk', None),  # light=dusk
        ('10066', '46', None, 'fog'),   # weather=fog
        ('10066', '47', None, 'fog'),   # weather=fog
        ('9483', '91', None, 'fog'),    # weather=fog
        ('9483', '92', None, 'fog'),    # weather=fog
    ]
    
    print("\n處理要新增的圖片...")
    new_rows = []
    for camera_id, image_id, light_override, weather_override in new_images:
        key = (camera_id, image_id)
        if key in existing_keys:
            print(f"  [跳過] {camera_id}/{image_id}: 已存在")
            continue
        
        # 尋找圖片和 mask
        image_path, mask_path = find_image_and_mask(camera_id, image_id)
        
        if not image_path or not mask_path:
            print(f"  [錯誤] {camera_id}/{image_id}: 找不到圖片或 mask")
            print(f"    image_path: {image_path}")
            print(f"    mask_path: {mask_path}")
            continue
        
        # 處理圖片
        result = process_image_for_csv(
            image_path, mask_path, camera_id, image_id,
            light_override=light_override,
            weather_override=weather_override
        )
        
        if result:
            new_rows.append(result)
            print(f"  [OK] {camera_id}/{image_id}: light={result['light (day|dusk|night)']}, weather={result['weather(clear|cloudy|rain|fog|snow|unknown)']}")
    
    if not new_rows:
        print("\n沒有新圖片需要新增")
        return
    
    # 合併現有和新增的資料
    all_rows = existing_rows + new_rows
    
    # 寫回 CSV
    print(f"\n寫入 CSV（總共 {len(all_rows)} 筆）...")
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)
    
    print(f"  已新增 {len(new_rows)} 筆資料")
    print(f"  總計 {len(all_rows)} 筆資料")
    
    # 生成縮圖
    print("\n生成縮圖 preview...")
    preview_dir = 'data/test_data/previews'
    thumbnail_size = 256
    
    for row in new_rows:
        camera_id = row['camera_id']
        image_id = row['image_id']
        image_path = os.path.join('data', row['path'])
        
        if not os.path.exists(image_path):
            continue
        
        # 建立輸出資料夾
        camera_preview_dir = os.path.join(preview_dir, f'camera_{camera_id}')
        os.makedirs(camera_preview_dir, exist_ok=True)
        
        try:
            # 讀取圖片
            image = Image.open(image_path).convert('RGB')
            
            # 生成縮圖（保持長寬比）
            image.thumbnail((thumbnail_size, thumbnail_size), Image.Resampling.LANCZOS)
            
            # 建立正方形畫布（白色背景）
            thumb = Image.new('RGB', (thumbnail_size, thumbnail_size), (255, 255, 255))
            
            # 置中貼上縮圖
            x = (thumbnail_size - image.width) // 2
            y = (thumbnail_size - image.height) // 2
            thumb.paste(image, (x, y))
            
            # 儲存
            output_path = os.path.join(camera_preview_dir, f'{image_id}.jpg')
            thumb.save(output_path, 'JPEG', quality=85)
            print(f"  [OK] {camera_id}/{image_id}.jpg")
        except Exception as e:
            print(f"  [警告] 生成縮圖失敗 {image_path}: {e}")
    
    print("\n完成！")


if __name__ == '__main__':
    main()
