"""
產生情境診斷資料集與標註腳本

目標：
1. 產生 Hold-out Test 清單（camera 10870，全部 94 張）→ holdout_10870.csv
2. 產生 Diagnostic Scenario Set 清單（從所有 camera 抽樣，80-120 張）→ diagnostic_candidates.csv
3. 自動計算標籤：light, has_sky, pred_sky_area_ratio, weather（嘗試）
4. 生成縮圖 preview（按 camera_id 分組）

執行方式：
  python scripts/create_diagnostic_datasets.py
"""

import os
import sys
import csv
import glob
import random
import numpy as np
from pathlib import Path
from PIL import Image
import torch
import torchvision.transforms.functional as TF
from tqdm import tqdm
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

from utils.image_utils import mean_luma_linear


# Light 判斷閾值
LIGHT_NIGHT_THRESHOLD = 0.15
LIGHT_DUSK_THRESHOLD = 0.30


def infer_light(luma):
    """根據亮度推斷光線條件"""
    if luma < LIGHT_NIGHT_THRESHOLD:
        return 'night'
    elif luma < LIGHT_DUSK_THRESHOLD:
        return 'dusk'
    else:
        return 'day'


def calculate_image_features(image):
    """
    計算圖像特徵用於 weather 判斷
    返回：mean_luma, std_luma, saturation, contrast, edge_density
    """
    # 轉換為 numpy array
    if isinstance(image, Image.Image):
        img_array = np.array(image).astype(np.float32) / 255.0
    else:
        img_array = image
    
    # 計算亮度統計
    if len(img_array.shape) == 3:
        # RGB 轉灰階
        gray = 0.2126 * img_array[:, :, 0] + 0.7152 * img_array[:, :, 1] + 0.0722 * img_array[:, :, 2]
    else:
        gray = img_array
    
    mean_luma = float(np.mean(gray))
    std_luma = float(np.std(gray))
    
    # 計算顏色飽和度（轉換到 HSV）
    if len(img_array.shape) == 3:
        hsv = np.array(Image.fromarray((img_array * 255).astype(np.uint8)).convert('HSV'))
        saturation = float(np.mean(hsv[:, :, 1] / 255.0))
    else:
        saturation = 0.0
    
    # 計算對比度（使用標準差）
    contrast = std_luma
    
    # 簡化的邊緣密度（使用簡單的梯度近似）
    if len(img_array.shape) == 3:
        gray_uint8 = (gray * 255).astype(np.uint8)
    else:
        gray_uint8 = (gray * 255).astype(np.uint8)
    
    # 簡單的邊緣檢測（使用 numpy 的梯度）
    try:
        # 計算水平和垂直梯度
        grad_x = np.abs(np.diff(gray_uint8, axis=1))
        grad_y = np.abs(np.diff(gray_uint8, axis=0))
        # 簡單的邊緣強度
        edge_strength = np.concatenate([grad_x.flatten(), grad_y.flatten()])
        edge_density = float(np.mean(edge_strength > 30) / 255.0)  # 閾值 30
    except:
        edge_density = 0.0
    
    return mean_luma, std_luma, saturation, contrast, edge_density


def infer_weather(image):
    """
    根據圖像特徵粗略判斷天氣
    返回：clear | cloudy | fog | rain | snow | unknown
    """
    try:
        mean_luma, std_luma, saturation, contrast, edge_density = calculate_image_features(image)
        
        # 檢查是否有大量白色區域（snow）
        if mean_luma > 0.6 and saturation < 0.3:
            return 'snow'
        
        # 檢查是否有大量灰色區域且對比度低（fog）
        if 0.3 < mean_luma < 0.6 and contrast < 0.15 and edge_density < 0.1:
            return 'fog'
        
        # 檢查是否為多雲（中等亮度，低飽和度）
        if 0.2 < mean_luma < 0.5 and saturation < 0.4 and contrast < 0.2:
            return 'cloudy'
        
        # 檢查是否為雨天（低亮度，可能有反光）
        if mean_luma < 0.3 and contrast > 0.2:
            return 'rain'
        
        # 檢查是否為晴天（高亮度，高對比度，高飽和度）
        if mean_luma > 0.4 and contrast > 0.2 and saturation > 0.4:
            return 'clear'
        
        # 無法明確判斷
        return 'unknown'
    except Exception as e:
        print(f"  [警告] Weather 判斷失敗: {e}")
        return 'unknown'


def scan_all_cameras(base_dir='data'):
    """
    掃描所有 skyfinder_* 資料夾，收集所有圖片和 mask
    返回：{camera_id: [(image_path, mask_path, image_id), ...]}
    """
    camera_data = defaultdict(list)
    
    # 掃描所有 skyfinder_* 資料夾
    pattern = os.path.join(base_dir, 'skyfinder_*')
    camera_dirs = glob.glob(pattern)
    
    print(f"掃描 camera 資料夾...")
    for camera_dir in sorted(camera_dirs):
        camera_id = os.path.basename(camera_dir).replace('skyfinder_', '')
        
        # 跳過 sample 資料夾（與 10066 相同）
        if camera_id == 'sample':
            print(f"  [跳過] {camera_id}: 與 10066 相同，跳過")
            continue
        
        images_dir = os.path.join(camera_dir, 'images')
        masks_dir = os.path.join(camera_dir, 'masks')
        
        if not os.path.exists(images_dir) or not os.path.exists(masks_dir):
            print(f"  [跳過] {camera_id}: 缺少 images 或 masks 資料夾")
            continue
        
        # 讀取所有圖片檔案
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_files.extend(glob.glob(os.path.join(images_dir, ext)))
        
        count = 0
        for image_path in sorted(image_files):
            image_filename = os.path.basename(image_path)
            image_id = os.path.splitext(image_filename)[0]
            
            # 尋找對應的 mask
            mask_path = None
            for ext in ['.png', '.jpg', '.jpeg']:
                potential_mask = os.path.join(masks_dir, f"{image_id}{ext}")
                if os.path.exists(potential_mask):
                    mask_path = potential_mask
                    break
            
            if mask_path and os.path.exists(mask_path):
                camera_data[camera_id].append((image_path, mask_path, image_id))
                count += 1
        
        print(f"  [OK] {camera_id}: {count} 張圖片")
    
    return camera_data


def process_image(image_path, mask_path, camera_id, image_id):
    """
    處理單張圖片，計算所有自動標籤
    返回：字典包含所有欄位
    """
    try:
        # 讀取圖片
        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        
        # 轉換為 tensor 用於計算
        image_tensor = TF.to_tensor(image)  # [C, H, W], [0, 1]
        
        # 計算 mean_luma
        luma = mean_luma_linear(image_tensor)
        
        # 推斷 light
        light = infer_light(luma)
        
        # 檢查 has_sky
        mask_array = np.array(mask)
        has_sky = bool(np.any(mask_array > 128))  # 假設 mask 中 > 128 為天空
        
        # 計算 pred_sky_area_ratio
        sky_pixels = np.sum(mask_array > 128)
        total_pixels = mask_array.size
        pred_sky_area_ratio = float(sky_pixels / total_pixels) if total_pixels > 0 else 0.0
        
        # 推斷 weather
        weather = infer_weather(image)
        
        # 構建相對路徑
        rel_path = os.path.relpath(image_path, 'data')
        
        return {
            'camera_id': camera_id,
            'image_id': image_id,
            'path': rel_path,
            'has_sky': 'True' if has_sky else 'False',
            'sea_sky_confusable': '',
            'scene': 'unknown',
            'occlusion': 'unknown',
            'weather': weather,
            'light': light,
            'pred_sky_area_ratio': f"{pred_sky_area_ratio:.6f}",
            'notes': ''
        }
    except Exception as e:
        print(f"  [錯誤] 處理 {image_path} 失敗: {e}")
        return None


def create_holdout_test(camera_data, output_dir='data/test_data'):
    """
    產生 Hold-out Test 清單（camera 10870，全部 94 張）
    """
    print("\n" + "=" * 60)
    print("產生 Hold-out Test 清單（camera 10870）")
    print("=" * 60)
    
    camera_id = '10870'
    if camera_id not in camera_data:
        print(f"[錯誤] 找不到 camera {camera_id}")
        return
    
    items = camera_data[camera_id]
    print(f"處理 {len(items)} 張圖片...")
    
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'holdout_10870.csv')
    
    # CSV 欄位
    fieldnames = [
        'camera_id', 'image_id', 'path', 'has_sky', 'sea_sky_confusable',
        'scene(sea|urban|forest|other)', 'occlusion(none|partial|heavy)',
        'weather(clear|cloudy|rain|fog|snow|unknown)', 'light (day|dusk|night)',
        'pred_sky_area_ratio', 'notes'
    ]
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        stats = {'day': 0, 'dusk': 0, 'night': 0}
        
        for image_path, mask_path, image_id in tqdm(items, desc='  處理'):
            result = process_image(image_path, mask_path, camera_id, image_id)
            if result:
                # 調整欄位名稱以符合 CSV header
                row = {
                    'camera_id': result['camera_id'],
                    'image_id': result['image_id'],
                    'path': result['path'],
                    'has_sky': result['has_sky'],
                    'sea_sky_confusable': result['sea_sky_confusable'],
                    'scene(sea|urban|forest|other)': result['scene'],
                    'occlusion(none|partial|heavy)': result['occlusion'],
                    'weather(clear|cloudy|rain|fog|snow|unknown)': result['weather'],
                    'light (day|dusk|night)': result['light'],
                    'pred_sky_area_ratio': result['pred_sky_area_ratio'],
                    'notes': result['notes']
                }
                writer.writerow(row)
                stats[result['light']] += 1
        
        print(f"\n  已寫入: {output_file}")
        print(f"  統計: day={stats['day']}, dusk={stats['dusk']}, night={stats['night']}")


def sample_diagnostic_set(camera_data, target_total=100):
    """
    從所有 camera 抽樣產生 Diagnostic Scenario Set
    策略：確保情境多樣性，偏向抽取少見情境
    """
    print("\n" + "=" * 60)
    print("產生 Diagnostic Scenario Set（從所有 camera 抽樣）")
    print("=" * 60)
    
    # 先計算每個 camera 的亮度分布（包含 10870，但要抽樣）
    camera_stats = {}
    for camera_id, items in camera_data.items():
        # 10870 也要抽樣，不跳過
        print(f"  分析 camera {camera_id}...")
        luma_dist = {'day': [], 'dusk': [], 'night': []}
        
        for image_path, mask_path, image_id in tqdm(items, desc=f'    camera {camera_id}', leave=False):
            try:
                image = Image.open(image_path).convert('RGB')
                image_tensor = TF.to_tensor(image)
                luma = mean_luma_linear(image_tensor)
                light = infer_light(luma)
                luma_dist[light].append((image_path, mask_path, image_id, luma))
            except:
                continue
        
        camera_stats[camera_id] = {
            'total': len(items),
            'day': luma_dist['day'],
            'dusk': luma_dist['dusk'],
            'night': luma_dist['night']
        }
        
        day_count = len(luma_dist['day'])
        dusk_count = len(luma_dist['dusk'])
        night_count = len(luma_dist['night'])
        print(f"    day={day_count}, dusk={dusk_count}, night={night_count}")
    
    # 計算每個 camera 的配額（總量約 100 張）
    camera_quotas = {}
    total_available = sum(len(items) for cid, items in camera_data.items())
    
    # 10870 固定抽 10 張（取代原本 sample 的 10 張）
    if '10870' in camera_stats:
        camera_quotas['10870'] = 10
        target_total -= 10  # 從總量中扣除
        total_available -= camera_stats['10870']['total']
    
    for camera_id, stats in camera_stats.items():
        if camera_id == '10870':
            continue  # 已經設定為 10 張
        
        # 基礎配額：根據總數比例
        base_quota = int(target_total * stats['total'] / total_available)
        # 調整：night 比例高的多抽一些
        night_ratio = len(stats['night']) / stats['total'] if stats['total'] > 0 else 0
        if night_ratio > 0.5:
            base_quota = int(base_quota * 1.2)  # 多抽 20%
        camera_quotas[camera_id] = min(base_quota, stats['total'])
    
    # 調整總量
    total_quota = sum(camera_quotas.values())
    if total_quota > target_total * 1.2:
        # 按比例縮減
        scale = target_total / total_quota
        for camera_id in camera_quotas:
            camera_quotas[camera_id] = int(camera_quotas[camera_id] * scale)
    
    print(f"\n  抽樣配額:")
    for camera_id, quota in camera_quotas.items():
        print(f"    {camera_id}: {quota} 張")
    
    # 從每個 camera 抽樣
    sampled_items = []
    for camera_id, quota in camera_quotas.items():
        stats = camera_stats[camera_id]
        
        # 按比例抽樣，確保每組都有代表
        day_items = stats['day']
        dusk_items = stats['dusk']
        night_items = stats['night']
        
        # 計算每組的配額（至少每組 2 張）
        day_quota = max(2, int(quota * len(day_items) / stats['total']))
        dusk_quota = max(2, int(quota * len(dusk_items) / stats['total']))
        night_quota = max(2, quota - day_quota - dusk_quota)
        
        # 如果 night 樣本少，多抽一些
        if len(night_items) < len(day_items) * 0.3:
            night_quota = min(len(night_items), int(quota * 0.4))
            day_quota = quota - dusk_quota - night_quota
        
        # 隨機抽樣
        random.shuffle(day_items)
        random.shuffle(dusk_items)
        random.shuffle(night_items)
        
        sampled_items.extend(day_items[:day_quota])
        sampled_items.extend(dusk_items[:dusk_quota])
        sampled_items.extend(night_items[:night_quota])
        
        print(f"    {camera_id}: day={day_quota}, dusk={dusk_quota}, night={night_quota}")
    
    print(f"\n  總抽樣數: {len(sampled_items)} 張")
    return sampled_items


def create_diagnostic_set(sampled_items, camera_data, output_dir='data/test_data'):
    """
    產生 Diagnostic Scenario Set CSV
    """
    print("\n" + "=" * 60)
    print("產生 Diagnostic Scenario Set CSV")
    print("=" * 60)
    
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'diagnostic_candidates.csv')
    
    # CSV 欄位
    fieldnames = [
        'camera_id', 'image_id', 'path', 'has_sky', 'sea_sky_confusable',
        'scene(sea|urban|forest|other)', 'occlusion(none|partial|heavy)',
        'weather(clear|cloudy|rain|fog|snow|unknown)', 'light (day|dusk|night)',
        'pred_sky_area_ratio', 'notes'
    ]
    
    stats = {'day': 0, 'dusk': 0, 'night': 0, 'weather': defaultdict(int)}
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for image_path, mask_path, image_id, luma in tqdm(sampled_items, desc='  處理'):
            # 從 image_path 提取 camera_id
            camera_id = None
            for cid, items in camera_data.items():
                if any(img_path == image_path for img_path, _, _ in items):
                    camera_id = cid
                    break
            
            if not camera_id:
                continue
            
            result = process_image(image_path, mask_path, camera_id, image_id)
            if result:
                row = {
                    'camera_id': result['camera_id'],
                    'image_id': result['image_id'],
                    'path': result['path'],
                    'has_sky': result['has_sky'],
                    'sea_sky_confusable': result['sea_sky_confusable'],
                    'scene(sea|urban|forest|other)': result['scene'],
                    'occlusion(none|partial|heavy)': result['occlusion'],
                    'weather(clear|cloudy|rain|fog|snow|unknown)': result['weather'],
                    'light (day|dusk|night)': result['light'],
                    'pred_sky_area_ratio': result['pred_sky_area_ratio'],
                    'notes': result['notes']
                }
                writer.writerow(row)
                stats[result['light']] += 1
                stats['weather'][result['weather']] += 1
    
    print(f"\n  已寫入: {output_file}")
    print(f"  Light 統計: day={stats['day']}, dusk={stats['dusk']}, night={stats['night']}")
    print(f"  Weather 統計: {dict(stats['weather'])}")


def generate_previews(csv_file, output_dir='data/test_data/previews', thumbnail_size=256):
    """
    為 Diagnostic Set 生成縮圖 preview
    """
    print("\n" + "=" * 60)
    print("生成縮圖 Preview")
    print("=" * 60)
    
    if not os.path.exists(csv_file):
        print(f"[錯誤] 找不到 CSV 檔案: {csv_file}")
        return
    
    # 讀取 CSV
    items = []
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            items.append(row)
    
    print(f"處理 {len(items)} 張圖片...")
    
    for item in tqdm(items, desc='  生成縮圖'):
        camera_id = item['camera_id']
        image_id = item['image_id']
        image_path = os.path.join('data', item['path'])
        
        if not os.path.exists(image_path):
            continue
        
        # 建立輸出資料夾
        camera_dir = os.path.join(output_dir, f'camera_{camera_id}')
        os.makedirs(camera_dir, exist_ok=True)
        
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
            output_path = os.path.join(camera_dir, f'{image_id}.jpg')
            thumb.save(output_path, 'JPEG', quality=85)
        except Exception as e:
            print(f"  [警告] 生成縮圖失敗 {image_path}: {e}")
    
    print(f"\n  縮圖已儲存至: {output_dir}")


def main():
    print("=" * 60)
    print("產生情境診斷資料集與標註")
    print("=" * 60)
    
    # 1. 掃描所有 camera
    camera_data = scan_all_cameras('data')
    
    if not camera_data:
        print("[錯誤] 找不到任何 camera 資料")
        return
    
    # 2. 產生 Hold-out Test
    create_holdout_test(camera_data)
    
    # 3. 產生 Diagnostic Scenario Set
    sampled_items = sample_diagnostic_set(camera_data, target_total=100)
    create_diagnostic_set(sampled_items, camera_data)
    
    # 4. 生成縮圖 preview（僅為 Diagnostic Set）
    diagnostic_csv = 'data/test_data/diagnostic_candidates.csv'
    generate_previews(diagnostic_csv)
    
    print("\n" + "=" * 60)
    print("完成！")
    print("=" * 60)
    print("\n產出檔案：")
    print("  - data/test_data/holdout_10870.csv")
    print("  - data/test_data/diagnostic_candidates.csv")
    print("  - data/test_data/previews/ (縮圖資料夾)")


if __name__ == '__main__':
    # 設定隨機種子
    random.seed(42)
    np.random.seed(42)
    
    main()
