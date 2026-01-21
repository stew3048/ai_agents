"""
建立 SkyFinder camera 10066 的資料切分
使用時間順序切分避免相鄰影像洩漏

切分比例：
- train: 前 70%
- val: 中間 15%
- test: 後 15%

輸出：outputs/splits_10066.json
"""

import os
import json

# 設定
IMAGES_DIR = 'data/skyfinder_sample/images'
MASKS_DIR = 'data/skyfinder_sample/masks'
OUTPUT_FILE = 'outputs/splits_10066.json'

# 切分比例
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# 隨機種子（雖然時間切分用不太到，但保留以備不時之需）
SEED = 42


def main():
    print('=' * 60)
    print('  Creating Train/Val/Test Splits for SkyFinder 10066')
    print('=' * 60)
    print()
    
    # 確保輸出目錄存在
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    # 取得所有圖片檔案，按名稱排序（名稱是時間順序）
    image_files = sorted([f for f in os.listdir(IMAGES_DIR) if f.endswith(('.jpg', '.png'))])
    
    total_images = len(image_files)
    print(f'  Total images: {total_images}')
    print()
    
    # 計算切分點
    train_end = int(total_images * TRAIN_RATIO)
    val_end = train_end + int(total_images * VAL_RATIO)
    
    # 時間順序切分
    train_files = image_files[:train_end]
    val_files = image_files[train_end:val_end]
    test_files = image_files[val_end:]
    
    print(f'  Split ratios: {TRAIN_RATIO*100:.0f}% / {VAL_RATIO*100:.0f}% / {TEST_RATIO*100:.0f}%')
    print(f'  Train: {len(train_files)} images (indices 0-{train_end-1})')
    print(f'  Val:   {len(val_files)} images (indices {train_end}-{val_end-1})')
    print(f'  Test:  {len(test_files)} images (indices {val_end}-{total_images-1})')
    print()
    
    # 建立完整的資料結構
    def create_split_data(files):
        """為每個檔案建立 image 和 mask 的配對"""
        data = []
        for f in files:
            # 取得檔名（不含副檔名）
            base_name = os.path.splitext(f)[0]
            
            # mask 檔案名稱（假設是 .png）
            mask_file = f'{base_name}.png'
            
            # 確認 mask 存在
            mask_path = os.path.join(MASKS_DIR, mask_file)
            if os.path.exists(mask_path):
                data.append({
                    'image': f,
                    'mask': mask_file
                })
            else:
                print(f'  Warning: Mask not found for {f}')
        
        return data
    
    # 建立 splits 資料
    splits = {
        'metadata': {
            'camera_id': '10066',
            'total_images': total_images,
            'split_method': 'temporal_sequential',
            'ratios': {
                'train': TRAIN_RATIO,
                'val': VAL_RATIO,
                'test': TEST_RATIO
            },
            'seed': SEED,
            'images_dir': IMAGES_DIR,
            'masks_dir': MASKS_DIR,
            'note': 'SkyFinder uses ONE mask per camera (static scene). All images share the same mask.'
        },
        'train': create_split_data(train_files),
        'val': create_split_data(val_files),
        'test': create_split_data(test_files)
    }
    
    # 儲存 JSON
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(splits, f, indent=2, ensure_ascii=False)
    
    print(f'  Splits saved to: {OUTPUT_FILE}')
    print()
    
    # 顯示摘要
    print('=' * 60)
    print('  Summary')
    print('=' * 60)
    print()
    print(f'  Train: {len(splits["train"])} samples')
    print(f'  Val:   {len(splits["val"])} samples')
    print(f'  Test:  {len(splits["test"])} samples')
    print()
    print('  Split method: Temporal Sequential')
    print('    - Train uses EARLIEST images')
    print('    - Val uses MIDDLE images')
    print('    - Test uses LATEST images')
    print()
    print('  Why temporal split?')
    print('    - Adjacent images are very similar (same weather)')
    print('    - Random split would cause data leakage')
    print('    - Temporal split simulates real-world usage:')
    print('      train on past, predict future')
    print()
    print('=' * 60)


if __name__ == '__main__':
    main()
