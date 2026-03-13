"""
印出 train/val/test 切分統計資訊
用於確認切分是否正確（按時間順序）
"""

import os
import json

SPLITS_FILE = 'outputs/splits_10066.json'


def main():
    print('=' * 60)
    print('  Train/Val/Test Split Statistics')
    print('=' * 60)
    print()
    
    # 載入 splits
    if not os.path.exists(SPLITS_FILE):
        print(f'  Error: {SPLITS_FILE} not found!')
        print('  Please run create_splits.py first.')
        return
    
    with open(SPLITS_FILE, 'r', encoding='utf-8') as f:
        splits = json.load(f)
    
    # 印出 metadata
    meta = splits['metadata']
    print('  Metadata:')
    print(f'    Camera ID: {meta["camera_id"]}')
    print(f'    Total images: {meta["total_images"]}')
    print(f'    Split method: {meta["split_method"]}')
    print(f'    Ratios: train={meta["ratios"]["train"]*100:.0f}%, val={meta["ratios"]["val"]*100:.0f}%, test={meta["ratios"]["test"]*100:.0f}%')
    print()
    
    # 印出各 split 的統計
    print('-' * 60)
    print('  Split Statistics:')
    print('-' * 60)
    print()
    
    for split_name in ['train', 'val', 'test']:
        split_data = splits[split_name]
        count = len(split_data)
        
        print(f'  [{split_name.upper()}] {count} samples')
        print()
        
        if count > 0:
            # 取得檔名列表
            files = [item['image'] for item in split_data]
            
            # 前 3 個
            print('    First 3:')
            for i, f in enumerate(files[:3]):
                print(f'      {i+1}. {f}')
            
            # 後 3 個
            print('    Last 3:')
            start_idx = max(0, count - 3)
            for i, f in enumerate(files[-3:]):
                print(f'      {start_idx + i + 1}. {f}')
            
            print()
    
    # 驗證時間順序
    print('-' * 60)
    print('  Temporal Order Verification:')
    print('-' * 60)
    print()
    
    train_files = [item['image'] for item in splits['train']]
    val_files = [item['image'] for item in splits['val']]
    test_files = [item['image'] for item in splits['test']]
    
    # 檢查是否有重疊
    train_set = set(train_files)
    val_set = set(val_files)
    test_set = set(test_files)
    
    overlap_train_val = train_set & val_set
    overlap_val_test = val_set & test_set
    overlap_train_test = train_set & test_set
    
    print(f'    Train-Val overlap: {len(overlap_train_val)} files')
    print(f'    Val-Test overlap: {len(overlap_val_test)} files')
    print(f'    Train-Test overlap: {len(overlap_train_test)} files')
    print()
    
    if len(overlap_train_val) == 0 and len(overlap_val_test) == 0 and len(overlap_train_test) == 0:
        print('    [OK] No overlap between splits!')
    else:
        print('    [WARNING] Overlap detected!')
    
    # 檢查時間順序
    if train_files and val_files:
        train_last = train_files[-1]
        val_first = val_files[0]
        print()
        print(f'    Train ends with: {train_last}')
        print(f'    Val starts with: {val_first}')
        
        if train_last < val_first:
            print('    [OK] Train < Val (temporal order correct)')
        else:
            print('    [WARNING] Train >= Val (order issue!)')
    
    if val_files and test_files:
        val_last = val_files[-1]
        test_first = test_files[0]
        print()
        print(f'    Val ends with: {val_last}')
        print(f'    Test starts with: {test_first}')
        
        if val_last < test_first:
            print('    [OK] Val < Test (temporal order correct)')
        else:
            print('    [WARNING] Val >= Test (order issue!)')
    
    print()
    print('=' * 60)


if __name__ == '__main__':
    main()
