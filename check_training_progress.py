"""
檢查訓練進度的簡單腳本
"""
import os
import time
import csv
from pathlib import Path

# 找到最新的訓練目錄
outputs_dir = Path('outputs')
train_dirs = list(outputs_dir.glob('train_multi_camera_*'))
if not train_dirs:
    print("找不到訓練目錄")
    exit(1)

latest_dir = max(train_dirs, key=os.path.getmtime)
log_file = latest_dir / 'training_log.csv'

print(f"監控訓練目錄: {latest_dir}")
print(f"日誌文件: {log_file}")
print("=" * 60)

if not log_file.exists():
    print("訓練日誌文件尚未生成，等待中...")
    while not log_file.exists():
        time.sleep(5)
    print("訓練日誌文件已生成！")

last_epoch = 0
while True:
    if log_file.exists():
        with open(log_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            current_epoch = len(rows)
            
            if current_epoch > last_epoch:
                print(f"\n進度更新: Epoch {current_epoch}/10")
                if rows:
                    latest = rows[-1]
                    print(f"  Train Loss:  {latest['train_loss']}")
                    print(f"  Train IoU:   {latest['train_iou']}")
                    print(f"  Sanity IoU:  {latest['sanity_iou']}")
                    print(f"  Val IoU:     {latest['val_iou']}")
                    print(f"  Val Dice:    {latest['val_dice']}")
                
                last_epoch = current_epoch
                
                if current_epoch >= 10:
                    print("\n" + "=" * 60)
                    print("訓練完成！")
                    print("=" * 60)
                    
                    # 檢查是否有 overlay 目錄
                    overlay_dir = latest_dir / 'val_overlays'
                    if overlay_dir.exists():
                        overlays = list(overlay_dir.glob('*.png'))
                        print(f"\nVal Overlay 已生成: {len(overlays)} 張")
                        for overlay in sorted(overlays):
                            print(f"  - {overlay.name}")
                    else:
                        print("\n等待生成 Val Overlay...")
                        time.sleep(5)
                        if overlay_dir.exists():
                            overlays = list(overlay_dir.glob('*.png'))
                            print(f"Val Overlay 已生成: {len(overlays)} 張")
                    
                    print(f"\n所有結果在: {latest_dir}")
                    break
    
    time.sleep(10)
