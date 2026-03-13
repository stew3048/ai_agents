"""
確保訓練持續運行的腳本
如果訓練停止，會自動重新啟動
"""
import os
import time
import subprocess
import sys
from pathlib import Path
from datetime import datetime

def check_training_status():
    """檢查訓練狀態"""
    outputs_dir = Path('outputs')
    train_dirs = list(outputs_dir.glob('train_multi_camera_*'))
    if not train_dirs:
        return None, 0, None
    
    latest_dir = max(train_dirs, key=os.path.getmtime)
    log_file = latest_dir / 'training_log.csv'
    
    if not log_file.exists():
        return latest_dir, 0, None
    
    # 讀取已完成的 epochs
    with open(log_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        completed_epochs = len(lines) - 1  # 減去 header
    
    # 檢查最後修改時間
    mtime = os.path.getmtime(log_file)
    age_seconds = time.time() - mtime
    
    return latest_dir, completed_epochs, age_seconds

def is_training_running():
    """檢查是否有訓練進程在運行"""
    try:
        result = subprocess.run(
            ['tasklist', '/FI', 'IMAGENAME eq python.exe', '/FO', 'CSV'],
            capture_output=True,
            text=True,
            timeout=5
        )
        # 簡單檢查：如果有 python 進程，假設可能是訓練
        # 更精確的方法需要檢查命令行參數，但這在 Windows 上比較複雜
        return 'python.exe' in result.stdout
    except:
        return False

def main():
    print("=" * 60)
    print("訓練狀態檢查")
    print("=" * 60)
    print()
    
    latest_dir, completed_epochs, age_seconds = check_training_status()
    
    if latest_dir is None:
        print("[錯誤] 找不到訓練目錄，啟動新訓練...")
        subprocess.Popen([sys.executable, 'train_multi_camera.py'], 
                        cwd=os.getcwd(),
                        creationflags=subprocess.CREATE_NO_WINDOW)
        print("[完成] 已啟動新訓練")
        return
    
    print(f"訓練目錄: {latest_dir.name}")
    print(f"已完成 Epochs: {completed_epochs}/10")
    
    if age_seconds is not None:
        age_minutes = age_seconds / 60
        last_update = datetime.fromtimestamp(time.time() - age_seconds).strftime("%H:%M:%S")
        print(f"最後更新: {last_update} (約 {age_minutes:.1f} 分鐘前)")
    
    # 判斷訓練狀態
    if completed_epochs >= 10:
        print()
        print("[完成] 訓練已完成！")
        overlay_dir = latest_dir / 'val_overlays'
        if overlay_dir.exists() and list(overlay_dir.glob('*.png')):
            print("[完成] Val Overlay 已生成")
        else:
            print("[等待] 等待生成 Val Overlay...")
        return
    
    # 如果超過 10 分鐘沒有更新，可能已停止
    if age_seconds and age_seconds > 600:
        print()
        print("[警告] 訓練可能已停止（超過 10 分鐘無更新）")
        print("[重新啟動] 重新啟動訓練...")
        subprocess.Popen([sys.executable, 'train_multi_camera.py'], 
                        cwd=os.getcwd(),
                        creationflags=subprocess.CREATE_NO_WINDOW)
        print("[完成] 已重新啟動訓練")
    elif age_seconds and age_seconds < 300:
        print()
        print("[進行中] 訓練正在進行中...")
        print(f"   預計還需要約 {int((10 - completed_epochs) * 3)} 分鐘完成")
    else:
        print()
        print("[等待] 訓練可能正在進行中，請稍候...")

if __name__ == '__main__':
    main()
