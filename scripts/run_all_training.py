"""
批次執行所有訓練腳本（依序執行，使用 CPU）

功能：
- 依序執行 5 個訓練腳本
- 強制使用 CPU（避免 GPU 問題）
- 即使某個訓練失敗，其他訓練仍可繼續
- 記錄每個訓練的開始/結束時間和狀態
"""

import os
import sys
import subprocess
import datetime
from pathlib import Path

# 確保輸出立即刷新
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# 強制使用 CPU
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# 訓練腳本列表
TRAIN_SCRIPTS = [
    'train_in_domain.py',
    'train_dino_mlp.py',
    'train_dino_simple_cnn.py',
    'train_dino_fpn.py',
    'train_dino_hybrid.py'
]

# 記錄檔案
LOG_FILE = 'outputs/training_batch_log.txt'
os.makedirs('outputs', exist_ok=True)


def log_message(message, log_file=LOG_FILE):
    """記錄訊息到檔案和終端"""
    print(message)
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(message + '\n')


def main():
    start_time = datetime.datetime.now()
    
    # 清空或建立記錄檔案
    with open(LOG_FILE, 'w', encoding='utf-8') as f:
        f.write('=' * 60 + '\n')
        f.write('  訓練批次執行記錄\n')
        f.write('=' * 60 + '\n')
        f.write(f'開始時間: {start_time.strftime("%Y-%m-%d %H:%M:%S")}\n')
        f.write('使用設備: CPU（強制）\n')
        f.write('\n')
    
    log_message('=' * 60)
    log_message('  開始執行所有訓練腳本')
    log_message(f'  開始時間: {start_time.strftime("%Y-%m-%d %H:%M:%S")}')
    log_message('  使用設備: CPU（強制）')
    log_message('=' * 60)
    log_message('')
    
    # 依序執行每個訓練腳本
    for idx, script in enumerate(TRAIN_SCRIPTS, 1):
        script_start = datetime.datetime.now()
        
        log_message('=' * 60)
        log_message(f'  [{idx}/{len(TRAIN_SCRIPTS)}] 執行: {script}')
        log_message(f'  開始時間: {script_start.strftime("%Y-%m-%d %H:%M:%S")}')
        log_message('=' * 60)
        log_message('')
        
        # 記錄到 log 檔案
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            f.write('-' * 60 + '\n')
            f.write(f'腳本: {script}\n')
            f.write(f'開始時間: {script_start.strftime("%Y-%m-%d %H:%M:%S")}\n')
            f.write('\n')
        
        # 執行訓練腳本
        script_path = os.path.join('scripts', script)
        cmd = [sys.executable, script_path, '--device', 'cpu']  # 明確指定使用 CPU
        
        # 確保環境變數設定（雖然已經在腳本開頭設定，但子進程需要繼承）
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = ''
        
        try:
            # 執行腳本，將輸出同時顯示在終端和記錄檔案
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                bufsize=1,
                universal_newlines=True,
                env=env  # 傳遞環境變數
            )
            
            # 即時輸出
            for line in process.stdout:
                print(line, end='')
                with open(LOG_FILE, 'a', encoding='utf-8') as f:
                    f.write(line)
            
            process.wait()
            
            script_end = datetime.datetime.now()
            duration = script_end - script_start
            
            if process.returncode == 0:
                log_message('')
                log_message(f'  [成功] {script} 執行成功')
                log_message(f'  結束時間: {script_end.strftime("%Y-%m-%d %H:%M:%S")}')
                log_message(f'  耗時: {duration}')
                
                with open(LOG_FILE, 'a', encoding='utf-8') as f:
                    f.write(f'\n結束時間: {script_end.strftime("%Y-%m-%d %H:%M:%S")}\n')
                    f.write(f'耗時: {duration}\n')
                    f.write('狀態: 成功\n')
            else:
                log_message('')
                log_message(f'  [失敗] {script} 執行失敗（繼續執行下一個）')
                log_message(f'  結束時間: {script_end.strftime("%Y-%m-%d %H:%M:%S")}')
                log_message(f'  耗時: {duration}')
                log_message(f'  錯誤碼: {process.returncode}')
                
                with open(LOG_FILE, 'a', encoding='utf-8') as f:
                    f.write(f'\n結束時間: {script_end.strftime("%Y-%m-%d %H:%M:%S")}\n')
                    f.write(f'耗時: {duration}\n')
                    f.write(f'錯誤碼: {process.returncode}\n')
                    f.write('狀態: 失敗\n')
        
        except Exception as e:
            script_end = datetime.datetime.now()
            duration = script_end - script_start
            
            log_message('')
            log_message(f'  [錯誤] {script} 執行時發生異常: {e}')
            log_message(f'  結束時間: {script_end.strftime("%Y-%m-%d %H:%M:%S")}')
            log_message(f'  耗時: {duration}')
            
            with open(LOG_FILE, 'a', encoding='utf-8') as f:
                f.write(f'\n結束時間: {script_end.strftime("%Y-%m-%d %H:%M:%S")}\n')
                f.write(f'耗時: {duration}\n')
                f.write(f'異常: {str(e)}\n')
                f.write('狀態: 錯誤\n')
        
        log_message('')
        log_message('')
    
    # 記錄結束時間
    end_time = datetime.datetime.now()
    total_duration = end_time - start_time
    
    log_message('=' * 60)
    log_message('  所有訓練腳本執行完成')
    log_message(f'  結束時間: {end_time.strftime("%Y-%m-%d %H:%M:%S")}')
    log_message(f'  總耗時: {total_duration}')
    log_message('=' * 60)
    log_message('')
    
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write('\n' + '=' * 60 + '\n')
        f.write('  所有訓練腳本執行完成\n')
        f.write(f'結束時間: {end_time.strftime("%Y-%m-%d %H:%M:%S")}\n')
        f.write(f'總耗時: {total_duration}\n')
        f.write('=' * 60 + '\n')
    
    print(f"\n詳細記錄已儲存至: {LOG_FILE}")


if __name__ == "__main__":
    main()
