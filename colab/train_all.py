"""
統一訓練入口腳本（Colab 專屬版本）

功能：
- 支援選擇性訓練特定模型（--models 參數）
- 所有路徑和參數都可通過 CLI 指定
- 自動偵測環境（本機/Colab）並設定預設 device
- 即使某個訓練失敗，其他訓練仍可繼續
- 記錄每個訓練的開始/結束時間和狀態
"""

import os
import sys
import subprocess
import datetime
import argparse
from pathlib import Path

# 確保輸出立即刷新
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# 添加專案根目錄到 Python 路徑
_script_dir = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_script_dir)
sys.path.insert(0, _root)
sys.path.insert(0, _script_dir)

from utils_colab import detect_environment, get_default_device, get_default_num_workers, resolve_device

# 模型映射
MODEL_MAP = {
    'dl': ('train_dl.py', 'DL baseline'),
    'dino-mlp': ('train_dino_mlp.py', 'DINO-MLP'),
    'dino-simplecnn': ('train_dino_simple_cnn.py', 'DINO-SimpleCNN'),
    'dino-fpn': ('train_dino_fpn.py', 'DINO-FPN'),
    'dino-hybrid': ('train_dino_hybrid.py', 'DINO-Hybrid'),
}


def log_message(message, log_file):
    """記錄訊息到檔案和終端"""
    print(message)
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(message + '\n')


def parse_models(models_str):
    """解析模型選擇字串"""
    if models_str.lower() == 'all':
        return list(MODEL_MAP.keys())
    
    models = [m.strip() for m in models_str.split(',')]
    valid_models = []
    for model in models:
        if model in MODEL_MAP:
            valid_models.append(model)
        else:
            print(f"[警告] 未知的模型：{model}，將跳過")
    
    return valid_models


def build_train_command(script_name, args):
    """構建訓練命令"""
    cmd = [sys.executable, os.path.join(_script_dir, script_name)]
    
    # 路徑參數
    if args.data_dir:
        cmd.extend(['--data_dir', args.data_dir])
    if args.outputs_dir:
        cmd.extend(['--outputs_dir', args.outputs_dir])
    if args.train_list:
        cmd.extend(['--train_list', args.train_list])
    if args.val_list:
        cmd.extend(['--val_list', args.val_list])
    
    # 訓練參數
    if args.batch_size:
        cmd.extend(['--batch_size', str(args.batch_size)])
    if args.epochs:
        cmd.extend(['--epochs', str(args.epochs)])
    if args.learning_rate:
        cmd.extend(['--learning_rate', str(args.learning_rate)])
    if args.image_size:
        cmd.extend(['--image_size'] + [str(s) for s in args.image_size])
    if args.device:
        cmd.extend(['--device', args.device])
    if args.num_workers is not None:
        cmd.extend(['--num_workers', str(args.num_workers)])
    
    return cmd


def main():
    parser = argparse.ArgumentParser(description='統一訓練入口（Colab 專屬版本）')
    
    # 模型選擇
    parser.add_argument('--models', type=str, default='all',
                        help='要訓練的模型（預設：all）。可選：all, dl, dino-mlp, dino-simplecnn, dino-fpn, dino-hybrid。可多選（逗號分隔）：dl,dino-mlp')
    
    # 路徑參數
    parser.add_argument('--data_dir', type=str, default='data',
                        help='數據根目錄（預設：data）')
    parser.add_argument('--outputs_dir', type=str, default='outputs',
                        help='輸出目錄（預設：outputs）')
    parser.add_argument('--train_list', type=str, default=None,
                        help='train_list.txt 路徑（預設：{outputs_dir}/train_list.txt）')
    parser.add_argument('--val_list', type=str, default=None,
                        help='val_list.txt 路徑（預設：{outputs_dir}/val_list.txt）')
    
    # 訓練參數
    parser.add_argument('--batch_size', type=int, default=2,
                        help='批次大小（預設：2，Colab 可用 8-16）')
    parser.add_argument('--epochs', type=int, default=10,
                        help='訓練輪數（預設：10）')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='學習率（預設：1e-4）')
    parser.add_argument('--image_size', type=int, nargs=2, default=[256, 256],
                        help='影像尺寸（預設：256 256，可指定如 512 512）')
    parser.add_argument('--device', type=str, default=None,
                        help='設備（cpu/cuda/auto，預設：自動偵測環境）')
    parser.add_argument('--num_workers', type=int, default=None,
                        help='數據載入進程數（預設：本機 0，Colab 2）')
    
    args = parser.parse_args()
    
    # 解析模型選擇
    models_to_train = parse_models(args.models)
    if not models_to_train:
        print("[錯誤] 沒有有效的模型可訓練")
        sys.exit(1)
    
    # 環境偵測
    env = detect_environment()
    
    # 解析 device（如果未指定，使用環境預設值）
    if args.device is None:
        args.device = get_default_device(env)
    
    # 解析 num_workers（如果未指定，使用環境預設值）
    if args.num_workers is None:
        args.num_workers = get_default_num_workers(env)
    
    # 設定預設路徑
    if args.train_list is None:
        args.train_list = os.path.join(args.outputs_dir, 'train_list.txt')
    if args.val_list is None:
        args.val_list = os.path.join(args.outputs_dir, 'val_list.txt')
    
    # 建立輸出目錄
    os.makedirs(args.outputs_dir, exist_ok=True)
    
    # 記錄檔案
    log_file = os.path.join(args.outputs_dir, 'training_batch_log.txt')
    
    start_time = datetime.datetime.now()
    
    # 清空或建立記錄檔案
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write('=' * 60 + '\n')
        f.write('  訓練批次執行記錄（Colab 專屬版本）\n')
        f.write('=' * 60 + '\n')
        f.write(f'開始時間: {start_time.strftime("%Y-%m-%d %H:%M:%S")}\n')
        f.write(f'環境: {env}\n')
        f.write(f'設備: {args.device}\n')
        f.write(f'模型: {", ".join(models_to_train)}\n')
        f.write('\n')
    
    log_message('=' * 60, log_file)
    log_message('  統一訓練入口（Colab 專屬版本）', log_file)
    log_message(f'  開始時間: {start_time.strftime("%Y-%m-%d %H:%M:%S")}', log_file)
    log_message(f'  環境: {env}', log_file)
    log_message(f'  設備: {args.device}', log_file)
    log_message(f'  批次大小: {args.batch_size}', log_file)
    log_message(f'  影像尺寸: {args.image_size}', log_file)
    log_message(f'  訓練模型: {", ".join(models_to_train)}', log_file)
    log_message('=' * 60, log_file)
    log_message('', log_file)
    
    # 依序執行每個訓練腳本
    for idx, model_key in enumerate(models_to_train, 1):
        script_name, model_display_name = MODEL_MAP[model_key]
        script_start = datetime.datetime.now()
        
        log_message('=' * 60, log_file)
        log_message(f'  [{idx}/{len(models_to_train)}] 執行: {model_display_name} ({script_name})', log_file)
        log_message(f'  開始時間: {script_start.strftime("%Y-%m-%d %H:%M:%S")}', log_file)
        log_message('=' * 60, log_file)
        log_message('', log_file)
        
        # 記錄到 log 檔案
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write('-' * 60 + '\n')
            f.write(f'模型: {model_display_name}\n')
            f.write(f'腳本: {script_name}\n')
            f.write(f'開始時間: {script_start.strftime("%Y-%m-%d %H:%M:%S")}\n')
            f.write('\n')
        
        # 構建命令
        cmd = build_train_command(script_name, args)
        
        # 確保環境變數設定
        env_vars = os.environ.copy()
        if args.device == 'cpu':
            env_vars['CUDA_VISIBLE_DEVICES'] = ''
        
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
                env=env_vars
            )
            
            # 即時輸出
            for line in process.stdout:
                print(line, end='')
                with open(log_file, 'a', encoding='utf-8') as f:
                    f.write(line)
            
            process.wait()
            
            script_end = datetime.datetime.now()
            duration = script_end - script_start
            
            if process.returncode == 0:
                log_message('', log_file)
                log_message(f'  [成功] {model_display_name} 執行成功', log_file)
                log_message(f'  結束時間: {script_end.strftime("%Y-%m-%d %H:%M:%S")}', log_file)
                log_message(f'  耗時: {duration}', log_file)
                
                with open(log_file, 'a', encoding='utf-8') as f:
                    f.write(f'\n結束時間: {script_end.strftime("%Y-%m-%d %H:%M:%S")}\n')
                    f.write(f'耗時: {duration}\n')
                    f.write('狀態: 成功\n')
            else:
                log_message('', log_file)
                log_message(f'  [失敗] {model_display_name} 執行失敗（繼續執行下一個）', log_file)
                log_message(f'  結束時間: {script_end.strftime("%Y-%m-%d %H:%M:%S")}', log_file)
                log_message(f'  耗時: {duration}', log_file)
                log_message(f'  錯誤碼: {process.returncode}', log_file)
                
                with open(log_file, 'a', encoding='utf-8') as f:
                    f.write(f'\n結束時間: {script_end.strftime("%Y-%m-%d %H:%M:%S")}\n')
                    f.write(f'耗時: {duration}\n')
                    f.write(f'錯誤碼: {process.returncode}\n')
                    f.write('狀態: 失敗\n')
        
        except Exception as e:
            script_end = datetime.datetime.now()
            duration = script_end - script_start
            
            log_message('', log_file)
            log_message(f'  [錯誤] {model_display_name} 執行時發生異常: {e}', log_file)
            log_message(f'  結束時間: {script_end.strftime("%Y-%m-%d %H:%M:%S")}', log_file)
            log_message(f'  耗時: {duration}', log_file)
            
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f'\n結束時間: {script_end.strftime("%Y-%m-%d %H:%M:%S")}\n')
                f.write(f'耗時: {duration}\n')
                f.write(f'異常: {str(e)}\n')
                f.write('狀態: 錯誤\n')
        
        log_message('', log_file)
        log_message('', log_file)
    
    # 記錄結束時間
    end_time = datetime.datetime.now()
    total_duration = end_time - start_time
    
    log_message('=' * 60, log_file)
    log_message('  所有訓練腳本執行完成', log_file)
    log_message(f'  結束時間: {end_time.strftime("%Y-%m-%d %H:%M:%S")}', log_file)
    log_message(f'  總耗時: {total_duration}', log_file)
    log_message('=' * 60, log_file)
    log_message('', log_file)
    
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write('\n' + '=' * 60 + '\n')
        f.write('  所有訓練腳本執行完成\n')
        f.write(f'結束時間: {end_time.strftime("%Y-%m-%d %H:%M:%S")}\n')
        f.write(f'總耗時: {total_duration}\n')
        f.write('=' * 60 + '\n')
    
    print(f"\n詳細記錄已儲存至: {log_file}")


if __name__ == "__main__":
    main()
