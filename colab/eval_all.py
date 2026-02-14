"""
統一評估入口腳本（Colab 專屬版本）

功能：
- 支援選擇性評估特定模型（--models 參數）
- 所有路徑和參數都可通過 CLI 指定
- 支援 overlay 輸出（可指定輸出路徑）
- 自動偵測環境（本機/Colab）並設定預設 device
"""

import os
import sys
import subprocess
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

from utils_colab import detect_environment, get_default_device

# 模型映射
MODEL_MAP = {
    'dl': ('eval_dl.py', 'DL baseline'),
    'sam': ('eval_sam.py', 'SAM'),
    'clipseg': ('eval_clipseg.py', 'CLIPSeg'),
    'dino-dl': ('eval_dino_dl.py', 'DINO-DL'),
    'grounding-dino-sam': ('eval_grounding_dino_sam.py', 'GroundingDINO-SAM'),
}


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


def build_eval_command(script_name, args):
    """構建評估命令"""
    cmd = [sys.executable, os.path.join(_script_dir, script_name)]
    
    # 路徑參數
    if args.data_dir:
        cmd.extend(['--data_dir', args.data_dir])
    if args.outputs_dir:
        cmd.extend(['--outputs_dir', args.outputs_dir])
    if args.test_list:
        cmd.extend(['--test_list', args.test_list])
    if args.checkpoint_dir:
        cmd.extend(['--checkpoint_dir', args.checkpoint_dir])
    if args.output_csv:
        cmd.extend(['--output_csv', args.output_csv])
    if args.overlay_dir:
        cmd.extend(['--overlay_dir', args.overlay_dir])
    
    # 評估參數
    if args.image_size:
        cmd.extend(['--image_size'] + [str(s) for s in args.image_size])
    if args.device:
        cmd.extend(['--device', args.device])
    if args.batch_size:
        cmd.extend(['--batch_size', str(args.batch_size)])
    
    return cmd


def main():
    parser = argparse.ArgumentParser(description='統一評估入口（Colab 專屬版本）')
    
    # 模型選擇
    parser.add_argument('--models', type=str, default='all',
                        help='要評估的模型（預設：all）。可選：all, dl, sam, clipseg, dino-dl。可多選（逗號分隔）：dl,sam')
    
    # 路徑參數
    parser.add_argument('--data_dir', type=str, default='data',
                        help='數據根目錄（預設：data）')
    parser.add_argument('--outputs_dir', type=str, default='outputs',
                        help='輸出目錄（預設：outputs）')
    parser.add_argument('--test_list', type=str, default=None,
                        help='test_list.txt 路徑（預設：{outputs_dir}/test_list.txt）')
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                        help='checkpoint 目錄（預設：{outputs_dir}，自動尋找）')
    parser.add_argument('--output_csv', type=str, default=None,
                        help='輸出 CSV 路徑（預設：{outputs_dir}/metrics_summary.csv）')
    parser.add_argument('--overlay_dir', type=str, default=None,
                        help='overlay 輸出目錄（預設：不輸出 overlay；指定則輸出到該目錄）')
    
    # 評估參數
    parser.add_argument('--image_size', type=int, nargs=2, default=[256, 256],
                        help='與訓練時一致的 image_size（預設：256 256）')
    parser.add_argument('--device', type=str, default=None,
                        help='設備（cpu/cuda/auto，預設：自動偵測環境）')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='批次大小（預設：4）')
    
    args = parser.parse_args()
    
    # 解析模型選擇
    models_to_eval = parse_models(args.models)
    if not models_to_eval:
        print("[錯誤] 沒有有效的模型可評估")
        sys.exit(1)
    
    # 環境偵測
    env = detect_environment()
    
    # 解析 device（如果未指定，使用環境預設值）
    if args.device is None:
        args.device = get_default_device(env)
    
    # 設定預設路徑
    if args.test_list is None:
        args.test_list = os.path.join(args.outputs_dir, 'test_list.txt')
    if args.output_csv is None:
        args.output_csv = os.path.join(args.outputs_dir, 'metrics_summary.csv')
    if args.checkpoint_dir is None:
        args.checkpoint_dir = args.outputs_dir
    
    # 建立輸出目錄
    os.makedirs(args.outputs_dir, exist_ok=True)
    if args.overlay_dir:
        os.makedirs(args.overlay_dir, exist_ok=True)
    
    print("=" * 60)
    print("  統一評估入口（Colab 專屬版本）")
    print("=" * 60)
    print()
    print(f"環境: {env}")
    print(f"設備: {args.device}")
    print(f"影像尺寸: {args.image_size}")
    print(f"評估模型: {', '.join(models_to_eval)}")
    if args.overlay_dir:
        print(f"Overlay 輸出目錄: {args.overlay_dir}")
    print()
    
    # 依序執行每個評估腳本
    for idx, model_key in enumerate(models_to_eval, 1):
        script_name, model_display_name = MODEL_MAP[model_key]
        script_path = os.path.join(_script_dir, script_name)
        
        if not os.path.exists(script_path):
            print(f"[警告] 找不到評估腳本：{script_path}，跳過 {model_display_name}")
            continue
        
        print("=" * 60)
        print(f"  [{idx}/{len(models_to_eval)}] 評估: {model_display_name} ({script_name})")
        print("=" * 60)
        print()
        
        # 構建命令
        cmd = build_eval_command(script_name, args)
        
        # 確保環境變數設定
        env_vars = os.environ.copy()
        if args.device == 'cpu':
            env_vars['CUDA_VISIBLE_DEVICES'] = ''
        
        try:
            # 執行腳本
            result = subprocess.run(
                cmd,
                env=env_vars,
                cwd=_root
            )
            
            if result.returncode == 0:
                print()
                print(f"  [成功] {model_display_name} 評估完成")
            else:
                print()
                print(f"  [失敗] {model_display_name} 評估失敗（錯誤碼：{result.returncode}）")
        
        except Exception as e:
            print()
            print(f"  [錯誤] {model_display_name} 評估時發生異常: {e}")
        
        print()
        print()
    
    print("=" * 60)
    print("  所有評估腳本執行完成")
    print("=" * 60)
    print()
    print(f"結果 CSV: {args.output_csv}")
    if args.overlay_dir:
        print(f"Overlay 目錄: {args.overlay_dir}")
    print()


if __name__ == "__main__":
    main()
