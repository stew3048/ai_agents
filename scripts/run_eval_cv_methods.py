"""
本機 / Colab 共用：CV 方法評估入口（CLIPSeg、Grounding DINO+SAM2）

從專案根目錄執行即可，會自動選擇 scripts（本機）或 colab（Colab）腳本。
  - 本機：python scripts/run_eval_cv_methods.py --models clipseg,grounding-dino-sam
  - Colab：同上，或在 Colab 環境下執行，會改用 colab/ 下腳本

用法：
  python scripts/run_eval_cv_methods.py --models clipseg
  python scripts/run_eval_cv_methods.py --models grounding-dino-sam
  python scripts/run_eval_cv_methods.py --models clipseg,grounding-dino-sam --test_list outputs/test_list.txt
"""
import os
import sys
import subprocess
import argparse

_script_dir = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_script_dir)
sys.path.insert(0, _root)

# 偵測是否在 Colab
def _is_colab():
    try:
        import google.colab
        return True
    except ImportError:
        return False

# 本機 final 腳本 vs Colab 腳本對應
MODEL_SCRIPTS = {
    'clipseg': {
        'local': 'scripts/eval_clipseg_final.py',
        'colab': 'colab/eval_clipseg.py',
    },
    'grounding-dino-sam': {
        'local': 'scripts/eval_grounding_dino_sam_final.py',
        'colab': 'colab/eval_grounding_dino_sam.py',
    },
}

def main():
    parser = argparse.ArgumentParser(description='本機/Colab 共用 CV 評估入口')
    parser.add_argument('--models', type=str, default='clipseg,grounding-dino-sam',
                        help='逗號分隔：clipseg, grounding-dino-sam（預設兩者都跑）')
    parser.add_argument('--env', type=str, default='auto', choices=['auto', 'local', 'colab'],
                        help='環境：auto=自動偵測, local=本機 scripts, colab=colab 腳本')
    parser.add_argument('--test_list', type=str, default=None,
                        help='test 清單路徑（預設：outputs/test_list.txt）')
    parser.add_argument('--output_csv', type=str, default=None,
                        help='輸出 CSV（預設：outputs/metrics_summary_test.csv）')
    parser.add_argument('--data_dir', type=str, default='data', help='資料根目錄')
    parser.add_argument('--overlay_dir', type=str, default=None, help='overlay 輸出目錄')
    parser.add_argument('--device', type=str, default=None, help='cuda / cpu')
    # CLIPSeg 改善項開關（僅本機 final 支援；Colab 版若支援可再傳）
    parser.add_argument('--no_dynamic_threshold', action='store_true', help='CLIPSeg: 關閉動態閾值')
    parser.add_argument('--no_spatial_prior', action='store_true', help='CLIPSeg: 關閉空間先驗')
    parser.add_argument('--no_morphology_open', action='store_true', help='CLIPSeg: 關閉形態學 OPEN')
    parser.add_argument('--no_multi_prompt_contrast', action='store_true', help='CLIPSeg: 關閉多 prompt 對比')
    args = parser.parse_args()

    if args.test_list is None:
        args.test_list = os.path.join(_root, 'outputs', 'test_list.txt')
    if args.output_csv is None:
        args.output_csv = os.path.join(_root, 'outputs', 'metrics_summary_test.csv')

    env = args.env
    if env == 'auto':
        env = 'colab' if _is_colab() else 'local'

    models = [m.strip() for m in args.models.split(',') if m.strip()]
    valid = [m for m in models if m in MODEL_SCRIPTS]
    if not valid:
        print('可選 --models: clipseg, grounding-dino-sam')
        sys.exit(1)

    print('=' * 60)
    print('  CV 方法評估（本機/Colab 共用入口）')
    print('=' * 60)
    print(f'  環境: {env}')
    print(f'  模型: {", ".join(valid)}')
    print(f'  test_list: {args.test_list}')
    print(f'  output_csv: {args.output_csv}')
    print()

    for model in valid:
        script_rel = MODEL_SCRIPTS[model][env]
        script_path = os.path.join(_root, script_rel)
        if not os.path.exists(script_path):
            print(f'[跳過] 找不到 {script_path}')
            continue
        cmd = [sys.executable, script_path]
        cmd += ['--test_list', args.test_list, '--output_csv', args.output_csv]
        if args.data_dir and 'grounding' in script_path:
            cmd += ['--data_dir', args.data_dir]
        if args.overlay_dir:
            cmd += ['--overlay_dir', args.overlay_dir]
        if args.device:
            cmd += ['--device', args.device]
        if model == 'clipseg':
            if args.no_dynamic_threshold:
                cmd.append('--no_dynamic_threshold')
            if args.no_spatial_prior:
                cmd.append('--no_spatial_prior')
            if args.no_morphology_open:
                cmd.append('--no_morphology_open')
            if args.no_multi_prompt_contrast:
                cmd.append('--no_multi_prompt_contrast')

        print(f'執行: {" ".join(cmd)}')
        r = subprocess.run(cmd, cwd=_root)
        if r.returncode != 0:
            print(f'[警告] {script_rel} 結束碼 {r.returncode}')
        print()

    print('完成。結果 CSV:', args.output_csv)

if __name__ == '__main__':
    main()
