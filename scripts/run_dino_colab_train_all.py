"""
依序執行 colab 四支 DINO 訓練：MLP → Simple CNN → FPN → Hybrid。
任一支崩潰時會把錯誤寫入 outputs/dino_train_runner_error.txt 並結束，
方便修正後重新執行本 script 繼續跑。
"""

import os
import sys
import subprocess
import traceback
from datetime import datetime

_script_dir = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_script_dir)
os.chdir(_root)
sys.path.insert(0, _root)

SCRIPTS = [
    ("train_dino_mlp", "colab/train_dino_mlp.py"),
    ("train_dino_simple_cnn", "colab/train_dino_simple_cnn.py"),
    ("train_dino_fpn", "colab/train_dino_fpn.py"),
    ("train_dino_hybrid", "colab/train_dino_hybrid.py"),
]
ERROR_LOG = os.path.join(_root, "outputs", "dino_train_runner_error.txt")


def main():
    os.makedirs(os.path.join(_root, "outputs"), exist_ok=True)
    print("=" * 60)
    print("  依序執行 DINO Colab 四支訓練")
    print("  MLP → Simple CNN → FPN → Hybrid")
    print("=" * 60)
    for name, rel_path in SCRIPTS:
        script_path = os.path.join(_root, rel_path)
        if not os.path.isfile(script_path):
            msg = f"找不到腳本: {script_path}"
            _write_error(name, msg)
            print(msg)
            return 1
        print()
        print(f">>> 執行 {name}: {rel_path}")
        try:
            r = subprocess.run(
                [sys.executable, rel_path],
                cwd=_root,
                capture_output=False,
            )
            if r.returncode != 0:
                msg = f"{name} 結束時 exit code = {r.returncode}"
                _write_error(name, msg)
                print(msg)
                return 1
        except Exception as e:
            tb = traceback.format_exc()
            _write_error(name, f"{e}\n\n{tb}")
            print(tb)
            return 1
    print()
    print("=" * 60)
    print("  四支訓練皆已完成")
    print("=" * 60)
    if os.path.isfile(ERROR_LOG):
        try:
            os.remove(ERROR_LOG)
        except Exception:
            pass
    return 0


def _write_error(script_name, content):
    with open(ERROR_LOG, "w", encoding="utf-8") as f:
        f.write(f"[{datetime.now().isoformat()}] 失敗腳本: {script_name}\n\n")
        f.write(content)
    print(f"錯誤已寫入: {ERROR_LOG}")


if __name__ == "__main__":
    sys.exit(main())
