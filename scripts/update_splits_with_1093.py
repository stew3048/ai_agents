"""
更新 splits，將 1093 加入 train，維持 val=9483, test=10870
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.create_multi_camera_splits import create_splits

if __name__ == '__main__':
    # 手動指定：1093 加入 train，維持 val=9483, test=10870
    result = create_splits(
        train_cameras=['10066', '9291', '9112', '1093'],
        val_camera='9483',
        test_camera='10870',
        train_ratio=0.9,
        seed=42,
        output_file='outputs/multi_camera_splits.json'
    )
    
    print("\n" + "=" * 60)
    print("  Splits 更新完成！")
    print("=" * 60)
    print(f"\n  Train cameras: ['10066', '9291', '9112', '1093']")
    print(f"  Val camera: 9483")
    print(f"  Test camera: 10870")
    print(f"\n  Train: {len(result['train'])} 張")
    print(f"  Sanity: {len(result['sanity'])} 張")
    print(f"  Val: {len(result['val'])} 張")
    print(f"  Test: {len(result['test'])} 張")
