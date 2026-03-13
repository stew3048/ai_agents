import os
import sys
import argparse
from typing import List, Dict

import numpy as np
import torch

_script_dir = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_script_dir)
sys.path.insert(0, _root)
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

from scripts.eval_custom_data_three_methods import (  # type: ignore
    find_image_mask_pairs,
    evaluate_grounding_dino_sam,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="只在 data/image & data/mask 上跑 Grounding DINO + SAM（自動 overlay 目錄：grounding_dino_sam_overlays_auto）。"
    )
    parser.add_argument("--data_dir", type=str, default="data", help="資料根目錄（預設：data）")
    parser.add_argument("--output_dir", type=str, default="output", help="輸出根目錄（預設：output）")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="裝置（cpu 或 cuda；預設自動偵測）",
    )
    parser.add_argument(
        "--overlay_subdir",
        type=str,
        default="grounding_dino_sam_overlays_auto",
        help="overlay 子目錄名稱（預設：grounding_dino_sam_overlays_auto）",
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    output_dir = args.output_dir
    overlay_subdir = args.overlay_subdir

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("  自訂資料集：Grounding DINO + SAM 評估（單獨）")
    print("=" * 60)
    print(f"Data dir:    {data_dir}")
    print(f"Output dir:  {output_dir}")
    print(f"Device:      {device}")
    print(f"Overlay dir: {os.path.join(output_dir, overlay_subdir)}")
    print()

    pairs: List[Dict[str, str]] = find_image_mask_pairs(
        data_dir=data_dir,
        image_subdir="image",
        mask_subdir="mask",
    )
    print(f"共找到 {len(pairs)} 組 image-mask 配對。")
    for p in pairs:
        print(f"  - {p['image_name']}  (camera {p['camera_id']})")
    print()

    results = evaluate_grounding_dino_sam(
        pairs,
        device=device,
        output_dir=output_dir,
        overlay_subdir=overlay_subdir,
    )

    if not results:
        print("[錯誤] 沒有成功評估任何樣本。")
        return

    # 只統計有 GT 的樣本（iou / precision / recall 都非 None）
    valid = [
        r
        for r in results
        if r.get("iou") is not None
        and r.get("precision") is not None
        and r.get("recall") is not None
        and r.get("fp_rate") is not None
        and r.get("fn_rate") is not None
    ]
    if not valid:
        print("[警告] 沒有任何有 GT 的樣本可計算平均 IoU/Precision/Recall。")
        return

    ious = [float(r["iou"]) for r in valid]
    precisions = [float(r["precision"]) for r in valid]
    recalls = [float(r["recall"]) for r in valid]
    fp_rates = [float(r["fp_rate"]) for r in valid]
    fn_rates = [float(r["fn_rate"]) for r in valid]

    iou_mean = float(np.mean(ious))
    precision_mean = float(np.mean(precisions))
    recall_mean = float(np.mean(recalls))
    fp_rate_mean = float(np.mean(fp_rates))
    fn_rate_mean = float(np.mean(fn_rates))

    print()
    print("=" * 60)
    print("  Summary（Grounding DINO + SAM on custom data）")
    print("=" * 60)
    print(f"IoU_mean:      {iou_mean:.4f}")
    print(f"Precision_mean:{precision_mean:.4f}")
    print(f"Recall_mean:   {recall_mean:.4f}")
    print(f"FP_rate_mean:  {fp_rate_mean:.4f}")
    print(f"FN_rate_mean:  {fn_rate_mean:.4f}")
    print()
    print(f"Overlay 目錄: {os.path.join(output_dir, overlay_subdir)}")
    print()


if __name__ == "__main__":
    main()

