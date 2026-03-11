import os
import sys
import argparse
from typing import List, Dict, Tuple, Optional

import numpy as np
from PIL import Image

import torch

_script_dir = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_script_dir)
sys.path.insert(0, _root)
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

from models import create_dinov2_segmentation_model  # type: ignore
from utils.metrics import calculate_metrics  # type: ignore
from scripts.eval_dino_segmentation_small import (  # type: ignore
    load_gt_mask as load_gt_resize,
    create_overlay,
)


def find_image_mask_pairs(
    data_dir: str,
    image_subdir: str = "image",
    mask_subdir: str = "mask",
) -> List[Dict[str, str]]:
    """
    掃描 data/image 與 data/mask，依照 camera id 自行配對。
    規則：
      - image 檔名: <camera>_<frame>.jpg，例如 10066_046.jpg
      - mask 檔名:  <camera>_mask.png，例如 10066_mask.png
    """
    image_dir = os.path.join(data_dir, image_subdir)
    mask_dir = os.path.join(data_dir, mask_subdir)

    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"找不到影像資料夾: {image_dir}")
    if not os.path.isdir(mask_dir):
        raise FileNotFoundError(f"找不到 mask 資料夾: {mask_dir}")

    mask_map: Dict[str, str] = {}
    for fname in os.listdir(mask_dir):
        name, ext = os.path.splitext(fname)
        if ext.lower() not in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
            continue
        if name.endswith("_mask"):
            cam_id = name[:-5]
        else:
            cam_id = name
        mask_map[cam_id] = os.path.join(mask_dir, fname)

    pairs: List[Dict[str, str]] = []
    for fname in os.listdir(image_dir):
        name, ext = os.path.splitext(fname)
        if ext.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
            continue
        cam_id = name.split("_")[0] if "_" in name else name
        mask_path = mask_map.get(cam_id)
        if not mask_path:
            print(f"  [警告] 找不到 camera {cam_id} 的 mask，略過 {fname}")
            continue
        image_path = os.path.join(image_dir, fname)
        pairs.append(
            {
                "camera_id": cam_id,
                "image_path": image_path,
                "mask_path": mask_path,
                "image_name": fname,
            }
        )

    if not pairs:
        raise RuntimeError("在 data/image 與 data/mask 中找不到任何可配對的樣本。")

    return sorted(pairs, key=lambda x: x["image_name"])


def compute_precision_recall_from_logits(
    logits: torch.Tensor,
    gt: torch.Tensor,
) -> Tuple[float, float]:
    """
    logits, gt: shape (1,1,H,W), gt 為 0/1。
    回傳 (precision, recall)。
    """
    smooth = 1e-6
    pred_b = (torch.sigmoid(logits) > 0.5).float()
    gt_b = (gt > 0.5).float()

    tp = ((pred_b == 1) & (gt_b == 1)).sum().item()
    fp = ((pred_b == 1) & (gt_b == 0)).sum().item()
    fn = ((pred_b == 0) & (gt_b == 1)).sum().item()

    precision = tp / (tp + fp + smooth) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn + smooth) if tp + fn > 0 else 0.0
    return float(precision), float(recall)


def evaluate_custom_dino_segmentation(
    pairs: List[Dict[str, str]],
    checkpoint_path: str,
    device: torch.device,
    image_size: Tuple[int, int] = (160, 160),
    overlay_dir: Optional[str] = None,
) -> List[Dict[str, object]]:
    """
    使用 DINOv2SegmentationModel（最後四層 concat + conv decoder）
    在 data/image & data/mask 上評估。
    """
    print("=== 評估 DINO + dino_segmentation（custom data）===")
    print(f"  使用 checkpoint: {checkpoint_path}")
    if overlay_dir:
        os.makedirs(overlay_dir, exist_ok=True)
        print(f"  Overlay 輸出目錄: {overlay_dir}")

    model = create_dinov2_segmentation_model().to(device)
    ck = torch.load(checkpoint_path, map_location=device)
    # 與 run_dino_segmentation 一致：ck 可能直接是 state_dict 或含 model_state_dict
    model.load_state_dict(ck.get("model_state_dict", ck), strict=False)
    model.eval()

    results: List[Dict[str, object]] = []

    for item in pairs:
        camera_id = item["camera_id"]
        image_path = item["image_path"]
        mask_path = item["mask_path"]
        image_name = item["image_name"]

        if not os.path.exists(image_path) or not os.path.exists(mask_path):
            print(f"  [警告] 缺檔，略過: {image_path}")
            continue

        img_pil = Image.open(image_path).convert("RGB")
        img_np = np.array(img_pil).astype(np.float32) / 255.0
        img_np = img_np.transpose(2, 0, 1)  # C,H,W
        img_t = torch.from_numpy(img_np).unsqueeze(0).to(device)
        if img_t.shape[2:] != image_size:
            img_t = torch.nn.functional.interpolate(
                img_t, size=image_size, mode="bilinear", align_corners=False
            )

        with torch.no_grad():
            logits = model(img_t)  # (1,1,H,W)

        pred_h, pred_w = logits.shape[2], logits.shape[3]
        gt = load_gt_resize(mask_path, pred_h, pred_w)
        if gt is None:
            print(f"  [警告] 無法載入 GT mask，略過: {mask_path}")
            continue
        gt = gt.to(device)

        metrics = calculate_metrics(logits, gt)
        iou = float(metrics["iou"])
        precision, recall = compute_precision_recall_from_logits(logits, gt)

        # FP/FN rate（沿用 compute_iou_precision_recall 的定義）
        pred_b = (torch.sigmoid(logits) > 0.5).float()
        gt_b = (gt > 0.5).float()
        tp = ((pred_b == 1) & (gt_b == 1)).sum().item()
        fp = ((pred_b == 1) & (gt_b == 0)).sum().item()
        fn = ((pred_b == 0) & (gt_b == 1)).sum().item()
        tn = ((pred_b == 0) & (gt_b == 0)).sum().item()
        smooth = 1e-6
        fp_rate = fp / (tn + fp + smooth) if tn + fp > 0 else 0.0
        fn_rate = fn / (tp + fn + smooth) if tp + fn > 0 else 0.0

        overlay_path = None
        if overlay_dir:
            try:
                # 還原到原圖尺寸做 overlay
                pred_np = torch.sigmoid(logits).squeeze().cpu().numpy()
                if pred_np.ndim == 3:
                    pred_np = pred_np[0]
                orig_w, orig_h = img_pil.size
                if pred_np.shape != (orig_h, orig_w):
                    pred_pil = Image.fromarray((np.clip(pred_np, 0, 1) * 255).astype(np.uint8))
                    pred_pil = pred_pil.resize((orig_w, orig_h), Image.NEAREST)
                    pred_np = np.array(pred_pil, dtype=np.float32) / 255.0

                gt_orig_img = Image.open(mask_path).convert("L")
                gt_orig_np = np.array(gt_orig_img, dtype=np.float32)
                if gt_orig_np.max() > 1.0:
                    gt_orig_np = gt_orig_np / 255.0

                overlay_img = create_overlay(image_path, pred_np, gt_orig_np, alpha=0.5)
                base = os.path.splitext(image_name)[0]
                overlay_filename = f"{camera_id}_{base}_dino_segmentation_overlay.png"
                overlay_path = os.path.join(overlay_dir, overlay_filename)
                overlay_img.save(overlay_path)
            except Exception as e:  # noqa: BLE001
                print(f"  [警告] 無法產生 overlay {image_path}: {e}")
                overlay_path = None

        results.append(
            {
                "camera_id": camera_id,
                "image_path": image_path,
                "mask_path": mask_path,
                "iou": iou,
                "precision": precision,
                "recall": recall,
                 "fp_rate": fp_rate,
                 "fn_rate": fn_rate,
                "overlay_path": overlay_path,
            }
        )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="在 data/image & data/mask 上評估 DINO + dino_segmentation（in-domain segmentation 模型）。"
    )
    parser.add_argument("--data_dir", type=str, default="data", help="資料根目錄（預設：data）")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=os.path.join(
            "outputs",
            "train_dino_segmentation_20260214_101339",
            "checkpoints",
            "best.pth",
        ),
        help="DINO segmentation checkpoint 路徑",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="輸出根目錄（overlay 會放在 output/dino_segmentation_overlays）",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="裝置（cpu 或 cuda；預設自動偵測）",
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    checkpoint = args.checkpoint
    output_dir = args.output_dir

    if not os.path.isfile(checkpoint):
        print(f"[錯誤] 找不到 checkpoint: {checkpoint}")
        sys.exit(1)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("  自訂資料集：DINO + dino_segmentation 評估")
    print("=" * 60)
    print(f"Data dir:    {data_dir}")
    print(f"Checkpoint:  {checkpoint}")
    print(f"Output dir:  {output_dir}")
    print(f"Device:      {device}")
    print()

    pairs = find_image_mask_pairs(data_dir=data_dir, image_subdir="image", mask_subdir="mask")
    print(f"共找到 {len(pairs)} 組 image-mask 配對。")
    for p in pairs:
        print(f"  - {p['image_name']}  (camera {p['camera_id']})")
    print()

    overlay_dir = os.path.join(output_dir, "dino_segmentation_overlays")
    results = evaluate_custom_dino_segmentation(
        pairs,
        checkpoint_path=checkpoint,
        device=device,
        image_size=(160, 160),
        overlay_dir=overlay_dir,
    )

    if not results:
        print("[錯誤] 沒有成功評估任何樣本。")
        return

    ious = [float(r["iou"]) for r in results]
    precisions = [float(r["precision"]) for r in results]
    recalls = [float(r["recall"]) for r in results]
    fp_rates = [float(r["fp_rate"]) for r in results]
    fn_rates = [float(r["fn_rate"]) for r in results]

    iou_mean = float(np.mean(ious))
    precision_mean = float(np.mean(precisions))
    recall_mean = float(np.mean(recalls))
    fp_rate_mean = float(np.mean(fp_rates))
    fn_rate_mean = float(np.mean(fn_rates))

    print()
    print("=" * 60)
    print("  Summary（DINO + dino_segmentation on custom data）")
    print("=" * 60)
    print(f"IoU_mean:      {iou_mean:.4f}")
    print(f"Precision_mean:{precision_mean:.4f}")
    print(f"Recall_mean:   {recall_mean:.4f}")
    print(f"FP_rate_mean:  {fp_rate_mean:.4f}")
    print(f"FN_rate_mean:  {fn_rate_mean:.4f}")
    print()
    print(f"Overlay 目錄: {overlay_dir}")
    print()


if __name__ == "__main__":
    main()

