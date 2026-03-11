import os
import sys
import csv
import argparse
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

import numpy as np
from PIL import Image

import torch
from torchvision import transforms


_script_dir = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_script_dir)
sys.path.insert(0, _root)
sys.path.insert(0, _script_dir)
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")


# Grounding DINO + SAM utilities
from scripts.eval_grounding_dino_sam_final import (  # type: ignore
    load_grounding_dino_model,
    load_sam_model,
    predict_grounding_dino_sam,
    load_gt_mask,
    create_overlay as create_overlay_gdino_sam,
)

# CLIPSeg utilities
from scripts.eval_clipseg_final import (  # type: ignore
    load_clipseg_model,
    predict_clipseg,
    create_overlay as create_overlay_clipseg,
)

# DINOv2 + CNN utilities (只用 simple_cnn 方案，對應「DINOv2 + cnn」)
from scripts.eval_dino_dl_final import (  # type: ignore
    find_latest_checkpoint,
    load_dino_model,
    create_overlay as create_overlay_dino,
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

    # 先建立 camera_id -> mask_path 對應
    mask_map: Dict[str, str] = {}
    for fname in os.listdir(mask_dir):
        name, ext = os.path.splitext(fname)
        if not ext.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
            continue
        # 假設 mask 命名為 <camera>_mask
        if name.endswith("_mask"):
            cam_id = name.replace("_mask", "")
        else:
            cam_id = name
        mask_map[cam_id] = os.path.join(mask_dir, fname)

    pairs: List[Dict[str, str]] = []
    valid_img_ext = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    for fname in os.listdir(image_dir):
        name, ext = os.path.splitext(fname)
        if ext.lower() not in valid_img_ext:
            continue
        # image 命名: <camera>_<frame>
        if "_" in name:
            cam_id = name.split("_")[0]
        else:
            cam_id = name

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


def compute_iou_precision_recall(
    pred_mask: np.ndarray,
    gt_mask: Optional[np.ndarray],
    has_gt: bool = True,
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[float]]:
    """
    計算單張圖的 IoU / Precision / Recall / FP rate / FN rate。

    其中：
        FP rate = FP / (TN + FP)
        FN rate = FN / (TP + FN)
    """
    smooth = 1e-6

    if not has_gt or gt_mask is None:
        return None, None, None, None, None

    # 正規化與二值化
    if pred_mask.max() > 1.0:
        pred_mask = pred_mask.astype(np.float32) / 255.0
    else:
        pred_mask = pred_mask.astype(np.float32)

    if gt_mask.max() > 1.0:
        gt_mask = gt_mask.astype(np.float32) / 255.0
    else:
        gt_mask = gt_mask.astype(np.float32)

    # 尺寸對齊：pred → gt 尺寸
    if pred_mask.shape != gt_mask.shape:
        pred_img = Image.fromarray((pred_mask * 255).astype(np.uint8))
        pred_img = pred_img.resize((gt_mask.shape[1], gt_mask.shape[0]), Image.NEAREST)
        pred_mask = np.array(pred_img, dtype=np.float32) / 255.0

    pred_b = (pred_mask > 0.5)
    gt_b = (gt_mask > 0.5)

    tp = float((pred_b & gt_b).sum())
    fp = float((pred_b & ~gt_b).sum())
    fn = float((~pred_b & gt_b).sum())
    tn = float((~pred_b & ~gt_b).sum())

    iou = tp / (tp + fp + fn + smooth) if tp + fp + fn > 0 else 0.0
    precision = tp / (tp + fp + smooth) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn + smooth) if tp + fn > 0 else 0.0
    fp_rate = fp / (tn + fp + smooth) if tn + fp > 0 else 0.0
    fn_rate = fn / (tp + fn + smooth) if tp + fn > 0 else 0.0

    return float(iou), float(precision), float(recall), float(fp_rate), float(fn_rate)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def evaluate_grounding_dino_sam(
    pairs: List[Dict[str, str]],
    device: torch.device,
    output_dir: str,
    grounding_dino_model_id: str = "IDEA-Research/grounding-dino-tiny",
    sam_model_type: str = "sam2_hiera_small",
    text_prompt: Optional[str] = None,
    box_threshold: float = 0.4,
    text_threshold: float = 0.35,
    overlay_subdir: str = "grounding_dino_sam_overlays",
) -> List[Dict[str, object]]:
    """在自訂 data/image & data/mask 上評估 Grounding DINO + SAM。"""
    print("=== 評估 Grounding DINO + SAM（custom data）===")
    print(f"  Model: {grounding_dino_model_id}, SAM: {sam_model_type}, device: {device}")

    overlay_dir = os.path.join(output_dir, overlay_subdir)
    ensure_dir(overlay_dir)

    # 載入模型
    processor, model = load_grounding_dino_model(
        model_id=grounding_dino_model_id,
        device=device,
    )
    sam_predictor = load_sam_model(
        model_type=sam_model_type,
        device=device,
        sam_legacy_checkpoint=None,
    )

    results: List[Dict[str, object]] = []

    for item in pairs:
        image_path = item["image_path"]
        mask_path = item["mask_path"]
        camera_id = item["camera_id"]
        image_name = item["image_name"]

        if not os.path.exists(image_path):
            print(f"  [警告] 找不到影像: {image_path}")
            continue

        # 推論
        pred_mask = predict_grounding_dino_sam(
            processor,
            model,
            sam_predictor,
            image_path,
            text_prompt=text_prompt,
            device=device,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )

        gt_mask, has_gt = load_gt_mask(image_path, mask_path)
        iou, precision, recall, fp_rate, fn_rate = compute_iou_precision_recall(pred_mask, gt_mask, has_gt)

        # overlay
        overlay_path = None
        try:
            overlay_img = create_overlay_gdino_sam(image_path, pred_mask, gt_mask)
            base = os.path.splitext(image_name)[0]
            overlay_name = f"{camera_id}_{base}_grounding_dino_sam_overlay.png"
            overlay_path = os.path.join(overlay_dir, overlay_name)
            overlay_img.save(overlay_path)
        except Exception as e:  # noqa: BLE001
            print(f"  [警告] 無法產生 GroundingDINO+SAM overlay ({image_path}): {e}")

        results.append(
            {
                "method": "GroundingDINO+SAM",
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


def evaluate_clipseg(
    pairs: List[Dict[str, str]],
    device: torch.device,
    output_dir: str,
    model_id: str = "CIDAS/clipseg-rd64-refined",
) -> List[Dict[str, object]]:
    """在自訂 data/image & data/mask 上評估 CLIPSeg。"""
    print("=== 評估 CLIPSeg（custom data）===")
    print(f"  Model: {model_id}, device: {device}")

    overlay_dir = os.path.join(output_dir, "clipseg_overlays")
    ensure_dir(overlay_dir)

    processor, model = load_clipseg_model(model_id=model_id, device=str(device))

    results: List[Dict[str, object]] = []

    for item in pairs:
        image_path = item["image_path"]
        mask_path = item["mask_path"]
        camera_id = item["camera_id"]
        image_name = item["image_name"]

        if not os.path.exists(image_path):
            print(f"  [警告] 找不到影像: {image_path}")
            continue

        # 推論
        pred_mask = predict_clipseg(
            processor,
            model,
            image_path,
            device=str(device),
            use_dynamic_threshold=True,
            use_spatial_prior=True,
            use_morphology_open=True,
            use_multi_prompt_contrast=True,
            morph_open_kernel_size=5,
        )

        gt_mask, has_gt = load_gt_mask(image_path, mask_path)
        iou, precision, recall, fp_rate, fn_rate = compute_iou_precision_recall(pred_mask, gt_mask, has_gt)

        # overlay
        overlay_path = None
        try:
            overlay_img = create_overlay_clipseg(image_path, pred_mask, gt_mask)
            base = os.path.splitext(image_name)[0]
            overlay_name = f"{camera_id}_{base}_clipseg_overlay.png"
            overlay_path = os.path.join(overlay_dir, overlay_name)
            overlay_img.save(overlay_path)
        except Exception as e:  # noqa: BLE001
            print(f"  [警告] 無法產生 CLIPSeg overlay ({image_path}): {e}")

        results.append(
            {
                "method": "CLIPSeg",
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


def evaluate_dinov2_cnn(
    pairs: List[Dict[str, str]],
    device: torch.device,
    output_dir: str,
    image_size: Tuple[int, int] = (256, 256),
    checkpoint_path: Optional[str] = None,
) -> List[Dict[str, object]]:
    """
    在自訂 data/image & data/mask 上評估 DINOv2 + 簡單 CNN Decoder。
    使用 scripts.eval_dino_dl_final 中的 simple_cnn 方案與 checkpoint。
    """
    print("=== 評估 DINOv2 + CNN（custom data）===")

    overlay_dir = os.path.join(output_dir, "dinov2_cnn_overlays")
    ensure_dir(overlay_dir)

    # 預設使用使用者提供的 dino_cnn/best.pth
    default_ckpt = os.path.join("outputs", "dino_cnn", "best.pth")
    if checkpoint_path is None:
        checkpoint_path = default_ckpt

    if not os.path.exists(checkpoint_path):
        print(f"  [警告] 找不到 DINO-CNN checkpoint: {checkpoint_path}，略過此方法。")
        return []

    print(f"  使用 checkpoint: {checkpoint_path}")
    model = load_dino_model("simple_cnn", checkpoint_path, device)
    model.to(device)
    model.eval()

    # ImageNet normalization（與訓練一致）
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    to_tensor = transforms.ToTensor()

    results: List[Dict[str, object]] = []

    for item in pairs:
        image_path = item["image_path"]
        mask_path = item["mask_path"]
        camera_id = item["camera_id"]
        image_name = item["image_name"]

        if not os.path.exists(image_path):
            print(f"  [警告] 找不到影像: {image_path}")
            continue

        # 載入並 resize 圖片到訓練尺寸
        image = Image.open(image_path).convert("RGB")
        image_resized = image.resize(image_size, Image.BILINEAR)
        image_tensor = to_tensor(image_resized)
        image_tensor = normalize(image_tensor)
        image_tensor = image_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(image_tensor)  # (1, 1, H, W)
            pred_probs = torch.sigmoid(outputs)[0, 0].cpu().numpy()

        # GT mask
        gt_mask, has_gt = load_gt_mask(image_path, mask_path)
        iou, precision, recall = compute_iou_precision_recall(pred_probs, gt_mask, has_gt)

        # overlay（先把 pred mask resize 回原圖尺寸，再呼叫現有 overlay util）
        try:
            orig_w, orig_h = image.size
            pred_img = Image.fromarray((pred_probs * 255).astype(np.uint8))
            pred_img = pred_img.resize((orig_w, orig_h), Image.NEAREST)
            pred_resized = np.array(pred_img, dtype=np.float32) / 255.0

            overlay_img = create_overlay_dino(image_path, pred_resized, gt_mask)
            base = os.path.splitext(image_name)[0]
            overlay_name = f"{camera_id}_{base}_dinov2_cnn_overlay.png"
            overlay_path = os.path.join(overlay_dir, overlay_name)
            overlay_img.save(overlay_path)
        except Exception as e:  # noqa: BLE001
            print(f"  [警告] 無法產生 DINOv2+CNN overlay ({image_path}): {e}")
            overlay_path = None

        results.append(
            {
                "method": "DINOv2+CNN",
                "camera_id": camera_id,
                "image_path": image_path,
                "mask_path": mask_path,
                "iou": iou,
                "precision": precision,
                "recall": recall,
                "overlay_path": overlay_path,
            }
        )

    return results


def aggregate_results(results: List[Dict[str, object]]) -> Dict[str, Dict[str, float]]:
    """
    依 method 聚合 IoU / Precision / Recall 的平均值。
    僅對有 GT 的樣本計算平均（iou/precision/recall 皆非 None 時）。
    """
    per_method: Dict[str, Dict[str, float]] = {}

    grouped: Dict[str, List[Tuple[float, float, float, float, float]]] = defaultdict(list)
    for r in results:
        method = str(r["method"])
        iou = r.get("iou")
        precision = r.get("precision")
        recall = r.get("recall")
        fp_rate = r.get("fp_rate")
        fn_rate = r.get("fn_rate")
        if iou is None or precision is None or recall is None or fp_rate is None or fn_rate is None:
            continue
        grouped[method].append(
            (float(iou), float(precision), float(recall), float(fp_rate), float(fn_rate))
        )

    for method, vals in grouped.items():
        if not vals:
            continue
        ious, precisions, recalls, fp_rates, fn_rates = zip(*vals)
        per_method[method] = {
            "iou_mean": float(np.mean(ious)),
            "precision_mean": float(np.mean(precisions)),
            "recall_mean": float(np.mean(recalls)),
            "fp_rate_mean": float(np.mean(fp_rates)),
            "fn_rate_mean": float(np.mean(fn_rates)),
        }

    return per_method


def save_results_csv(
    all_results: List[Dict[str, object]],
    summary: Dict[str, Dict[str, float]],
    output_dir: str,
    filename: str = "custom_data_metrics.csv",
) -> str:
    """將 per-image 與 per-method 統計寫入 CSV，並嘗試同步輸出為 Excel。"""
    ensure_dir(output_dir)
    csv_path = os.path.join(output_dir, filename)

    fieldnames = [
        "method",
        "camera_id",
        "image_path",
        "mask_path",
        "iou",
        "precision",
        "recall",
        "fp_rate",
        "fn_rate",
        "overlay_path",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(fieldnames)
        for r in all_results:
            writer.writerow(
                [
                    r.get("method"),
                    r.get("camera_id"),
                    r.get("image_path"),
                    r.get("mask_path"),
                    f"{r.get('iou'):.6f}" if r.get("iou") is not None else "",
                    f"{r.get('precision'):.6f}" if r.get("precision") is not None else "",
                    f"{r.get('recall'):.6f}" if r.get("recall") is not None else "",
                    f"{r.get('fp_rate'):.6f}" if r.get("fp_rate") is not None else "",
                    f"{r.get('fn_rate'):.6f}" if r.get("fn_rate") is not None else "",
                    r.get("overlay_path") or "",
                ]
            )

        # 在 CSV 末尾附上 summary（以空行分隔）
        writer.writerow([])
        writer.writerow(
            ["method", "iou_mean", "precision_mean", "recall_mean", "fp_rate_mean", "fn_rate_mean"]
        )
        for method, m in summary.items():
            writer.writerow(
                [
                    method,
                    f"{m['iou_mean']:.6f}",
                    f"{m['precision_mean']:.6f}",
                    f"{m['recall_mean']:.6f}",
                    f"{m['fp_rate_mean']:.6f}",
                    f"{m['fn_rate_mean']:.6f}",
                ]
            )

    # 另外嘗試匯出為 Excel（若環境中有 pandas）
    try:
        import pandas as pd  # type: ignore

        df = pd.read_csv(csv_path)
        xlsx_path = os.path.join(output_dir, filename.replace(".csv", ".xlsx"))
        df.to_excel(xlsx_path, index=False)
        print(f"  亦已輸出 Excel：{xlsx_path}")
    except ImportError:
        print("  [提示] 環境中未安裝 pandas，僅輸出 CSV。若需要 .xlsx，請先安裝 pandas。")
        xlsx_path = ""

    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="在 data/image & data/mask 上比較 Grounding DINO+SAM / CLIPSeg / DINOv2+CNN，並產生 overlay 與 IoU/Precision/Recall。"
    )
    parser.add_argument("--data_dir", type=str, default="data", help="資料根目錄（預設：data）")
    parser.add_argument("--output_dir", type=str, default="output", help="輸出根目錄（預設：output）")
    parser.add_argument("--device", type=str, default=None, help="裝置（cpu 或 cuda；預設自動偵測）")
    # 允許只跑部分方法
    parser.add_argument("--no_grounding", action="store_true", help="關閉 Grounding DINO+SAM 評估")
    parser.add_argument("--no_clipseg", action="store_true", help="關閉 CLIPSeg 評估")
    parser.add_argument("--no_dino", action="store_true", help="關閉 DINOv2+CNN 評估")
    args = parser.parse_args()

    data_dir = args.data_dir
    output_dir = args.output_dir

    # 裝置
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("  自訂資料集：三種方法比較 (Grounding DINO+SAM / CLIPSeg / DINOv2+CNN)")
    print("=" * 60)
    print(f"Data dir:    {data_dir}")
    print(f"Output dir:  {output_dir}")
    print(f"Device:      {device}")
    print()

    # 掃描資料
    pairs = find_image_mask_pairs(data_dir=data_dir, image_subdir="image", mask_subdir="mask")
    print(f"共找到 {len(pairs)} 組 image-mask 配對。")
    for p in pairs:
        print(f"  - {p['image_name']}  (camera {p['camera_id']})")
    print()

    all_results: List[Dict[str, object]] = []

    # 1. Grounding DINO + SAM
    if not args.no_grounding:
        try:
            gdino_results = evaluate_grounding_dino_sam(pairs, device=device, output_dir=output_dir)
            all_results.extend(gdino_results)
        except Exception as e:  # noqa: BLE001
            print(f"[錯誤] Grounding DINO + SAM 評估失敗: {e}")

    # 2. CLIPSeg
    if not args.no_clipseg:
        try:
            clipseg_results = evaluate_clipseg(pairs, device=device, output_dir=output_dir)
            all_results.extend(clipseg_results)
        except Exception as e:  # noqa: BLE001
            print(f"[錯誤] CLIPSeg 評估失敗: {e}")

    # 3. DINOv2 + CNN（simple_cnn）
    if not args.no_dino:
        try:
            dinov2_results = evaluate_dinov2_cnn(pairs, device=device, output_dir=output_dir)
            all_results.extend(dinov2_results)
        except Exception as e:  # noqa: BLE001
            print(f"[錯誤] DINOv2 + CNN 評估失敗: {e}")

    if not all_results:
        print("[錯誤] 三種方法皆無法成功評估，請檢查環境依賴（torch、transformers、sam2、timm 等）。")
        return

    # 聚合
    summary = aggregate_results(all_results)

    # 儲存 CSV
    csv_path = save_results_csv(all_results, summary, output_dir=output_dir)

    print()
    print("=" * 60)
    print("  Summary（IoU / Precision / Recall 平均）")
    print("=" * 60)
    for method, m in summary.items():
        print(
            f"{method:18s} | IoU: {m['iou_mean']:.4f} | "
            f"Precision: {m['precision_mean']:.4f} | Recall: {m['recall_mean']:.4f}"
        )
    print()
    print(f"詳細 per-image 指標與 summary 已寫入: {csv_path}")
    print(f"Overlay 輸出目錄：")
    print(f"  - {os.path.join(output_dir, 'grounding_dino_sam_overlays')}")
    print(f"  - {os.path.join(output_dir, 'clipseg_overlays')}")
    print(f"  - {os.path.join(output_dir, 'dinov2_cnn_overlays')}")
    print()


if __name__ == "__main__":
    main()

