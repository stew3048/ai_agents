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

# DINO segmentation utilities（使用 in-domain segmentation 模型）
from models import create_dinov2_segmentation_model  # type: ignore
from utils.metrics import calculate_metrics  # type: ignore
from scripts.eval_dino_segmentation_small import (  # type: ignore
    load_gt_mask as load_gt_resize,
    create_overlay as create_overlay_dino_seg,
)


def load_pairs_from_testlist(
    test_list_path: str,
    base_data_dir: str,
) -> List[Dict[str, str]]:
    """
    從 test_list.txt 讀取 image-mask 配對。
    test_list.txt 每行格式：skyfinder_10066/images/046.jpg
    對應 mask 路徑：       skyfinder_10066/masks/046.png
    base_data_dir 為實際存放 skyfinder_xxx/ 的根目錄（例如 data/data）。
    """
    if not os.path.exists(test_list_path):
        raise FileNotFoundError(f"找不到 test_list 檔案: {test_list_path}")

    pairs: List[Dict[str, str]] = []
    skipped = 0

    with open(test_list_path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]

    for rel_path in lines:
        # rel_path: skyfinder_10066/images/046.jpg
        parts = rel_path.replace("\\", "/").split("/")
        if len(parts) < 3:
            print(f"  [警告] 無法解析路徑格式，略過: {rel_path}")
            skipped += 1
            continue

        camera_folder = parts[0]   # skyfinder_10066
        frame_file = parts[-1]     # 046.jpg
        frame_stem = os.path.splitext(frame_file)[0]  # 046

        image_path = os.path.join(base_data_dir, camera_folder, "images", frame_file)
        mask_file = frame_stem + ".png"
        mask_path = os.path.join(base_data_dir, camera_folder, "masks", mask_file)

        if not os.path.exists(image_path):
            print(f"  [警告] 找不到影像，略過: {image_path}")
            skipped += 1
            continue
        if not os.path.exists(mask_path):
            print(f"  [警告] 找不到 mask，略過: {mask_path}")
            skipped += 1
            continue

        camera_id = camera_folder  # e.g. skyfinder_10066
        image_name = f"{camera_folder}_{frame_file}"  # e.g. skyfinder_10066_046.jpg

        pairs.append(
            {
                "camera_id": camera_id,
                "image_path": image_path,
                "mask_path": mask_path,
                "image_name": image_name,
            }
        )

    print(f"  從 test_list 讀取 {len(pairs)} 組配對（略過 {skipped} 筆）。")
    return pairs


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


def evaluate_dino_segmentation(
    pairs: List[Dict[str, str]],
    device: torch.device,
    output_dir: str,
    image_size: Tuple[int, int] = (160, 160),
    checkpoint_path: Optional[str] = None,
) -> List[Dict[str, object]]:
    """
    在自訂 data/image & data/mask 上評估 DINO + dino_segmentation（in-domain segmentation 模型）。
    與 eval_dino_segmentation_small / run_dino_segmentation 的設定一致：
      - 輸入為 [0,1] 範圍，不做 ImageNet normalize
      - 影像 resize 到 image_size
    """
    print("=== 評估 DINO + dino_segmentation（custom data）===")

    overlay_dir = os.path.join(output_dir, "dino_segmentation_overlays")
    ensure_dir(overlay_dir)

    # 預設使用最新的 train_dino_segmentation_* best.pth
    if checkpoint_path is None:
        outputs_dir = os.path.join(_root, "outputs")
        from pathlib import Path as _Path

        candidates = list(_Path(outputs_dir).glob("train_dino_segmentation_*/checkpoints/best.pth"))
        if candidates:
            checkpoint_path = str(max(candidates, key=os.path.getmtime))

    if not checkpoint_path or not os.path.exists(checkpoint_path):
        print(f"  [警告] 找不到 DINO segmentation checkpoint: {checkpoint_path}，略過此方法。")
        return []

    print(f"  使用 checkpoint: {checkpoint_path}")
    model = create_dinov2_segmentation_model().to(device)
    ck = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ck.get("model_state_dict", ck), strict=False)
    model.eval()

    results: List[Dict[str, object]] = []

    for item in pairs:
        image_path = item["image_path"]
        mask_path = item["mask_path"]
        camera_id = item["camera_id"]
        image_name = item["image_name"]

        if not os.path.exists(image_path) or not os.path.exists(mask_path):
            print(f"  [警告] 缺檔，略過: {image_path}")
            continue

        img_pil = Image.open(image_path).convert("RGB")
        img_np = np.array(img_pil).astype(np.float32) / 255.0
        img_np = img_np.transpose(2, 0, 1)
        img_t = torch.from_numpy(img_np).unsqueeze(0).to(device)
        if img_t.shape[2:] != image_size:
            img_t = torch.nn.functional.interpolate(
                img_t, size=image_size, mode="bilinear", align_corners=False
            )

        with torch.no_grad():
            logits = model(img_t)

        pred_h, pred_w = logits.shape[2], logits.shape[3]
        gt = load_gt_resize(mask_path, pred_h, pred_w)
        if gt is None:
            print(f"  [警告] 無法載入 GT mask，略過: {mask_path}")
            continue
        gt = gt.to(device)

        m = calculate_metrics(logits, gt)
        iou = float(m["iou"])

        # precision / recall / FP rate / FN rate
        pred_b = (torch.sigmoid(logits) > 0.5).float()
        gt_b = (gt > 0.5).float()
        tp = ((pred_b == 1) & (gt_b == 1)).sum().item()
        fp = ((pred_b == 1) & (gt_b == 0)).sum().item()
        fn = ((pred_b == 0) & (gt_b == 1)).sum().item()
        tn = ((pred_b == 0) & (gt_b == 0)).sum().item()
        smooth = 1e-6
        precision = tp / (tp + fp + smooth) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn + smooth) if tp + fn > 0 else 0.0
        fp_rate = fp / (tn + fp + smooth) if tn + fp > 0 else 0.0
        fn_rate = fn / (tp + fn + smooth) if tp + fn > 0 else 0.0

        overlay_path = None
        try:
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

            overlay_img = create_overlay_dino_seg(image_path, pred_np, gt_orig_np, alpha=0.5)
            base = os.path.splitext(image_name)[0]
            overlay_name = f"{camera_id}_{base}_dino_segmentation_overlay.png"
            overlay_path = os.path.join(overlay_dir, overlay_name)
            overlay_img.save(overlay_path)
        except Exception as e:  # noqa: BLE001
            print(f"  [警告] 無法產生 DINO segmentation overlay ({image_path}): {e}")
            overlay_path = None

        results.append(
            {
                "method": "DINO-segmentation",
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


def load_existing_csv_results(csv_path: str) -> List[Dict[str, object]]:
    """
    讀取已存在的 CSV 檔，只取 per-image 明細列（跳過 summary 段落）。
    CSV 結構：header → data rows → 空行 → summary header → summary rows
    """
    rows: List[Dict[str, object]] = []
    if not os.path.exists(csv_path):
        print(f"  [警告] 找不到既有 CSV：{csv_path}，將建立新檔。")
        return rows

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # 只在 method 為空時停止（代表到達分隔空行）
            method = row.get("method", "").strip()
            if not method:
                break
            # 跳過 summary header 行（method="method"）或 summary 資料行（camera_id 看起來是數字）
            if method == "method":
                continue
            # 還原數值型態（None 表示該欄位無值或評估失敗）
            try:
                iou = float(row["iou"]) if row.get("iou", "").strip() else None
                precision = float(row["precision"]) if row.get("precision", "").strip() else None
                recall = float(row["recall"]) if row.get("recall", "").strip() else None
                fp_rate = float(row["fp_rate"]) if row.get("fp_rate", "").strip() else None
                fn_rate = float(row["fn_rate"]) if row.get("fn_rate", "").strip() else None
            except (ValueError, KeyError):
                continue
            rows.append(
                {
                    "method": method,
                    "camera_id": row.get("camera_id", ""),
                    "image_path": row.get("image_path", ""),
                    "mask_path": row.get("mask_path", ""),
                    "iou": iou,
                    "precision": precision,
                    "recall": recall,
                    "fp_rate": fp_rate,
                    "fn_rate": fn_rate,
                    "overlay_path": row.get("overlay_path", "") or "",
                }
            )
    print(f"  從既有 CSV 載入 {len(rows)} 筆 per-image 紀錄。")
    return rows


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
        description="在自訂資料集上比較 Grounding DINO+SAM / CLIPSeg / DINO-segmentation，並產生 overlay 與指標報告。"
    )
    parser.add_argument("--data_dir", type=str, default="data", help="舊版資料根目錄（掃描 image/ 與 mask/ 子目錄，預設：data）")
    parser.add_argument("--output_dir", type=str, default="output", help="輸出根目錄（預設：output）")
    parser.add_argument("--device", type=str, default=None, help="裝置（cpu 或 cuda；預設自動偵測）")
    # test_list 模式
    parser.add_argument("--test_list", type=str, default=None,
                        help="test_list.txt 路徑（每行格式: skyfinder_xxx/images/yyy.jpg）")
    parser.add_argument("--base_data_dir", type=str, default=None,
                        help="test_list 模式下 skyfinder_xxx/ 的根目錄（例如 data/data）")
    # 允許只跑部分方法
    parser.add_argument("--no_grounding", action="store_true", help="關閉 Grounding DINO+SAM 評估")
    parser.add_argument("--no_clipseg", action="store_true", help="關閉 CLIPSeg 評估")
    parser.add_argument("--no_dino", action="store_true", help="關閉 DINO segmentation 評估")
    # 補評估專用
    parser.add_argument("--filter_camera", type=str, default=None,
                        help="只評估指定 camera（例如 skyfinder_3888），其他略過；預設全跑")
    parser.add_argument("--merge_existing", type=str, default=None,
                        help="指定一份已存在的 CSV，將新結果合併後重算平均並覆寫該 CSV+Excel")
    args = parser.parse_args()

    output_dir = args.output_dir

    # 裝置
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("  資料集評估：三種方法比較 (Grounding DINO+SAM / CLIPSeg / DINO-segmentation)")
    print("=" * 60)
    print(f"Output dir:  {output_dir}")
    print(f"Device:      {device}")
    print()

    # 讀取資料配對
    if args.test_list:
        base_data_dir = args.base_data_dir or os.path.join("data", "data")
        print(f"模式: test_list  →  {args.test_list}")
        print(f"Base data dir: {base_data_dir}")
        print()
        pairs = load_pairs_from_testlist(
            test_list_path=args.test_list,
            base_data_dir=base_data_dir,
        )
    else:
        data_dir = args.data_dir
        print(f"模式: 掃描資料夾  →  {data_dir}")
        print()
        pairs = find_image_mask_pairs(data_dir=data_dir, image_subdir="image", mask_subdir="mask")

    # filter_camera：只保留指定 camera
    if args.filter_camera:
        before = len(pairs)
        pairs = [p for p in pairs if p["camera_id"] == args.filter_camera]
        print(f"[filter_camera] 只保留 camera={args.filter_camera}，{before} → {len(pairs)} 組")
    print(f"共 {len(pairs)} 組 image-mask 配對。")
    for p in pairs[:10]:
        print(f"  - {p['image_name']}  (camera {p['camera_id']})")
    if len(pairs) > 10:
        print(f"  ... 共 {len(pairs)} 組（只顯示前 10 筆）")
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

    # 3. DINO segmentation
    if not args.no_dino:
        try:
            dino_seg_results = evaluate_dino_segmentation(pairs, device=device, output_dir=output_dir)
            all_results.extend(dino_seg_results)
        except Exception as e:  # noqa: BLE001
            print(f"[錯誤] DINO segmentation 評估失敗: {e}")

    if not all_results:
        print("[錯誤] 三種方法皆無法成功評估，請檢查環境依賴（torch、transformers、sam2、timm 等）。")
        return

    # merge_existing：載入舊結果合併後重算
    if args.merge_existing and os.path.exists(args.merge_existing):
        print(f"[merge] 載入既有結果：{args.merge_existing}")
        existing = load_existing_csv_results(args.merge_existing)
        # 避免重複：移除 existing 中與 new results 同 (method, image_path) 的舊紀錄
        new_keys = {(str(r["method"]), str(r["image_path"])) for r in all_results}
        existing = [r for r in existing if (str(r["method"]), str(r["image_path"])) not in new_keys]
        all_results = existing + all_results
        print(f"[merge] 合併後共 {len(all_results)} 筆（舊 {len(existing)} + 新 {len(all_results)-len(existing)}）")

    # 聚合
    summary = aggregate_results(all_results)

    # 決定輸出 CSV 路徑
    if args.merge_existing:
        # 直接覆寫原本的 CSV（其 output_dir 由 merge_existing 路徑推算）
        merge_dir = os.path.dirname(args.merge_existing)
        merge_filename = os.path.basename(args.merge_existing)
        csv_path = save_results_csv(all_results, summary, output_dir=merge_dir, filename=merge_filename)
    else:
        csv_filename = "test_data_metrics.csv" if args.test_list else "custom_data_metrics.csv"
        csv_path = save_results_csv(all_results, summary, output_dir=output_dir, filename=csv_filename)

    print()
    print("=" * 60)
    print("  Summary（IoU / Precision / Recall 平均）")
    print("=" * 60)
    for method, m in summary.items():
        print(
            f"{method:18s} | IoU: {m['iou_mean']:.4f} | "
            f"Precision: {m['precision_mean']:.4f} | Recall: {m['recall_mean']:.4f} | "
            f"FP_rate: {m['fp_rate_mean']:.4f} | FN_rate: {m['fn_rate_mean']:.4f}"
        )
    print()
    print(f"詳細 per-image 指標與 summary 已寫入: {csv_path}")
    print(f"Overlay 輸出目錄：")
    print(f"  - {os.path.join(output_dir, 'grounding_dino_sam_overlays')}")
    print(f"  - {os.path.join(output_dir, 'clipseg_overlays')}")
    print(f"  - {os.path.join(output_dir, 'dino_segmentation_overlays')}")
    print()


if __name__ == "__main__":
    main()

