"""
dino_clahm：DINOv2SegmentationModel + CLAHE 預處理（大霧/低對比用）

推論流程：
  1. 讀取 BGR 圖 (cv2.imread)
  2. LAB -> 僅 L channel CLAHE (clipLimit=2.0, tileGridSize=(8,8))
  3. 合併後轉回 RGB
  4. 轉 Tensor [0,1]，Resize 到 image_size
  5. ImageNet Normalize (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])（可 --no_normalize 關閉）
  6. 送入 DINOv2SegmentationModel 預測

使用清單：outputs/test_list_small.txt（10066×2、10870×2、1093×1，共 5 張）
預設產出 overlay（TP=綠、FP=紅、FN=藍）。

用法:
  python scripts/eval_dino_clahm_small.py
  python scripts/eval_dino_clahm_small.py --checkpoint outputs/train_dino_segmentation_20260214_101339/checkpoints/best.pth --overlay_dir outputs/dino_clahm_small_overlays
"""
import os
import sys
import argparse
import torch
import numpy as np
import cv2
from PIL import Image
from collections import defaultdict

_script_dir = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_script_dir)
sys.path.insert(0, _root)
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

from models import create_dinov2_segmentation_model
from utils.metrics import calculate_metrics
from utils.clahm_preprocess import bgr_to_tensor_clahm_normalized
from scripts.train_in_domain import load_list_file


def create_overlay(image_path, pred_mask, gt_mask=None, alpha=0.5):
    """原圖 + 預測/GT 疊圖：TP=綠、FP=紅、FN=藍。"""
    image = Image.open(image_path).convert('RGB')
    image_np = np.array(image, dtype=np.float32)
    h, w = image_np.shape[:2]
    pred = np.asarray(pred_mask, dtype=np.float32).squeeze()
    if pred.ndim == 3:
        pred = pred[0]
    if pred.shape[:2] != (h, w):
        pred = np.array(
            Image.fromarray((np.clip(pred, 0, 1) * 255).astype(np.uint8)).resize((w, h), Image.NEAREST),
            dtype=np.float32,
        ) / 255.0
    pred_b = pred > 0.5
    overlay = image_np.copy()
    if gt_mask is not None:
        gt = np.asarray(gt_mask, dtype=np.float32).squeeze()
        if gt.ndim == 3:
            gt = gt[0]
        if gt.shape[:2] != (h, w):
            gt = np.array(
                Image.fromarray((np.clip(gt, 0, 1) * 255).astype(np.uint8)).resize((w, h), Image.NEAREST),
                dtype=np.float32,
            ) / 255.0
        gt_b = gt > 0.5
        fn_mask = gt_b & (~pred_b)
        fp_mask = pred_b & (~gt_b)
        tp_mask = gt_b & pred_b
        blue = np.array([0, 100, 255], dtype=np.float32)
        red = np.array([255, 50, 50], dtype=np.float32)
        green = np.array([50, 255, 50], dtype=np.float32)
        fn_3d = np.stack([fn_mask] * 3, axis=-1)
        fp_3d = np.stack([fp_mask] * 3, axis=-1)
        tp_3d = np.stack([tp_mask] * 3, axis=-1)
        overlay = np.where(tp_3d, overlay * (1 - alpha * 0.3) + green * (alpha * 0.3), overlay)
        overlay = np.where(fn_3d, overlay * (1 - alpha) + blue * alpha, overlay)
        overlay = np.where(fp_3d, overlay * (1 - alpha * 0.9) + red * (alpha * 0.9), overlay)
    else:
        pred_3d = np.stack([pred_b] * 3, axis=-1)
        green = np.array([50, 255, 50], dtype=np.float32)
        overlay = np.where(pred_3d, overlay * (1 - alpha * 0.5) + green * (alpha * 0.5), overlay)
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    return Image.fromarray(overlay)


def load_gt_mask(mask_path, target_h, target_w):
    """載入 GT mask 並 resize 到 (target_h, target_w)，回傳 (1,1,H,W) tensor 0~1。"""
    if not os.path.exists(mask_path):
        return None
    img = Image.open(mask_path).convert('L')
    arr = np.array(img)
    if arr.max() > 1:
        arr = (arr > 127).astype(np.float32)
    else:
        arr = arr.astype(np.float32)
    t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
    t = torch.nn.functional.interpolate(t, size=(target_h, target_w), mode='nearest')
    return t


def main():
    parser = argparse.ArgumentParser(description='dino_clahm：DINOv2Segmentation + CLAHE 預處理')
    parser.add_argument(
        '--test_list',
        type=str,
        default=os.path.join(_root, 'outputs', 'test_list_small.txt'),
        help='測試清單（預設 5 張）',
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=os.path.join(_root, 'outputs', 'train_dino_segmentation_20260214_101339', 'checkpoints', 'best.pth'),
    )
    parser.add_argument('--data_dir', type=str, default=os.path.join(_root, 'data'))
    parser.add_argument('--image_size', type=int, nargs=2, default=[160, 160])
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--overlay_dir', type=str, default=None)
    parser.add_argument('--no_overlay', action='store_true')
    parser.add_argument(
        '--no_normalize',
        action='store_true',
        help='關閉 ImageNet Normalize（若 checkpoint 為 0~1 訓練可試用）',
    )
    parser.add_argument('--clahe_clip', type=float, default=2.0)
    parser.add_argument('--clahe_tile', type=int, nargs=2, default=[8, 8])
    args = parser.parse_args()

    if args.overlay_dir is None and not args.no_overlay:
        base = os.path.dirname(os.path.dirname(args.checkpoint))
        args.overlay_dir = os.path.join(base, 'dino_clahm_small_overlays')
    if not args.no_overlay:
        os.makedirs(args.overlay_dir, exist_ok=True)
        print(f'Overlay 輸出: {args.overlay_dir}')

    device = torch.device(args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'))
    image_size = tuple(args.image_size)
    use_normalize = not args.no_normalize
    clahe_tile = tuple(args.clahe_tile)

    # 載入模型
    print('載入模型: DINOv2SegmentationModel（dino_clahm：+ CLAHE 預處理）')
    model = create_dinov2_segmentation_model().to(device)
    ck = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ck['model_state_dict'])
    model.eval()

    print(f'前處理: CLAHE (clipLimit={args.clahe_clip}, tileGridSize={clahe_tile})', end='')
    print(f', Normalize={"ImageNet" if use_normalize else "關閉"}')
    print()

    test_list = load_list_file(args.test_list, base_data_dir=args.data_dir)
    print(f'Test 樣本數: {len(test_list)}')
    print()

    results = []
    by_camera = defaultdict(lambda: {'iou': [], 'fp': [], 'fn': [], 'dice': [], 'pixel_acc': []})

    for item in test_list:
        cid = item['camera_id']
        camera_folder = f'skyfinder_{cid}'
        img_path = os.path.join(args.data_dir, camera_folder, 'images', item['image'])
        mask_path = os.path.join(args.data_dir, camera_folder, 'masks', item['mask'])
        if not os.path.exists(img_path) or not os.path.exists(mask_path):
            print(f'  跳過（缺檔）: {img_path}')
            continue

        # BGR 讀取 -> CLAHE -> Tensor（含可選 Normalize）-> resize 已在 bgr_to_tensor 內
        bgr = cv2.imread(img_path)
        if bgr is None:
            print(f'  跳過（無法讀取）: {img_path}')
            continue
        h_orig, w_orig = bgr.shape[0], bgr.shape[1]

        if use_normalize:
            img_t = bgr_to_tensor_clahm_normalized(
                bgr,
                image_size=image_size,
                device=device,
                clip_limit=args.clahe_clip,
                tile_grid_size=clahe_tile,
            )
        else:
            from utils.clahm_preprocess import bgr_to_rgb_clahe
            rgb = bgr_to_rgb_clahe(bgr, clip_limit=args.clahe_clip, tile_grid_size=clahe_tile)
            rgb_float = rgb.astype(np.float32) / 255.0
            img_t = torch.from_numpy(rgb_float).permute(2, 0, 1).unsqueeze(0).to(device)
            if img_t.shape[2:] != image_size:
                img_t = torch.nn.functional.interpolate(
                    img_t, size=image_size, mode='bilinear', align_corners=False
                )

        with torch.no_grad():
            logits = model(img_t)

        pred_h, pred_w = logits.shape[2], logits.shape[3]
        gt = load_gt_mask(mask_path, pred_h, pred_w).to(device)
        if gt is None:
            continue

        metrics = calculate_metrics(logits, gt)
        iou = metrics['iou']
        dice = metrics['dice']
        pixel_acc = metrics['pixel_acc']
        pred_b = (torch.sigmoid(logits) > 0.5).float()
        gt_b = (gt > 0.5).float()
        fp = ((pred_b == 1) & (gt_b == 0)).sum().item()
        fn = ((pred_b == 0) & (gt_b == 1)).sum().item()
        n_neg = (gt_b == 0).sum().item()
        n_pos = (gt_b == 1).sum().item()
        fp_rate = fp / (n_neg + 1e-6)
        fn_rate = fn / (n_pos + 1e-6)

        results.append({
            'camera_id': cid,
            'image': item['image'],
            'iou': float(iou),
            'dice': float(dice),
            'pixel_acc': float(pixel_acc),
            'fp_rate': fp_rate,
            'fn_rate': fn_rate,
        })
        by_camera[cid]['iou'].append(float(iou))
        by_camera[cid]['dice'].append(float(dice))
        by_camera[cid]['pixel_acc'].append(float(pixel_acc))
        by_camera[cid]['fp'].append(fp_rate)
        by_camera[cid]['fn'].append(fn_rate)

        if not args.no_overlay and args.overlay_dir:
            pred_probs = torch.sigmoid(logits).cpu().numpy().squeeze()
            if pred_probs.ndim == 3:
                pred_probs = pred_probs[0]
            pred_orig = np.array(
                Image.fromarray((np.clip(pred_probs, 0, 1) * 255).astype(np.uint8)).resize((w_orig, h_orig), Image.NEAREST),
                dtype=np.float32,
            ) / 255.0
            gt_orig_img = Image.open(mask_path).convert('L')
            gt_orig_np = np.array(gt_orig_img)
            gt_orig = (gt_orig_np > 127).astype(np.float32) if gt_orig_np.max() > 1 else gt_orig_np.astype(np.float32)
            overlay_img = create_overlay(img_path, pred_orig, gt_orig, alpha=0.5)
            name_base = os.path.splitext(item['image'])[0]
            overlay_name = f"camera_{cid}_{name_base}_overlay.png"
            overlay_path = os.path.join(args.overlay_dir, overlay_name)
            overlay_img.save(overlay_path)

    print('=' * 60)
    print('  dino_clahm (DINOv2Segmentation + CLAHE) @ 小樣本 5 張')
    print('=' * 60)
    print()
    for r in results:
        print(f"  {r['camera_id']} / {r['image']}: IoU={r['iou']:.4f}, Dice={r['dice']:.4f}, FP_rate={r['fp_rate']:.4f}, FN_rate={r['fn_rate']:.4f}")
    print()
    print('Per-camera 平均:')
    print('-' * 60)
    for cid in sorted(by_camera.keys()):
        m = by_camera[cid]
        n = len(m['iou'])
        iou_mean = np.mean(m['iou'])
        dice_mean = np.mean(m['dice'])
        fp_mean = np.mean(m['fp'])
        fn_mean = np.mean(m['fn'])
        print(f"  {cid} (n={n}): IoU={iou_mean:.4f}, Dice={dice_mean:.4f}, FP_rate={fp_mean:.4f}, FN_rate={fn_mean:.4f}")
    print()
    overall_iou = np.mean([r['iou'] for r in results])
    overall_dice = np.mean([r['dice'] for r in results])
    print(f'整體平均 (5 張): IoU={overall_iou:.4f}, Dice={overall_dice:.4f}')
    print()
    print(f'Checkpoint: {args.checkpoint}')
    print(f'Test list:  {args.test_list}')
    if not args.no_overlay and args.overlay_dir:
        print(f'Overlay 目錄: {args.overlay_dir}')


if __name__ == '__main__':
    main()
