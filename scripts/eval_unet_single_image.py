"""
對「單張」影像跑與 SAM 比較用的 U-Net，輸出 IoU / FP rate / FN rate。

用法:
  python scripts/eval_unet_single_image.py --camera_id 3888 --image_id 95
  python scripts/eval_unet_single_image.py --checkpoint path/to/best.pth --camera_id 3888 --image_id 95
"""

import os
import sys
import argparse
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

import torch
import numpy as np
from PIL import Image

from models import create_unet_model
import torchvision.transforms.functional as TF


def find_latest_checkpoint():
    """找最新的 train_multi_camera_* 下的 checkpoints/best.pth"""
    outputs_dir = Path('outputs')
    candidates = list(outputs_dir.glob('train_multi_camera_*/checkpoints/best.pth'))
    if not candidates:
        return None
    return str(max(candidates, key=os.path.getmtime))


def load_model(checkpoint_path, device='cpu'):
    model = create_unet_model(n_channels=3, n_classes=1)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model


def preprocess(image_path, image_size=(160, 160)):
    image = Image.open(image_path).convert('RGB')
    original_size = image.size
    image = image.resize((image_size[1], image_size[0]), Image.BILINEAR)
    tensor = TF.to_tensor(image).unsqueeze(0)
    return tensor, original_size


def predict(model, image_tensor, device, threshold=0.5):
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        logits = model(image_tensor)
        prob = torch.sigmoid(logits)
        mask = (prob > threshold).float()
    return mask.squeeze().cpu().numpy(), prob.squeeze().cpu().numpy()


def load_gt_mask(image_path, camera_id, image_id, has_sky=True):
    if not has_sky:
        img = Image.open(image_path).convert('RGB')
        w, h = img.size
        return np.zeros((h, w), dtype=np.float32), True
    path_parts = image_path.replace('\\', '/').split('/')
    camera_folder = None
    for part in path_parts:
        if part.startswith('skyfinder_'):
            camera_folder = part
            break
    if camera_folder is None:
        camera_folder = f'skyfinder_{camera_id}'
    image_id_str = str(image_id)
    mask_filename = f'{int(image_id_str):03d}.png'
    mask_path = os.path.join('data', camera_folder, 'masks', mask_filename)
    if not os.path.exists(mask_path):
        image_dir = os.path.dirname(image_path)
        if 'images' in image_dir:
            mask_dir = image_dir.replace('images', 'masks')
            mask_path = os.path.join(mask_dir, mask_filename)
    if not os.path.exists(mask_path):
        return None, False
    mask_img = Image.open(mask_path).convert('L')
    mask_np = np.array(mask_img, dtype=np.float32) / 255.0
    return mask_np, True


def calculate_metrics(pred_mask, gt_mask=None, has_gt=True, smooth=1e-6):
    pred_b = (pred_mask > 0.5).astype(np.float32)
    total_pixels = pred_mask.size
    pred_positive_ratio = pred_b.sum() / (total_pixels + smooth)
    metrics = {'dl_pred_positive_ratio': float(pred_positive_ratio)}
    if not has_gt or gt_mask is None:
        metrics['dl_iou'] = None
        metrics['dl_fn_rate'] = None
        metrics['dl_fp_rate'] = float(pred_positive_ratio)
        return metrics
    gt_b = (gt_mask > 0.5).astype(np.float32)
    if pred_b.shape != gt_b.shape:
        pred_img = Image.fromarray((pred_b * 255).astype(np.uint8))
        pred_img = pred_img.resize((gt_b.shape[1], gt_b.shape[0]), Image.NEAREST)
        pred_b = np.array(pred_img, dtype=np.float32) / 255.0
        pred_b = (pred_b > 0.5).astype(np.float32)
    tp = ((pred_b == 1) & (gt_b == 1)).sum()
    fp = ((pred_b == 1) & (gt_b == 0)).sum()
    fn = ((pred_b == 0) & (gt_b == 1)).sum()
    tn = ((pred_b == 0) & (gt_b == 0)).sum()
    union = tp + fp + fn
    iou = float(tp / union) if union > 0 else 1.0
    fp_rate = float(fp / (tn + fp + smooth))
    fn_rate = float(fn / (tp + fn + smooth))
    metrics['dl_iou'] = iou
    metrics['dl_fp_rate'] = fp_rate
    metrics['dl_fn_rate'] = fn_rate
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera_id', type=str, required=True)
    parser.add_argument('--image_id', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--image_size', type=int, nargs=2, default=[160, 160])
    args = parser.parse_args()

    camera_id = args.camera_id
    image_id = args.image_id
    image_id_int = int(image_id)
    image_path = os.path.join('data', f'skyfinder_{camera_id}', 'images', f'{image_id_int:03d}.jpg')
    if not os.path.exists(image_path):
        image_path = os.path.join('data', f'skyfinder_{camera_id}', 'images', f'{image_id}.jpg')
    if not os.path.exists(image_path):
        print(f"[ERROR] 找不到圖片: camera_id={camera_id}, image_id={image_id}")
        print(f"  嘗試: {image_path}")
        sys.exit(1)

    checkpoint_path = args.checkpoint or find_latest_checkpoint()
    if not checkpoint_path or not os.path.isfile(checkpoint_path):
        print(f"[ERROR] 找不到 checkpoint。請指定 --checkpoint 或確保 outputs/train_multi_camera_*/checkpoints/best.pth 存在。")
        sys.exit(1)

    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'

    print(f"Checkpoint: {checkpoint_path}")
    print(f"圖片: {image_path}")
    print(f"載入 U-Net...")
    model = load_model(checkpoint_path, device)
    image_tensor, original_size = preprocess(image_path, tuple(args.image_size))
    pred_mask, _ = predict(model, image_tensor, device)
    gt_mask, gt_exists = load_gt_mask(image_path, camera_id, image_id, has_sky=True)
    orig_img = Image.open(image_path).convert('RGB')
    orig_h, orig_w = orig_img.size[1], orig_img.size[0]
    if pred_mask.shape != (orig_h, orig_w):
        pred_img = Image.fromarray((pred_mask * 255).astype(np.uint8))
        pred_img = pred_img.resize((orig_w, orig_h), Image.NEAREST)
        pred_mask_orig = np.array(pred_img, dtype=np.float32) / 255.0
    else:
        pred_mask_orig = pred_mask
    metrics = calculate_metrics(
        pred_mask_orig,
        gt_mask if gt_exists else None,
        has_gt=gt_exists,
    )
    print()
    print(f"camera_id={camera_id}, image_id={image_id} — U-Net（與 SAM 比較用同一模型）")
    print(f"  IoU:       {metrics['dl_iou']:.6f}" if metrics['dl_iou'] is not None else "  IoU:       (無 GT)")
    print(f"  FP rate:   {metrics['dl_fp_rate']:.6f}")
    print(f"  FN rate:   {metrics['dl_fn_rate']:.6f}" if metrics['dl_fn_rate'] is not None else "  FN rate:   (無 GT)")
    print(f"  pred_ratio: {metrics['dl_pred_positive_ratio']:.6f}")


if __name__ == '__main__':
    main()
