"""
訓練 DINOv2CLIPSegModel（DINOv2 + CLIPSeg Heatmap Prior + CNN Decoder）

流程
----
1. Phase 1 – 預計算 CLIPSeg heatmap cache
   對每張 train + val 影像跑一次 CLIPSeg sky logit 推論，
   將 sigmoid heatmap 存成 float16 .pt 檔，後續訓練直接讀取，不需每次重算。
   Cache 目錄：data/clipseg_heatmap_cache/{skyfinder_xxx}/{yyy}.pt

2. Phase 2 – 訓練
   - 模型：models/dinov2_clipseg_decoder.py（DINOv2 backbone 凍結，decoder 1537-ch 可訓練）
   - 初始化方式（預設）：從舊版 1536-ch checkpoint 遷移，新 channel 零初始化
   - Loss：BCEWithLogitsLoss + Dice
   - 參數：batch_size=2, epochs=10, lr=1e-4, image_size=(160,160)

輸出
----
- outputs/train_dino_clipseg_YYYYMMDD_HHMMSS/checkpoints/best.pth
- outputs/train_dino_clipseg_YYYYMMDD_HHMMSS/training_log.csv

呼叫方式
--------
# 從現有 best.pth 遷移（推薦）
python scripts/train_dino_clipseg.py \
    --pretrained outputs/train_dino_segmentation_20260214_101339/checkpoints/best.pth

# 全新訓練
python scripts/train_dino_clipseg.py --from_scratch

# 跳過 heatmap 預計算（cache 已存在時）
python scripts/train_dino_clipseg.py --skip_cache \
    --pretrained outputs/.../best.pth
"""

import os
import sys
import csv
import argparse
from datetime import datetime
from pathlib import Path

_script_dir = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_script_dir)
sys.path.insert(0, _root)

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms.functional as TF
from tqdm import tqdm

from models.dinov2_clipseg_decoder import DINOv2CLIPSegModel, create_dinov2_clipseg_model
from scripts.train_in_domain import load_list_file, CombinedLoss, save_checkpoint
from utils.metrics import calculate_metrics


# ─── CLIPSeg heatmap 生成 ────────────────────────────────────────────────────

def _get_clipseg_raw_heatmap(processor, clipseg_model, image_path: str, device) -> np.ndarray:
    """
    對單張影像跑 CLIPSeg，回傳 sky sigmoid heatmap（HxW float32, 0~1）。
    不套用空間先驗或 building 對比，保留最純粹的語意機率用於 feature fusion。
    """
    image = Image.open(image_path).convert('RGB')
    inputs = processor(
        text=['sky'],
        images=image,
        return_tensors='pt',
        padding=True,
    )
    inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}
    with torch.no_grad():
        out = clipseg_model(**inputs)
    logits = out.logits.squeeze().cpu().float().numpy()
    if logits.ndim != 2:
        logits = logits[0]
    heatmap = 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))
    return heatmap.astype(np.float32)


def cache_clipseg_heatmaps(
    all_pairs,          # list of {'camera_id': str, 'image': str, ...}
    base_data_dir: str,
    cache_dir: str,
    device,
    model_id: str = "CIDAS/clipseg-rd64-refined",
):
    """
    對 all_pairs 中所有影像預計算 CLIPSeg heatmap 並存至 cache_dir。
    已存在的 cache 會自動略過（斷點續算）。
    """
    from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation  # type: ignore

    print(f"[Phase 1] 載入 CLIPSeg：{model_id}")
    processor = CLIPSegProcessor.from_pretrained(model_id)
    clipseg = CLIPSegForImageSegmentation.from_pretrained(model_id)
    clipseg.to(device).eval()

    os.makedirs(cache_dir, exist_ok=True)

    todo = []
    for p in all_pairs:
        cam_folder = f"skyfinder_{p['camera_id']}"
        img_path = os.path.join(base_data_dir, cam_folder, "images", p['image'])
        frame = os.path.splitext(p['image'])[0]
        cache_path = os.path.join(cache_dir, cam_folder, f"{frame}.pt")
        if not os.path.exists(cache_path):
            todo.append((img_path, cache_path))

    print(f"[Phase 1] 需計算 {len(todo)} 張（已 cache {len(all_pairs)-len(todo)} 張）")
    if not todo:
        print("[Phase 1] 全部已 cache，跳過。")
        clipseg.cpu()
        del clipseg
        return

    for img_path, cache_path in tqdm(todo, desc="  CLIPSeg heatmap"):
        try:
            heatmap = _get_clipseg_raw_heatmap(processor, clipseg, img_path, device)
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            # float16 可節省約一半空間，精度損失可忽略
            torch.save(torch.tensor(heatmap, dtype=torch.float16), cache_path)
        except Exception as e:
            print(f"  [警告] {img_path}: {e}")

    clipseg.cpu()
    del clipseg
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    print("[Phase 1] Heatmap cache 完成。")


# ─── Dataset ─────────────────────────────────────────────────────────────────

class SkySegWithHeatmapDataset(Dataset):
    """
    讀取 (image, clipseg_heatmap, gt_mask) 三元組。
    Heatmap 從 cache_dir 讀取；若找不到，以全零填補（不中斷訓練）。
    """

    def __init__(
        self,
        split_list,
        base_data_dir: str,
        cache_dir: str,
        image_size=(160, 160),
        augment: bool = True,
    ):
        self.items = []
        self.cache_dir = cache_dir
        self.image_size = image_size
        self.augment = augment

        for p in split_list:
            cam_folder = f"skyfinder_{p['camera_id']}"
            img_path  = os.path.join(base_data_dir, cam_folder, "images", p['image'])
            mask_path = os.path.join(base_data_dir, cam_folder, "masks",  p['mask'])
            frame = os.path.splitext(p['image'])[0]
            heat_path = os.path.join(cache_dir, cam_folder, f"{frame}.pt")
            if os.path.exists(img_path) and os.path.exists(mask_path):
                self.items.append((img_path, mask_path, heat_path))

        if len(self.items) == 0:
            raise ValueError("Dataset 找不到有效影像。")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path, mask_path, heat_path = self.items[idx]

        # ── 讀取影像 ─────────────────────────────────────────
        image = Image.open(img_path).convert('RGB')
        image = image.resize(self.image_size, Image.BILINEAR)
        image_t = TF.to_tensor(image)  # (3, H, W), [0,1]

        # ── 讀取 GT mask ──────────────────────────────────────
        mask = Image.open(mask_path).convert('L')
        mask = mask.resize(self.image_size, Image.NEAREST)
        mask_arr = np.array(mask, dtype=np.float32)
        if mask_arr.max() > 1.0:
            mask_arr = mask_arr / 255.0
        mask_t = torch.from_numpy(mask_arr).unsqueeze(0)  # (1, H, W)
        mask_t = (mask_t > 0.5).float()

        # ── 讀取 CLIPSeg heatmap ──────────────────────────────
        if os.path.exists(heat_path):
            heat = torch.load(heat_path, map_location='cpu').float()  # (H', W')
            heat = heat.unsqueeze(0).unsqueeze(0)                     # (1, 1, H', W')
            heat = F.interpolate(heat, size=self.image_size, mode='bilinear', align_corners=False)
            heat = heat.squeeze(0)                                     # (1, H, W)
        else:
            heat = torch.zeros(1, *self.image_size)

        return image_t, heat, mask_t


# ─── 訓練 / 評估 loop ─────────────────────────────────────────────────────────

def train_epoch_clipseg(model, loader, criterion, optimizer, device, use_amp=False):
    model.train()
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    total_loss, total_iou, n = 0.0, 0.0, 0

    for images, heatmaps, masks in tqdm(loader, desc='  Train', leave=False):
        images   = images.to(device)
        heatmaps = heatmaps.to(device)
        masks    = masks.to(device)

        optimizer.zero_grad()
        if use_amp:
            with torch.amp.autocast('cuda'):
                outputs = model(images, heatmaps)
                loss = criterion(outputs, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images, heatmaps)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        with torch.no_grad():
            total_iou += calculate_metrics(outputs, masks)['iou']
        n += 1

        del images, heatmaps, masks, outputs, loss
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    return total_loss / n if n else 0.0, total_iou / n if n else 0.0


def evaluate_clipseg(model, loader, criterion, device, use_amp=False):
    model.eval()
    total_loss = 0.0
    sums = dict(iou=0, dice=0, pixel_acc=0, fp_rate=0, fn_rate=0)
    n = 0

    with torch.no_grad():
        for images, heatmaps, masks in tqdm(loader, desc='  Val', leave=False):
            images   = images.to(device)
            heatmaps = heatmaps.to(device)
            masks    = masks.to(device)

            if use_amp:
                with torch.amp.autocast('cuda'):
                    outputs = model(images, heatmaps)
                    loss = criterion(outputs, masks)
            else:
                outputs = model(images, heatmaps)
                loss = criterion(outputs, masks)

            total_loss += loss.item()
            m = calculate_metrics(outputs, masks)
            for k in sums:
                sums[k] += m.get(k, 0)
            n += 1

    avg = {k: v / n for k, v in sums.items()} if n else sums
    return total_loss / n if n else 0.0, avg


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='訓練 DINOv2 + CLIPSeg Heatmap Prior 融合模型')
    parser.add_argument('--device',      type=str, default=None,
                        help='cpu / cuda，預設自動偵測')
    parser.add_argument('--pretrained',  type=str, default=None,
                        help='從舊版 1536-ch best.pth 遷移（decoder 第一層零 padding）')
    parser.add_argument('--from_scratch', action='store_true',
                        help='不載入任何 checkpoint，完全重頭訓練')
    parser.add_argument('--skip_cache',  action='store_true',
                        help='跳過 CLIPSeg heatmap 預計算（cache 已齊全時使用）')
    parser.add_argument('--epochs',      type=int, default=10)
    parser.add_argument('--batch_size',  type=int, default=2)
    parser.add_argument('--lr',          type=float, default=1e-4)
    parser.add_argument('--train_list',  type=str,
                        default=os.path.join(_root, 'data', 'train_list.txt'))
    parser.add_argument('--val_list',    type=str,
                        default=os.path.join(_root, 'data', 'val_list.txt'))
    args = parser.parse_args()

    # ── 設備 ─────────────────────────────────────────────────
    if args.device == 'cpu':
        device = torch.device('cpu')
        use_amp = False
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        use_amp = device.type == 'cuda'

    print("=" * 60)
    print("  DINOv2 + CLIPSeg Heatmap Prior 融合模型訓練")
    print("=" * 60)
    print(f"設備: {device}  混合精度: {use_amp}")
    print(f"batch_size={args.batch_size}, epochs={args.epochs}, lr={args.lr}")
    print()

    # ── 參數 ─────────────────────────────────────────────────
    image_size   = (160, 160)
    num_workers  = 0
    seed         = 42
    base_data_dir = os.path.join(_root, 'data', 'data')   # skyfinder_xxx/ 在 data/data/ 下
    cache_dir     = os.path.join(_root, 'data', 'clipseg_heatmap_cache')

    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)

    # ── 載入清單 ─────────────────────────────────────────────
    print(f"載入 train list：{args.train_list}")
    train_list = load_list_file(args.train_list, base_data_dir=base_data_dir)
    print(f"  共 {len(train_list)} 筆（含警告略過項）")
    print(f"載入 val list：{args.val_list}")
    val_list = load_list_file(args.val_list, base_data_dir=base_data_dir)
    print(f"  共 {len(val_list)} 筆\n")

    # ── Phase 1：CLIPSeg heatmap 預計算 ──────────────────────
    if not args.skip_cache:
        cache_clipseg_heatmaps(
            all_pairs=train_list + val_list,
            base_data_dir=base_data_dir,
            cache_dir=cache_dir,
            device=device,
        )
    else:
        print("[Phase 1] 已跳過 heatmap 預計算（--skip_cache）")

    # ── Phase 2：Dataset & DataLoader ────────────────────────
    print("\n[Phase 2] 建立 Dataset...")
    train_ds = SkySegWithHeatmapDataset(
        train_list, base_data_dir, cache_dir, image_size=image_size, augment=True)
    val_ds   = SkySegWithHeatmapDataset(
        val_list,   base_data_dir, cache_dir, image_size=image_size, augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=num_workers,
                              pin_memory=(device.type == 'cuda'))
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=num_workers,
                              pin_memory=(device.type == 'cuda'))
    print(f"  Train batches: {len(train_loader)}, Val batches: {len(val_loader)}\n")

    # ── Phase 3：建立模型 ─────────────────────────────────────
    print("[Phase 3] 建立模型...")
    if args.from_scratch:
        model = create_dinov2_clipseg_model().to(device)
        print("  模式：全新訓練（1537-ch decoder）")
    elif args.pretrained and os.path.exists(args.pretrained):
        model = DINOv2CLIPSegModel.load_from_dino_checkpoint(
            args.pretrained).to(device)
        print(f"  模式：從 {args.pretrained} 遷移（新 channel 零初始化）")
    else:
        print("  ⚠️  未指定 --pretrained 且未使用 --from_scratch")
        print("  → 自動使用全新初始化（1537-ch decoder）")
        model = create_dinov2_clipseg_model().to(device)

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  總參數: {total:,}  可訓練: {trainable:,}\n")

    # ── 訓練設定 ─────────────────────────────────────────────
    criterion = CombinedLoss(bce_weight=1.0, dice_weight=1.0)
    optimizer = optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr
    )

    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir  = os.path.join(_root, 'outputs', f'train_dino_clipseg_{timestamp}')
    ckpt_dir    = os.path.join(output_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"輸出目錄：{output_dir}\n")

    # ── 訓練迴圈 ─────────────────────────────────────────────
    training_log  = []
    best_val_iou  = 0.0

    print("開始訓練...")
    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        train_loss, train_iou = train_epoch_clipseg(
            model, train_loader, criterion, optimizer, device, use_amp)
        val_loss, val_m = evaluate_clipseg(
            model, val_loader, criterion, device, use_amp)
        val_iou = val_m['iou']

        entry = dict(
            epoch=epoch,
            train_loss=train_loss, train_iou=train_iou,
            val_loss=val_loss,     val_iou=val_iou,
            val_dice=val_m['dice'],val_pixel_acc=val_m['pixel_acc'],
            val_fp_rate=val_m['fp_rate'], val_fn_rate=val_m['fn_rate'],
        )
        training_log.append(entry)

        print(f"  Train  Loss={train_loss:.4f}  IoU={train_iou:.4f}")
        print(f"  Val    Loss={val_loss:.4f}  IoU={val_iou:.4f}  "
              f"Dice={val_m['dice']:.4f}  FP={val_m['fp_rate']:.4f}  FN={val_m['fn_rate']:.4f}")

        is_best = val_iou > best_val_iou
        if is_best:
            best_val_iou = val_iou
        save_checkpoint(model, optimizer, epoch, val_iou, ckpt_dir, is_best=is_best)
        print()

    # ── 儲存 log ─────────────────────────────────────────────
    log_file = os.path.join(output_dir, 'training_log.csv')
    with open(log_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=training_log[0].keys())
        writer.writeheader()
        writer.writerows(training_log)

    print("=" * 60)
    print("  訓練完成")
    print("=" * 60)
    print(f"Best Val IoU : {best_val_iou:.4f}")
    print(f"Best checkpoint : {os.path.join(ckpt_dir, 'best.pth')}")
    print(f"Training log    : {log_file}")


if __name__ == "__main__":
    main()
