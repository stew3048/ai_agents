"""生成 threshold 對照 overlay"""
import os
import sys
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

from models import create_unet_model
from utils.dataset import get_dataloader, load_splits_from_json
from utils.image_utils import mean_luma_linear
from train_multi_camera import create_overlay
from PIL import Image

def evaluate_with_threshold_for_overlay(model, dataloader, device, threshold, use_amp=False, T=0.20):
    """用指定 threshold 評估，返回記錄"""
    model.eval()
    records = []
    
    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc=f'  T={threshold:.2f}', leave=False):
            images = images.to(device)
            masks = masks.to(device)
            if use_amp:
                with torch.amp.autocast('cuda'):
                    outputs = model(images)
            else:
                outputs = model(images)
            pred_probs = torch.sigmoid(outputs)
            
            for i in range(images.shape[0]):
                luma = mean_luma_linear(images[i])
                records.append({
                    'luma': luma,
                    'image': images[i].cpu(),
                    'gt_mask': masks[i].cpu(),
                    'pred_probs': pred_probs[i].cpu(),
                })
            del images, masks, outputs, pred_probs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    return records

def main():
    checkpoint = 'outputs/train_multi_camera_20260126_141742/checkpoints/best.pth'
    best_night_t = 0.05  # 從分析結果得知
    
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    use_amp = use_cuda
    T = 0.20
    
    print("載入資料...")
    test_list = load_splits_from_json('outputs/multi_camera_splits.json', 'test')
    test_loader = get_dataloader(
        split_list=test_list,
        batch_size=4,
        shuffle=False,
        transform=False,
        image_size=(160, 160),
        num_workers=0
    )
    
    print("載入模型...")
    model = create_unet_model(n_channels=3, n_classes=1)
    ck = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ck['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print("生成 t=0.5 記錄...")
    records_t05 = evaluate_with_threshold_for_overlay(model, test_loader, device, 0.5, use_amp, T)
    
    print("生成最佳 night threshold 記錄...")
    records_best = evaluate_with_threshold_for_overlay(model, test_loader, device, best_night_t, use_amp, T)
    
    # 找出 night 最差的 5 張（用 t=0.5 的結果）
    night_records_t05 = [r for r in records_t05 if r['luma'] < T]
    
    # 計算每張的 FN rate（用 t=0.5）
    for r in night_records_t05:
        pred_t05 = (r['pred_probs'] > 0.5).float()
        if pred_t05.dim() == 2:
            pred_t05 = pred_t05.unsqueeze(0)
        gt = r['gt_mask']
        if gt.dim() == 2:
            gt = gt.unsqueeze(0)
        tp = ((pred_t05 == 1) & (gt > 0.5)).sum().item()
        fn = ((pred_t05 == 0) & (gt > 0.5)).sum().item()
        r['fn_rate'] = fn / (tp + fn + 1e-6)
    
    night_records_t05.sort(key=lambda x: x['fn_rate'], reverse=True)
    worst_5 = night_records_t05[:5]
    
    output_dir = Path('outputs/threshold_debug')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"輸出 5 張 night 最差樣本對照 overlay 到 {output_dir}")
    for idx, record_t05 in enumerate(worst_5):
        # 找到對應的 best_night_t 記錄（用 luma 匹配）
        record_best = next((r for r in records_best if abs(r['luma'] - record_t05['luma']) < 1e-6), None)
        if record_best is None:
            continue
        
        # t=0.5 overlay
        pred_t05 = (record_t05['pred_probs'] > 0.5).float()
        if pred_t05.dim() == 2:
            pred_t05 = pred_t05.unsqueeze(0)
        overlay_t05 = create_overlay(record_t05['image'], record_t05['gt_mask'], pred_t05)
        
        # best_t overlay
        pred_best = (record_best['pred_probs'] > best_night_t).float()
        if pred_best.dim() == 2:
            pred_best = pred_best.unsqueeze(0)
        overlay_best = create_overlay(record_best['image'], record_best['gt_mask'], pred_best)
        
        # 保存
        luma_str = f"{record_t05['luma']:.4f}".replace('.', '_')
        # create_overlay 返回的是 PIL Image
        overlay_t05.save(
            output_dir / f"worst_{idx+1}_t0.5_luma{luma_str}.png"
        )
        overlay_best.save(
            output_dir / f"worst_{idx+1}_t{best_night_t:.2f}_luma{luma_str}.png"
        )
    
    print(f"已保存到 {output_dir}")
    print("完成！")

if __name__ == '__main__':
    main()
