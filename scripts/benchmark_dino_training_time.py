"""
粗估 SkySegModel (DINOv2 + 2-layer decoder) 在 train data 上訓練的總耗時。
以實際跑數個 batch 測量 sec/batch，再外推至完整訓練。

資料來源：in-domain split（outputs/in_domain_splits_summary.json）
對應腳本：scripts/train_in_domain.py（train_list.txt / val_list.txt）
"""
import sys
import time
import torch
import json
import os

# 專案根目錄（方便 import models）
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)
from models import create_sky_seg_dinov2_linear

def load_in_domain_counts():
    """從 in_domain_splits_summary.json 讀取 train/val 數量。"""
    path = os.path.join(_root, "outputs", "in_domain_splits_summary.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到 in-domain split 摘要：{path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    total = data.get("total_images", {})
    return total.get("train", 0), total.get("val", 0)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # In-domain split（與 train_in_domain.py 一致）
    train_samples, val_samples = load_in_domain_counts()
    batch_size = 2
    image_size = (160, 160)  # 會對齊 14 的倍數
    epochs = 10   # train_in_domain.py 使用 epochs=10
    train_batches = (train_samples + batch_size - 1) // batch_size
    val_batches = (val_samples + batch_size - 1) // batch_size

    print(f"\n資料: in-domain split (outputs/in_domain_splits_summary.json)")
    print(f"  Train: {train_samples} 筆, Val: {val_samples} 筆")
    print(f"設定: batch_size={batch_size}, image_size={image_size}, epochs={epochs}")
    print(f"  Train batches/epoch: {train_batches}, Val batches/epoch: {val_batches}")

    model = create_sky_seg_dinov2_linear().to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    # 輸入會自動被 DINOv2 對齊到 14 的倍數 (160 -> 154 或 168)，用 160 即可
    dummy_x = torch.randn(batch_size, 3, image_size[0], image_size[1], device=device)
    dummy_y = torch.rand(batch_size, 1, image_size[0], image_size[1], device=device)

    # Warmup
    for _ in range(3):
        model.train()
        optimizer.zero_grad()
        out = model(dummy_x)
        loss = criterion(out, dummy_y)
        loss.backward()
        optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize()

    # 訓練 batch 計時 (forward + backward)
    n_train = 20
    t0 = time.perf_counter()
    for _ in range(n_train):
        model.train()
        optimizer.zero_grad()
        out = model(dummy_x)
        loss = criterion(out, dummy_y)
        loss.backward()
        optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    train_sec_per_batch = (t1 - t0) / n_train

    # 驗證 batch 計時 (forward only)
    model.eval()
    n_val = 20
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_val):
            _ = model(dummy_x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    val_sec_per_batch = (t1 - t0) / n_val

    # 外推（in-domain：每 epoch 只做 train + val，無 sanity / val_night_day）
    total_train_batches = train_batches * epochs
    total_val_batches = val_batches * epochs
    train_time_sec = total_train_batches * train_sec_per_batch
    val_time_sec = total_val_batches * val_sec_per_batch
    total_sec = train_time_sec + val_time_sec
    # 加一點 checkpoint / logging 餘裕
    total_sec *= 1.05

    def fmt(t):
        if t >= 3600:
            return f"{t/3600:.1f} 小時"
        elif t >= 60:
            return f"{t/60:.1f} 分鐘"
        return f"{t:.0f} 秒"

    print("\n--- 測量結果 ---")
    print(f"  訓練 (forward+backward): {train_sec_per_batch*1000:.0f} ms/batch")
    print(f"  驗證 (forward only):     {val_sec_per_batch*1000:.0f} ms/batch")
    print(f"\n外推 (in-domain, {epochs} epochs, {train_samples} train, {val_samples} val):")
    print(f"  訓練總 batch 數: {total_train_batches}  →  約 {fmt(train_time_sec)}")
    print(f"  驗證總 batch 數: {total_val_batches}  →  約 {fmt(val_time_sec)}")
    print(f"  預估總訓練時間: 約 {fmt(total_sec)}")
    print("\n(已含約 5% 餘裕給 checkpoint / logging；實際 DataLoader 讀取若在 CPU 可能略增)")

if __name__ == "__main__":
    main()
