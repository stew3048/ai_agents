"""
Dataset Loader 使用時機範例

說明：
1. Dataset Loader 在訓練前初始化（只執行一次）
2. 在訓練過程中會持續被調用來獲取批次資料
"""

import torch
from utils.dataset import get_dataloader, load_splits_from_json

# ============================================
# 時機 1: 訓練前初始化（只執行一次）
# ============================================
print("=" * 60)
print("步驟 1: 初始化 Dataset Loader（訓練前，只執行一次）")
print("=" * 60)

# 方式 A: 使用 split script 產生的 JSON 清單（推薦）
train_list = load_splits_from_json('outputs/multi_camera_splits.json', 'train')
val_list = load_splits_from_json('outputs/multi_camera_splits.json', 'val')

train_loader = get_dataloader(
    split_list=train_list,      # 使用 JSON 清單（已過濾受損檔案）
    batch_size=4,
    shuffle=True,
    transform=True,              # 訓練時開啟資料增強
    image_size=(256, 256)
)

val_loader = get_dataloader(
    split_list=val_list,
    batch_size=4,
    shuffle=False,
    transform=False,            # 驗證時關閉資料增強
    image_size=(256, 256)
)

print(f"訓練樣本數: {len(train_loader.dataset)}")
print(f"驗證樣本數: {len(val_loader.dataset)}")
print(f"訓練批次數: {len(train_loader)}")
print()

# ============================================
# 時機 2: 訓練過程中持續調用
# ============================================
print("=" * 60)
print("步驟 2: 訓練過程中持續調用（每個 epoch、每個 batch）")
print("=" * 60)

# 模擬一個簡單的訓練迴圈
model = torch.nn.Sequential(
    torch.nn.Conv2d(3, 1, 1)  # 簡單模型，僅供示範
)

optimizer = torch.optim.Adam(model.parameters())
criterion = torch.nn.BCEWithLogitsLoss()

num_epochs = 2  # 示範用，只跑 2 個 epoch

for epoch in range(1, num_epochs + 1):
    print(f"\n--- Epoch {epoch}/{num_epochs} ---")
    
    # ============================================
    # 訓練階段：遍歷 train_loader（持續調用）
    # ============================================
    model.train()
    train_loss = 0.0
    num_batches = 0
    
    print("  訓練階段：")
    for batch_idx, (images, masks) in enumerate(train_loader):
        # 注意：這裡的 for 迴圈會持續調用 train_loader
        # 每次迭代，train_loader 會：
        # 1. 從資料集中取出一批資料（batch）
        # 2. 應用資料增強（如果 transform=True）
        # 3. 轉換為張量
        # 4. 返回 (images, masks)
        
        # 前向傳播
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        # 反向傳播
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        num_batches += 1
        
        # 只顯示前 3 個 batch 作為示範
        if batch_idx < 3:
            print(f"    Batch {batch_idx + 1}: Loss = {loss.item():.4f}")
        elif batch_idx == 3:
            print(f"    ... (還有 {len(train_loader) - 4} 個 batch)")
    
    avg_train_loss = train_loss / num_batches
    print(f"  平均訓練損失: {avg_train_loss:.4f}")
    
    # ============================================
    # 驗證階段：遍歷 val_loader（持續調用）
    # ============================================
    model.eval()
    val_loss = 0.0
    num_batches = 0
    
    print("  驗證階段：")
    with torch.no_grad():  # 驗證時不需要計算梯度
        for batch_idx, (images, masks) in enumerate(val_loader):
            # 同樣地，這裡也會持續調用 val_loader
            # 每次迭代獲取一個 batch 的驗證資料
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            val_loss += loss.item()
            num_batches += 1
            
            if batch_idx < 2:
                print(f"    Batch {batch_idx + 1}: Loss = {loss.item():.4f}")
            elif batch_idx == 2:
                print(f"    ... (還有 {len(val_loader) - 3} 個 batch)")
    
    avg_val_loss = val_loss / num_batches
    print(f"  平均驗證損失: {avg_val_loss:.4f}")

print("\n" + "=" * 60)
print("總結：")
print("=" * 60)
print("1. Dataset Loader 在訓練前初始化（只執行一次）")
print("2. 在訓練迴圈中，每個 epoch 都會遍歷整個 DataLoader")
print("3. 每個 batch 迭代時，DataLoader 會自動載入下一批資料")
print("4. DataLoader 會處理：")
print("   - 批次化（batching）")
print("   - 資料增強（如果開啟）")
print("   - 多進程載入（如果 num_workers > 0）")
print("   - 隨機打亂（如果 shuffle=True）")
