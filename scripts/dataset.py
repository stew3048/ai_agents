from __future__ import annotations

import torch
from torch.utils.data import Dataset


class DummySkyDataset(Dataset):
    """產生假影像與假 mask 的 Dataset。"""

    def __init__(self, length: int = 100) -> None:
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Dataset 負責定義如何依索引取出單筆資料（影像與標註）。
        # 這裡用 index 當隨機種子，讓每個索引對應固定內容。
        generator = torch.Generator().manual_seed(index)
        image = torch.rand(3, 256, 256, generator=generator)
        # DataLoader 會批次與迭代 Dataset 的資料，供訓練流程使用。
        mask = torch.randint(0, 2, (256, 256), dtype=torch.int64, generator=generator)
        return image, mask
