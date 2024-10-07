import torch
from torch.utils.data import Dataset, Dataloader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from dataclasses import dataclass, asdict
from typing import Optional, Any, Dict
from collections import OrderedDict

@dataclass
class DataConfig:
    path: str = None
    block_size: int = None
    train_split: float = None
    truncate: float = 1.0


class CharDataset(Dataset):

    def __init__(self, cfg: DataConfig):
        with open('input.txt', 'r', encoding='utf-8') as f:
            self.data = f.read()

        chars = sorted(list(set(self.data)))
        self.vocab_size = len(chars)
        
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.block_size = cfg.block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        chunk = self.data[idx:idx + self.block_size + 1]
        dix = [self.stoi[s] for s in chunk]
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y

@dataclass
class TrainerConfig:
    max_epochs: int = None
    batch_size: int = None
    data_loader_workers: int = None
    grad_norm_clip: float = None
    snapshot_path: Optional[str] = None
    save_every: int = None
    use_amp: bool = None