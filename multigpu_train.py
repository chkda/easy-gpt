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

@dataclass
class Snapshot:
    model_state: 'OrderedDict[str, torch.Tensor]'
    optimizer_state: Dict[str, Any]
    finished_epoch: int


class Trainer:

    def __init__(self, cfg: TrainerConfig, model, optimizer, train_dataset, test_dataset=None):

        self.cfg = cfg
        self.local_rank = int(os.environ['LOCAL_RANK'])
        self.global_rank = int(os.environ['RANK'])

        self.train_dataset = train_dataset
        
    def _prepare_dataloader(self, dataset: Dataset):
        return Dataloader(
            dataset,
            batch_size=self.cfg.batch_size,
            pin_memory=True,
            shuffle=False,
            num_workers=self.cfg.data_loader_workers,
            sampler=DistributedSampler(dataset),
        )

    def _load_snapshot(self):
        try:
            snapshot_data = torch.load(self.cfg.snapshot_path, map_location="cpu")
        except FileNotFoundError:
            print("Snapshot was not found. Training model from scratch")
            return
        
        snapshot = Snapshot(**snapshot_data)
        self.model.load_state_dict(snapshot.model_state)
        self.optimizer.load_state_dict(snapshot.optimizer_state)
        self.epochs_run = snapshot.finished_epoch
        print("Resuming training from snapshot at epoch:", self.epochs_run)

    def _save_snapshot(self, epoch):
        model = self.model
        raw_model = model.module if hasattr(model, "module") else model
        snapshot = Snapshot(
            model_state=raw_model.state_dict(),
            optimizer_state=self.optimizer.state_dict(),
            finished_epoch=epoch,
        )

        snapshot = asdict(snapshot)
        torch.save(snapshot, self.config.snapshot_path)
        print("Snapshot saved at epoch:", epoch)