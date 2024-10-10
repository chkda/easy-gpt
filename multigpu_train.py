import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group

from gpt import  GPTConfig, GPT

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

        self.config = cfg
        self.local_rank = int(os.environ['LOCAL_RANK'])
        self.global_rank = int(os.environ['RANK'])

        self.train_dataset = train_dataset

        self.train_loader = self._prepare_dataloader(train_dataset)
        self.test_loader = self._prepare_dataloader(test_dataset) if test_dataset else None

        self.epochs_run = 0
        self.model = model.to(self.local_rank)
        self.optimizer = optimizer
        self.save_every = self.config.save_every
        if self.config.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        
        if self.config.snapshot_path is None:
            self.config.snapshot_path = "snapshot.pt"
        self._load_snapshot()

        self.model = DDP(self.model, device_ids=[self.local_rank])

        
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

    def _run_batch(self, source, targets, train:bool = True) -> float:
        with torch.set_grad_enabled(train), torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=(self.config.use_amp)):
            _, loss = self.model(source, targets)

        if train:
            self.optimizer.zero_grad(set_to_norm=True)
            if self.config.use_amp:
                self.scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm(self.model.parameters(), self.config.grad_norm_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm(self.model.parameters(), self.config.grad_norm_clip)
                self.optimizer.step()
        
        return loss.item()

    def _run_epoch(self, epochs: int, dataloader: DataLoader, train: bool = True):
        dataloader.sampler.set_epoch(epoch)
        for i, (source, targets) in enumerate(dataloader):
            step_type = "Train" if train else "Eval"
            source = source.to(self.local_rank)
            targets = targets.to(self.local_rank)
            batch_loss = self._run_batch(source, targets, train)
            if i % 100 == 0:
                print(f"[GPU{self.global_rank}] Epoch {epoch} | Iter {iter} | {step_type} Loss {batch_loss:.5f}") 

    def train(self):
        for epoch in range(self.epochs_run, self.config.max_epochs):
            epoch += 1
            self._run_epoch(epoch, self.train_loader, train=True)
            if self.local_rank == 0 and epoch % self.save_every == 0:
                self._save_snapshot(epoch)
            
            if self.test_loader:
                self._run_epoch(epoch, self.test_loader, train=False)


def ddp_setup():
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def main():

    batch_size = 64
    block_size = 256
    max_iters = 5000
    eval_interval = 500
    learning_rate = 3e-4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    eval_iters = 200
    n_embd = 384
    n_head = 6
    n_layer = 6
    dropout = 0.2
    beta1 = 0.9
    beta2 = 0.95

    torch.manual_seed(1337)

    data_cfg = DataConfig(
        path="input.txt",
        block_size=block_size,
        train_split=0.9,
        )

    dataset = CharDataset(data_cfg)
    train_len = int(len(dataset) * data_cfg.train_split)
    train_set, test_set = random_split(dataset, [train_len, len(dataset) - train_len])

    gpt_cfg = GPTConfig(
        block_size=block_size,
        vocab_size=dataset.vocab_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        dropout=dropout,
    )

    model = GPT(gpt_cfg)
    optimizer = model.configure_optimizers(0, learning_rate, (beta1, beta2), device)
    train_cfg = TrainerConfig(
        max_epochs=10,
        batch_size=256,
        data_loader_workers=4,
        grad_norm_clip=1.0,
        snapshot_path="gpt_snapshot.pt",
        save_every=3,
        use_amp=True,
    )
    trainer = Trainer(train_cfg, model, optimizer, train_set, test_set)
    trainer.train()
    destroy_process_group()





if __name__ == "__main__":
    main()
