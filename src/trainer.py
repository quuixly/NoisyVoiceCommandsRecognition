import csv
import logging
import time
from dataclasses import asdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import TrainConfig
from src.constants import CHECKPOINTS_DIR, REPORTS_DIR
from src.models import build_model

HISTORY_COLUMNS = ["epoch", "train_loss", "val_loss", "val_acc", "lr", "secs"]


class Trainer:
    """Owns one training run end to end: model/optimizer/scheduler setup,
    the epoch loop, checkpointing, and history logging. One run = one
    config.run_name = one checkpoints/<run_name>/ + reports/<run_name>/ pair,
    so multiple experiments never clobber each other."""

    def __init__(self, config: TrainConfig) -> None:
        self.config = config
        self.device = self._resolve_device(config.device)

        self.ckpt_dir = CHECKPOINTS_DIR / config.run_name
        self.report_dir = REPORTS_DIR / config.run_name
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.history_path = self.report_dir / "history.csv"

        self.model = build_model(config.model_name).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )
        self.criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
        self.scheduler = self._build_scheduler()

        self.start_epoch = 1
        self.best_val_acc = 0.0
        self.epochs_without_improvement = 0

        if config.resume:
            self._load_checkpoint(self.ckpt_dir / "last.pt")

        logging.info(
            f"Trainer ready: model={config.model_name} device={self.device} "
            f"run={config.run_name} params={sum(p.numel() for p in self.model.parameters()):,}"
        )

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device != "auto":
            return torch.device(device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _build_scheduler(self) -> torch.optim.lr_scheduler.LRScheduler:
        warmup = torch.optim.lr_scheduler.LinearLR(
            self.optimizer, start_factor=1e-2, end_factor=1.0, total_iters=self.config.warmup_epochs
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(1, self.config.epochs - self.config.warmup_epochs)
        )
        return torch.optim.lr_scheduler.SequentialLR(
            self.optimizer, schedulers=[warmup, cosine], milestones=[self.config.warmup_epochs]
        )

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> None:
        if self.config.overfit_batch:
            self._overfit_batch(train_loader)
            return

        self._write_history_header_if_needed()
        for epoch in range(self.start_epoch, self.config.epochs + 1):
            t0 = time.time()
            train_loss = self._train_epoch(train_loader)
            val_loss, val_acc = self._val_epoch(val_loader)
            self.scheduler.step()
            secs = time.time() - t0
            lr = self.optimizer.param_groups[0]["lr"]

            logging.info(
                f"epoch {epoch}/{self.config.epochs} - train_loss {train_loss:.4f} - "
                f"val_loss {val_loss:.4f} - val_acc {val_acc:.4f} - lr {lr:.2e} - {secs:.1f}s"
            )
            self._append_history(epoch, train_loss, val_loss, val_acc, lr, secs)

            # Update self.best_val_acc before saving last.pt, not after — last.pt's
            # snapshot of best_val_acc must be current, since --resume reads it back
            # as the improvement threshold. Saving last.pt first would persist a
            # stale (one-epoch-behind) best_val_acc, letting a worse post-resume
            # epoch silently overwrite a legitimately-better best.pt.
            improved = val_acc > self.best_val_acc
            if improved:
                self.best_val_acc = val_acc
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1

            self._save_checkpoint(epoch, val_acc, "last.pt")
            if improved:
                self._save_checkpoint(epoch, val_acc, "best.pt")
            elif self.epochs_without_improvement >= self.config.patience:
                logging.info(f"Early stopping at epoch {epoch} (patience {self.config.patience})")
                break

    def _train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss, n = 0.0, 0
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            logits = self.model(x)
            loss = self.criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.optimizer.step()
            total_loss += loss.item() * x.size(0)
            n += x.size(0)
        return total_loss / n

    @torch.no_grad()
    def _val_epoch(self, loader: DataLoader) -> tuple[float, float]:
        self.model.eval()
        total_loss, n_correct, n = 0.0, 0, 0
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            logits = self.model(x)
            loss = self.criterion(logits, y)
            total_loss += loss.item() * x.size(0)
            n_correct += (logits.argmax(1) == y).sum().item()
            n += x.size(0)
        return total_loss / n, n_correct / n

    def _overfit_batch(
        self, loader: DataLoader, max_steps: int = 300, target_acc: float = 1.0, lr: float = 1e-2
    ) -> None:
        """Trains on one fixed batch until it's memorized — a wiring check,
        not a model check. Uses its own aggressive, disposable optimizer and
        an unsmoothed loss — self.optimizer/self.criterion are tuned for a
        full run (label smoothing, weight decay, low lr for a 30-epoch cosine
        schedule) and would make even correctly-wired code converge too
        slowly here to tell "broken" from "just needs more steps"."""
        self.model.train()
        x, y = next(iter(loader))
        x, y = x.to(self.device), y.to(self.device)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        for step in range(1, max_steps + 1):
            optimizer.zero_grad()
            logits = self.model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            acc = (logits.argmax(1) == y).float().mean().item()
            if step % 10 == 0 or acc >= target_acc:
                logging.info(f"overfit step {step}: loss {loss.item():.4f} acc {acc:.4f}")
            if acc >= target_acc:
                logging.info(f"Overfit-batch reached {acc:.0%} train accuracy at step {step}")
                return
        logging.warning(f"Overfit-batch did not reach {target_acc:.0%} within {max_steps} steps — check wiring")

    def _write_history_header_if_needed(self) -> None:
        if not self.history_path.exists():
            with self.history_path.open("w", newline="") as f:
                csv.writer(f).writerow(HISTORY_COLUMNS)

    def _append_history(
        self, epoch: int, train_loss: float, val_loss: float, val_acc: float, lr: float, secs: float
    ) -> None:
        with self.history_path.open("a", newline="") as f:
            csv.writer(f).writerow([epoch, train_loss, val_loss, val_acc, lr, secs])

    def _save_checkpoint(self, epoch: int, val_acc: float, filename: str) -> None:
        torch.save(
            {
                "epoch": epoch,
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "scheduler_state": self.scheduler.state_dict(),
                "val_acc": val_acc,
                "best_val_acc": self.best_val_acc,
                "config": asdict(self.config),
            },
            self.ckpt_dir / filename,
        )

    def _load_checkpoint(self, path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"No checkpoint to resume from: {path}")
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.scheduler.load_state_dict(ckpt["scheduler_state"])
        self.start_epoch = ckpt["epoch"] + 1
        self.best_val_acc = ckpt.get("best_val_acc", 0.0)
        logging.info(f"Resumed from {path} at epoch {self.start_epoch}")
