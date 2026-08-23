import csv
import logging
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.config import Config
from src.models import build_model, count_parameters
from src.training.device import resolve_device
from src.training.metrics import accuracy, confusion_matrix, macro_f1

logger = logging.getLogger(__name__)

CHECKPOINTS_DIR = Path("checkpoints")
REPORTS_DIR = Path("reports")
HISTORY_COLUMNS = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "val_macro_f1", "lr", "secs"]


def run_dirs(run_name: str) -> tuple[Path, Path]:
    return CHECKPOINTS_DIR / run_name, REPORTS_DIR / run_name


class Trainer:
    """One training run end to end. A run is identified by `train.run_name`, which
    owns `checkpoints/<run>/` and `reports/<run>/`, so parallel experiments never
    overwrite each other. The full resolved config is written into both the run
    directory and every checkpoint, so a run can be reproduced or evaluated from its
    artifacts alone."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.device = resolve_device(config.train.device)
        self.ckpt_dir, self.report_dir = run_dirs(config.train.run_name)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.history_path = self.report_dir / "history.csv"
        config.save(self.report_dir / "config.yaml")

        self.model = build_model(config).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=config.train.lr, weight_decay=config.train.weight_decay
        )
        self.criterion = nn.CrossEntropyLoss(label_smoothing=config.train.label_smoothing)
        self.scheduler = self._build_scheduler()
        self.use_amp = config.train.amp and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler(enabled=self.use_amp)

        self.start_epoch = 1
        self.best_val_acc = 0.0
        self.epochs_without_improvement = 0
        if config.train.resume:
            self._load_checkpoint(self.ckpt_dir / "last.pt")

        logger.info(
            f"Trainer ready: model={config.model.name} features={config.features.type}"
            f"{config.features.shape} device={self.device} run={config.train.run_name} "
            f"params={count_parameters(self.model):,}"
        )

    def _build_scheduler(self) -> torch.optim.lr_scheduler.LRScheduler:
        train = self.config.train
        warmup = torch.optim.lr_scheduler.LinearLR(
            self.optimizer, start_factor=1e-2, end_factor=1.0, total_iters=max(1, train.warmup_epochs)
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(1, train.epochs - train.warmup_epochs)
        )
        return torch.optim.lr_scheduler.SequentialLR(
            self.optimizer, schedulers=[warmup, cosine], milestones=[max(1, train.warmup_epochs)]
        )

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> Path:
        if self.config.train.overfit_batch:
            self._overfit_batch(train_loader)
            return self.ckpt_dir

        self._write_history_header()
        for epoch in range(self.start_epoch, self.config.train.epochs + 1):
            t0 = time.time()
            train_loss, train_acc = self._train_epoch(train_loader)
            val_loss, val_acc, val_f1 = self._val_epoch(val_loader)
            self.scheduler.step()
            secs, lr = time.time() - t0, self.optimizer.param_groups[0]["lr"]

            logger.info(
                f"epoch {epoch}/{self.config.train.epochs} - loss {train_loss:.4f}/{val_loss:.4f} - "
                f"acc {train_acc:.4f}/{val_acc:.4f} - macro_f1 {val_f1:.4f} - lr {lr:.2e} - {secs:.1f}s"
            )
            self._append_history([epoch, train_loss, train_acc, val_loss, val_acc, val_f1, lr, secs])

            # best_val_acc is updated BEFORE last.pt is written: --resume reads it back
            # as the improvement threshold, so a stale value would let a worse
            # post-resume epoch silently overwrite a legitimately better best.pt.
            improved = val_acc > self.best_val_acc
            if improved:
                self.best_val_acc = val_acc
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1

            self._save_checkpoint(epoch, val_acc, "last.pt")
            if improved:
                self._save_checkpoint(epoch, val_acc, "best.pt")
            elif self.epochs_without_improvement >= self.config.train.patience:
                logger.info(f"Early stop at epoch {epoch} (patience {self.config.train.patience})")
                break

        logger.info(f"Best val accuracy {self.best_val_acc:.4f} — {self.ckpt_dir / 'best.pt'}")
        return self.ckpt_dir

    def _train_epoch(self, loader: DataLoader) -> tuple[float, float]:
        self.model.train()
        total_loss, n_correct, n = 0.0, 0, 0
        for x, y in loader:
            x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                logits = self.model(x)
                loss = self.criterion(logits, y)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.train.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            total_loss += loss.item() * x.size(0)
            n_correct += (logits.argmax(1) == y).sum().item()
            n += x.size(0)
        return total_loss / n, n_correct / n

    @torch.no_grad()
    def _val_epoch(self, loader: DataLoader) -> tuple[float, float, float]:
        self.model.eval()
        total_loss, n = 0.0, 0
        preds, targets = [], []
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            logits = self.model(x)
            total_loss += self.criterion(logits, y).item() * x.size(0)
            n += x.size(0)
            preds.append(logits.argmax(1).cpu().numpy())
            targets.append(y.cpu().numpy())
        y_pred, y_true = np.concatenate(preds), np.concatenate(targets)
        cm = confusion_matrix(y_true, y_pred, len(self.config.data.commands))
        return total_loss / n, accuracy(y_true, y_pred), macro_f1(cm)

    def _overfit_batch(self, loader: DataLoader, max_steps: int = 300, target_acc: float = 1.0, lr: float = 1e-2) -> None:
        """Memorize one fixed batch — a wiring check, not a model check. Uses its own
        aggressive throwaway optimizer and an unsmoothed loss: the real optimizer is
        tuned for a 30-epoch cosine schedule and would converge too slowly here to
        distinguish "broken" from "needs more steps"."""
        self.model.train()
        x, y = next(iter(loader))
        x, y = x.to(self.device), y.to(self.device)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        for step in range(1, max_steps + 1):
            optimizer.zero_grad(set_to_none=True)
            logits = self.model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            acc = (logits.argmax(1) == y).float().mean().item()
            if step % 25 == 0:
                logger.info(f"overfit step {step}: loss {loss.item():.4f} acc {acc:.4f}")
            if acc >= target_acc:
                logger.info(f"Overfit-batch reached {acc:.0%} at step {step} — wiring OK")
                return
        logger.warning(f"Overfit-batch did not reach {target_acc:.0%} in {max_steps} steps — check wiring")

    def _write_history_header(self) -> None:
        if not self.history_path.exists():
            with self.history_path.open("w", newline="") as f:
                csv.writer(f).writerow(HISTORY_COLUMNS)

    def _append_history(self, values: list) -> None:
        with self.history_path.open("a", newline="") as f:
            csv.writer(f).writerow(values)

    def _save_checkpoint(self, epoch: int, val_acc: float, filename: str) -> None:
        torch.save(
            {
                "epoch": epoch,
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "scheduler_state": self.scheduler.state_dict(),
                "val_acc": val_acc,
                "best_val_acc": self.best_val_acc,
                "config": self.config.to_dict(),
            },
            self.ckpt_dir / filename,
        )

    def _load_checkpoint(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"No checkpoint to resume from: {path}")
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.scheduler.load_state_dict(ckpt["scheduler_state"])
        self.start_epoch = ckpt["epoch"] + 1
        self.best_val_acc = ckpt.get("best_val_acc", 0.0)
        logger.info(f"Resumed {path} at epoch {self.start_epoch} (best val_acc {self.best_val_acc:.4f})")
