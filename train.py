import argparse
import logging
from dataclasses import asdict, fields
from pathlib import Path

from src.config import TrainConfig
from src.constants import MANIFEST_PATH
from src.logging_config import setup_logging
from src.seed import set_seed
from src.torch_dataset import get_dataloaders
from src.trainer import Trainer

setup_logging()


def parse_args() -> tuple[TrainConfig, Path]:
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(description="Train a voice-command classifier.")
    parser.add_argument("--model-name", default=defaults.model_name)
    parser.add_argument("--run-name", default=defaults.run_name)
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--lr", type=float, default=defaults.lr)
    parser.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
    parser.add_argument("--label-smoothing", type=float, default=defaults.label_smoothing)
    parser.add_argument("--warmup-epochs", type=int, default=defaults.warmup_epochs)
    parser.add_argument("--grad-clip", type=float, default=defaults.grad_clip)
    parser.add_argument("--patience", type=int, default=defaults.patience)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--device", default=defaults.device, choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--amp", action="store_true", default=defaults.amp)
    parser.add_argument("--overfit-batch", action="store_true", default=defaults.overfit_batch)
    parser.add_argument(
        "--resume", action="store_true", default=defaults.resume,
        help="Resume checkpoints/<run-name>/last.pt — --run-name must name an existing run.",
    )
    parser.add_argument("--num-workers", type=int, default=defaults.num_workers)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    args = parser.parse_args()

    config = TrainConfig(**{f.name: getattr(args, f.name) for f in fields(TrainConfig)})
    return config, args.manifest


def main() -> None:
    config, manifest_path = parse_args()
    set_seed(config.seed)
    logging.info(f"config: {asdict(config)}")

    train_loader, val_loader, _test_loader = get_dataloaders(
        manifest_path=manifest_path,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )

    trainer = Trainer(config)
    trainer.fit(train_loader, val_loader)


if __name__ == "__main__":
    main()
