import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

from src.constants import CHECKPOINTS_DIR, MANIFEST_PATH, REPORTS_DIR
from src.labels import INDEX_TO_LABEL, LABEL_TO_INDEX
from src.logging_config import setup_logging
from src.metrics import accuracy, confusion_matrix, per_class_precision_recall_f1
from src.models import build_model
from src.torch_dataset import get_dataloaders
from src.trainer import Trainer

setup_logging()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint on the test split.")
    parser.add_argument("--run-name", required=True, help="loads checkpoints/<run-name>/best.pt")
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ckpt_path = CHECKPOINTS_DIR / args.run_name / "best.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = ckpt["config"]

    device = Trainer._resolve_device(args.device)
    model = build_model(config["model_name"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    logging.info(f"Loaded {ckpt_path} (epoch {ckpt['epoch']}, val_acc {ckpt['val_acc']:.4f}) on {device}")

    _train_loader, _val_loader, test_loader = get_dataloaders(
        manifest_path=args.manifest, batch_size=args.batch_size, num_workers=args.num_workers
    )

    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            logits = model(x.to(device))
            all_preds.append(logits.argmax(1).cpu().numpy())
            all_labels.append(y.numpy())
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)

    num_classes = len(LABEL_TO_INDEX)
    acc = accuracy(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, num_classes)
    per_class = per_class_precision_recall_f1(cm)

    logging.info(f"test accuracy: {acc:.4f}")
    for idx, m in per_class.items():
        logging.info(
            f"  {INDEX_TO_LABEL[idx]}: precision {m['precision']:.3f} "
            f"recall {m['recall']:.3f} f1 {m['f1']:.3f}"
        )

    report_dir = REPORTS_DIR / args.run_name
    report_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "run_name": args.run_name,
        "checkpoint_epoch": ckpt["epoch"],
        "accuracy": acc,
        "confusion_matrix": cm.tolist(),
        "labels": [INDEX_TO_LABEL[i] for i in range(num_classes)],
        "per_class": {INDEX_TO_LABEL[idx]: m for idx, m in per_class.items()},
    }
    out_path = report_dir / "test_metrics.json"
    out_path.write_text(json.dumps(result, indent=2))
    logging.info(f"Saved {out_path}")


if __name__ == "__main__":
    main()
