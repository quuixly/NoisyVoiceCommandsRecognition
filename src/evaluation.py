"""Test-set evaluation, including the SNR sweep that is the thesis headline metric."""

import json
import logging
from pathlib import Path

import numpy as np
import torch

from src.config import Config
from src.data.dataset import SpeechCommandsDataset
from src.data.manifest import MANIFEST_PATH, read_manifest
from src.labels import index_to_label
from src.models import build_model, count_parameters
from src.training.device import resolve_device
from src.training.metrics import (
    accuracy,
    confusion_matrix,
    macro_f1,
    per_class_precision_recall_f1,
)
from src.training.trainer import run_dirs

logger = logging.getLogger(__name__)


def load_run(
    run_name: str, device_name: str | None = None
) -> tuple[torch.nn.Module, Config, dict, torch.device]:
    """Rebuilds a run from its checkpoint alone — the config travels inside the file,
    so evaluating an old run never depends on the current YAML."""
    ckpt_dir, _ = run_dirs(run_name)
    ckpt_path = ckpt_dir / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"No checkpoint at {ckpt_path} — train run {run_name!r} first"
        )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    config = _config_from_dict(ckpt["config"])
    device = resolve_device(device_name or config.train.device)
    model = build_model(config).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    logger.info(
        f"Loaded {ckpt_path} (epoch {ckpt['epoch']}, val_acc {ckpt['val_acc']:.4f}) on {device}"
    )
    return model, config, ckpt, device


def _config_from_dict(raw: dict) -> Config:
    from src.config import (
        _from_dict,  # local import: internal helper, not part of the public API
    )

    config = _from_dict(Config, raw)
    config.resolve()
    return config


@torch.no_grad()
def predict(
    model: torch.nn.Module, dataset: SpeechCommandsDataset, device, batch_size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (y_true, y_pred, max_softmax) — the confidence is what a deployed
    decision threshold would key off, so it's collected here rather than recomputed."""
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    preds, targets, confidences = [], [], []
    for x, y in loader:
        logits = model(x.to(device))
        probs = torch.softmax(logits, dim=1)
        confidence, predicted = probs.max(dim=1)
        preds.append(predicted.cpu().numpy())
        targets.append(y.numpy())
        confidences.append(confidence.cpu().numpy())
    return np.concatenate(targets), np.concatenate(preds), np.concatenate(confidences)


def evaluate_run(
    run_name: str,
    device_name: str | None = None,
    manifest_path: Path = MANIFEST_PATH,
    snr_db: list[float] | None = None,
) -> dict:
    model, config, ckpt, device = load_run(run_name, device_name)
    _, report_dir = run_dirs(run_name)
    report_dir.mkdir(parents=True, exist_ok=True)
    rows = read_manifest(manifest_path)
    labels = list(config.data.commands)

    clean = SpeechCommandsDataset(config, "test", rows=rows, augment=False)
    y_true, y_pred, confidence = predict(model, clean, device, config.eval.batch_size)

    cm = confusion_matrix(y_true, y_pred, len(labels))
    per_class_raw = per_class_precision_recall_f1(cm)
    support = cm.sum(axis=1)
    per_class = {
        labels[i]: {**per_class_raw[i], "support": int(support[i])}
        for i in range(len(labels))
    }

    sweep = [{"snr_db": None, "accuracy": accuracy(y_true, y_pred)}]
    for snr in config.eval.snr_db if snr_db is None else snr_db:
        noisy = SpeechCommandsDataset(
            config, "test", rows=rows, augment=False, snr_db=float(snr)
        )
        n_true, n_pred, _ = predict(model, noisy, device, config.eval.batch_size)
        acc = accuracy(n_true, n_pred)
        sweep.append({"snr_db": float(snr), "accuracy": acc})
        logger.info(f"  SNR {snr:>5g} dB: accuracy {acc:.4f}")

    result = {
        "run_name": run_name,
        "checkpoint_epoch": ckpt["epoch"],
        "parameters": count_parameters(model),
        "feature_shape": list(config.features.shape),
        "accuracy": accuracy(y_true, y_pred),
        "macro_f1": macro_f1(cm),
        "mean_confidence": float(confidence.mean()),
        "labels": labels,
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
        "snr_sweep": sweep,
    }
    (report_dir / "test_metrics.json").write_text(json.dumps(result, indent=2))

    logger.info(
        f"test accuracy {result['accuracy']:.4f}  macro F1 {result['macro_f1']:.4f}"
    )
    for name, m in per_class.items():
        logger.info(
            f"  {name:>6}: P {m['precision']:.3f}  R {m['recall']:.3f}  F1 {m['f1']:.3f}  n={m['support']}"
        )
    return result


def worst_errors(
    run_name: str,
    config: Config,
    model,
    device,
    dataset: SpeechCommandsDataset,
    limit: int = 8,
) -> list[tuple[str, np.ndarray]]:
    """The misclassified clips the model was most confident about — the ones worth
    looking at, since a confident error points at a data or feature problem rather
    than an ordinary hard case."""
    y_true, y_pred, confidence = predict(model, dataset, device, config.eval.batch_size)
    wrong = np.flatnonzero(y_true != y_pred)
    if wrong.size == 0:
        return []
    ranked = wrong[np.argsort(-confidence[wrong])][:limit]
    to_label = index_to_label(config.data.commands)
    return [
        (
            f"{to_label[int(y_true[i])]} -> {to_label[int(y_pred[i])]} ({confidence[i]:.2f})",
            dataset.waveform(int(i)),
        )
        for i in ranked
    ]
