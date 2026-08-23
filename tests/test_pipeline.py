"""End-to-end: prepare -> train -> evaluate -> report on a synthetic corpus."""

import json

from src.data.dataset import SpeechCommandsDataset, get_dataloaders
from src.data.manifest import read_manifest
from src.data.prepare import prepare_dataset


def test_prepare_builds_manifest_and_waveform_packs(config):
    rows = prepare_dataset(config)
    assert len(rows) == 12 * 2 * len(config.data.commands)
    manifest = read_manifest()
    assert {r["split"] for r in manifest} == {"train", "val", "test"}
    assert all(r["wave_idx"] != "" for r in manifest)


def test_prepare_is_idempotent(config):
    prepare_dataset(config)
    before = read_manifest()
    prepare_dataset(config)  # second run must reuse the packs, not corrupt them
    assert read_manifest() == before


def test_dataset_augments_train_and_freezes_val(config):
    prepare_dataset(config)
    train = SpeechCommandsDataset(config, "train")
    val = SpeechCommandsDataset(config, "val")

    a, _ = train[0]
    b, _ = train[0]
    assert not a.equal(b), "training items should get a fresh random transform each access"
    assert val[0][0].equal(val[0][0]), "validation must be deterministic"
    assert a.shape == tuple(config.features.shape)


def test_train_evaluate_report_round_trip(config):
    from src.cli import _build_report
    from src.evaluation import evaluate_run
    from src.training.trainer import Trainer, run_dirs

    config.train.run_name = "smoke"
    prepare_dataset(config)
    loaders = get_dataloaders(config, splits=("train", "val"))
    Trainer(config).fit(loaders["train"], loaders["val"])

    ckpt_dir, report_dir = run_dirs("smoke")
    assert (ckpt_dir / "best.pt").exists()
    assert (report_dir / "history.csv").exists()
    assert (report_dir / "config.yaml").exists()

    result = evaluate_run("smoke")
    assert 0.0 <= result["accuracy"] <= 1.0
    assert len(result["snr_sweep"]) == 2  # clean + one SNR point
    saved = json.loads((report_dir / "test_metrics.json").read_text())
    assert saved["labels"] == config.data.commands

    report = _build_report("smoke")
    assert report.exists()
    html = report.read_text()
    assert "Confusion matrix" in html and "data:image/png;base64," in html
