"""The `main.py run` chain and the cross-run summary page."""

import json

import pytest

from main import main
from src.data.prepare import needs_prepare, prepare_dataset, stamp_path
from src.training.trainer import REPORTS_DIR, run_dirs


def test_needs_prepare_tracks_the_corpus_config(config):
    assert needs_prepare(config) is True
    prepare_dataset(config)
    assert needs_prepare(config) is False

    # A change to what the corpus contains must invalidate the manifest, or the next
    # run trains against clips the config no longer describes.
    config.data.max_per_command = 4
    assert needs_prepare(config) is True


def test_needs_prepare_tracks_the_split_config(config):
    prepare_dataset(config)
    config.split.by = "file"
    assert needs_prepare(config) is True


def test_needs_prepare_ignores_unrelated_config_changes(config):
    prepare_dataset(config)
    config.train.lr = 0.5
    config.model.name = "logreg"
    assert needs_prepare(config) is False


def test_stamp_is_written_next_to_the_manifest(config):
    prepare_dataset(config)
    assert stamp_path().read_text().strip() == config.corpus_fingerprint()


def _run(config_path, run_name, *overrides):
    argv = ["run", "-c", str(config_path), "--set", f"train.run_name={run_name}", *overrides]
    return main(argv)


@pytest.fixture
def experiment(config, tmp_path):
    """A config file on disk that the CLI can load, pointing at the synthetic corpus."""
    import yaml

    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(config.to_dict(), sort_keys=False))
    return path


def test_run_does_the_whole_chain(experiment):
    assert _run(experiment, "chain") == 0

    ckpt_dir, report_dir = run_dirs("chain")
    assert (ckpt_dir / "best.pt").exists()
    assert (report_dir / "history.csv").exists()
    assert (report_dir / "config.yaml").exists()
    assert (report_dir / "test_metrics.json").exists()
    assert (report_dir / "report.html").exists()


def test_run_reuses_a_matching_manifest(experiment):
    """Asserted on the side effect rather than the log line: `prepare` rewrites the
    manifest, so an untouched mtime is proof it was skipped."""
    from src.data.manifest import MANIFEST_PATH

    _run(experiment, "first")
    before = MANIFEST_PATH.stat().st_mtime_ns

    _run(experiment, "second")
    assert MANIFEST_PATH.stat().st_mtime_ns == before

    _run(experiment, "third", "--set", "data.max_per_command=8")
    assert MANIFEST_PATH.stat().st_mtime_ns != before


def test_summary_covers_every_run_and_flags_split_corpora(experiment, tmp_path):
    _run(experiment, "runA")
    _run(experiment, "runB", "--set", "model.name=logreg")
    # A third run over a deliberately different corpus — the page must say so.
    _run(experiment, "runC", "--set", "data.max_per_command=8")

    out = tmp_path / "summary.html"
    assert main(["summary", "--out", str(out)]) == 0
    html = out.read_text()

    for name in ("runA", "runB", "runC"):
        assert name in html
    assert "do not share one corpus" in html
    assert "logreg" in html
    assert (REPORTS_DIR / "figures" / "runs_val_acc.png").exists()


def test_summary_is_quiet_when_every_run_shares_a_corpus(experiment, tmp_path):
    _run(experiment, "runA")
    _run(experiment, "runB", "--set", "train.lr=1e-3")

    out = tmp_path / "summary.html"
    main(["summary", "runA", "runB", "--out", str(out)])
    assert "do not share one corpus" not in out.read_text()


def test_summary_includes_a_trained_but_unevaluated_run(experiment, tmp_path):
    _run(experiment, "done")
    main(["train", "-c", str(experiment), "--set", "train.run_name=halfway", "--no-report"])

    out = tmp_path / "summary.html"
    main(["summary", "--out", str(out)])
    html = out.read_text()
    assert "halfway" in html
    assert "not evaluated" in html


def test_run_metrics_land_in_the_manifest_label_order(experiment):
    _run(experiment, "labels")
    _, report_dir = run_dirs("labels")
    metrics = json.loads((report_dir / "test_metrics.json").read_text())
    assert metrics["labels"] == ["up", "down", "left", "right", "stop", "go"]


def test_run_accepts_a_name_or_any_path_inside_the_run(experiment):
    """Tab completion produces paths, not names. Every form below names one run."""
    from src.pipeline import resolve_run

    _run(experiment, "pathy")
    ckpt_dir, report_dir = run_dirs("pathy")
    for value in (
        "pathy",
        report_dir,
        f"{report_dir}/",
        report_dir / "report.html",
        ckpt_dir / "best.pt",
    ):
        assert resolve_run(value) == "pathy", value


def test_a_path_outside_a_run_directory_is_rejected(experiment, tmp_path):
    """A run is a name, not a location: `checkpoints/<name>` and `reports/<name>` are
    derived from it, so accepting a foreign directory would write somewhere else."""
    import pytest

    from src.pipeline import resolve_run

    (tmp_path / "elsewhere" / "pathy").mkdir(parents=True)
    with pytest.raises(ValueError, match="not a run directory"):
        resolve_run(tmp_path / "elsewhere" / "pathy")
    with pytest.raises(FileNotFoundError, match="No run directory"):
        resolve_run("reports/never_trained")


def test_report_and_summary_take_paths(experiment, tmp_path):
    _run(experiment, "pathy")
    _, report_dir = run_dirs("pathy")

    assert main(["report", "--run", str(report_dir)]) == 0
    out = tmp_path / "summary.html"
    assert main(["summary", str(report_dir), "--out", str(out)]) == 0
    assert "pathy" in out.read_text()
