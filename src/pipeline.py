"""Pipeline stages, one function per thing the project can do.

`main.py` parses arguments and calls into here; nothing in this module knows about
argparse, so every stage is callable from a test or a notebook with a `Config` object
and no argv. `run_all` is the whole chain — prepare (if stale), train, evaluate,
report — and `sweep` is that chain once per experiment file.

Imports of torch, matplotlib and the data stack stay inside the functions on purpose:
`main.py report` and `main.py --help` should not pay for them.
"""

import logging
import subprocess
import sys
import time
from pathlib import Path

from src.config import Config
from src.data.manifest import MANIFEST_PATH
from src.training.trainer import CHECKPOINTS_DIR, REPORTS_DIR, run_dirs

logger = logging.getLogger(__name__)

EXPERIMENTS_DIR = Path("configs/experiments")
SWEEP_LOG_DIR = REPORTS_DIR / "sweep_logs"
SUMMARY_PATH = REPORTS_DIR / "summary.html"

# Left out of an unfiltered sweep, still runnable by name: smoke exists to be fast
# rather than to be compared against anything. Anything here that changes `data` or
# `split` would also re-prepare the manifest mid-sweep and land on the summary page as
# a non-comparable run, which is the other reason to skip a config by default.
SWEEP_SKIP = {"smoke"}


def resolve_run(value: str | Path) -> str:
    """A run name from whatever the shell produced. `report --run fullcheck` still
    works, and so does anything tab-completion hands you: `reports/fullcheck`,
    `reports/fullcheck/`, `reports/fullcheck/report.html`, `checkpoints/fullcheck/best.pt`.

    A run is a name, not a location — `checkpoints/<name>/` and `reports/<name>/` are
    derived from it — so a path that points somewhere else is an error rather than a
    silent write into the wrong directory.
    """
    text = str(value).rstrip("/\\")
    if not text:
        raise ValueError("Empty run name")

    path = Path(text)
    if path.parent == Path("."):
        return text  # a bare name: never touch the filesystem to interpret it

    if path.is_file():
        path = path.parent
    if not path.is_dir():
        raise FileNotFoundError(f"No run directory at {text}")

    parent = path.resolve().parent
    if parent not in {CHECKPOINTS_DIR.resolve(), REPORTS_DIR.resolve()}:
        raise ValueError(
            f"{text!r} is not a run directory: a run lives in {CHECKPOINTS_DIR}/<name> "
            f"and {REPORTS_DIR}/<name>, not under {parent}"
        )
    return path.resolve().name


def prepare(config: Config, manifest_path: Path = MANIFEST_PATH) -> None:
    from src.data.prepare import prepare_dataset

    prepare_dataset(config, manifest_path)


def train(config: Config, manifest_path: Path = MANIFEST_PATH, report: bool = True) -> None:
    from src.data.dataset import get_dataloaders
    from src.training.seed import set_seed
    from src.training.trainer import Trainer

    set_seed(config.seed)
    loaders = get_dataloaders(config, manifest_path, splits=("train", "val"))
    Trainer(config).fit(loaders["train"], loaders["val"])
    if report and not config.train.overfit_batch:
        build_report(config.train.run_name, config)


def evaluate(
    run_name: str,
    manifest_path: Path = MANIFEST_PATH,
    device_name: str | None = None,
    report: bool = True,
) -> dict:
    from src.evaluation import evaluate_run

    run_name = resolve_run(run_name)
    metrics = evaluate_run(run_name, device_name=device_name, manifest_path=manifest_path)
    if report:
        build_report(run_name, errors=error_figure(run_name, manifest_path, device_name))
    return metrics


def error_figure(run_name: str, manifest_path: Path, device_name: str | None) -> Path | None:
    """Renders the confidently-wrong test clips. A confident error points at a data
    or feature problem, which a confusion matrix alone will not show you."""
    from src.data.dataset import SpeechCommandsDataset
    from src.data.manifest import read_manifest
    from src.evaluation import load_run, worst_errors
    from src.visualization.audio import plot_examples

    run_name = resolve_run(run_name)
    model, config, _, device = load_run(run_name, device_name)
    dataset = SpeechCommandsDataset(config, "test", rows=read_manifest(manifest_path), augment=False)
    examples = worst_errors(run_name, config, model, device, dataset)
    if not examples:
        logger.info("No misclassified test clips — skipping the error figure")
        return None
    _, report_dir = run_dirs(run_name)
    return plot_examples(
        config, examples, report_dir / "figures" / "errors.png",
        title="Most confident misclassifications (actual -> predicted)",
    )


def run_all(config: Config, manifest_path: Path = MANIFEST_PATH, skip_prepare: bool = False) -> Path:
    """prepare (if stale) -> train -> evaluate -> report, for one config.

    The whole point of a run being reproducible from its config is that this can be a
    single command. Preparation is skipped when the manifest on disk was already built
    from this config's data/split settings, so repeated runs over the same corpus pay
    the decode cost once.
    """
    from src.data.prepare import needs_prepare

    run_name = config.train.run_name

    if skip_prepare:
        logger.info("Skipping prepare (--skip-prepare)")
    elif needs_prepare(config, manifest_path):
        logger.info("Corpus/split config differs from the manifest on disk — preparing")
        prepare(config, manifest_path)
    else:
        logger.info(f"Reusing {manifest_path} — matches this config's data and split settings")

    logger.info(f"=== train: {run_name} ===")
    train(config, manifest_path, report=False)

    logger.info(f"=== evaluate: {run_name} ===")
    evaluate(run_name, manifest_path, device_name=config.train.device, report=False)

    report = build_report(
        run_name, config, errors=error_figure(run_name, manifest_path, config.train.device)
    )
    logger.info(f"=== done: {run_name} -> {report} ===")
    return report


def summary(run_names: list[str] | None = None, out_path: Path = SUMMARY_PATH) -> Path:
    from src.visualization.charts import plot_run_comparison, plot_snr_comparison
    from src.visualization.summary import build_summary, collect_runs

    runs = collect_runs(REPORTS_DIR, [resolve_run(name) for name in run_names] if run_names else None)
    if not runs:
        raise FileNotFoundError(
            f"No runs with both config.yaml and history.csv under {REPORTS_DIR} — train something first"
        )

    figures_dir = REPORTS_DIR / "figures"
    figures = {"curves": plot_run_comparison(
        {name: data["history"] for name, data in runs.items()}, figures_dir / "runs_val_acc.png"
    )}

    sweeps = {
        name: [(p["snr_db"], p["accuracy"]) for p in data["metrics"]["snr_sweep"]]
        for name, data in runs.items()
        if data["metrics"] and data["metrics"].get("snr_sweep")
    }
    if sweeps:
        figures["snr"] = plot_snr_comparison(sweeps, figures_dir / "runs_snr.png")

    out = build_summary(runs, Path(out_path), figures)
    logger.info(f"Summary of {len(runs)} runs: {out}")
    return out


def compare(run_names: list[str], out_path: Path) -> Path:
    from src.visualization.charts import plot_run_comparison

    runs = {}
    for name in map(resolve_run, run_names):
        history = REPORTS_DIR / name / "history.csv"
        if not history.exists():
            raise FileNotFoundError(f"No history for run {name!r} at {history}")
        runs[name] = history
    out = plot_run_comparison(runs, Path(out_path))
    logger.info(f"Wrote {out}")
    return out


def preview(
    config: Config,
    manifest_path: Path,
    out_path: Path,
    count: int = 3,
    variants: int = 2,
) -> Path:
    """Augmentation preview straight from the config — the fast way to see whether a
    transform range is sensible before spending an epoch on it."""
    import numpy as np

    from src.data.manifest import read_manifest
    from src.data.waveforms import WaveformStore
    from src.visualization.audio import plot_augmentation_preview

    rows = read_manifest(manifest_path)
    pack = WaveformStore(config.data).load("train")
    clips = []
    seen: set[str] = set()
    for row in rows:
        if row["split"] != "train" or row["command"] in seen:
            continue
        seen.add(row["command"])
        clips.append((row["command"], np.asarray(pack[int(row["wave_idx"])]).copy()))
        if len(clips) >= count:
            break
    out = plot_augmentation_preview(config, clips, Path(out_path), n_variants=variants)
    logger.info(f"Wrote {out}")
    return out


def build_report(run_name: str, config: Config | None = None, errors: Path | None = None) -> Path:
    """Renders whatever artifacts exist for a run into one HTML file. Safe to call
    after training (history only) or after evaluation (adds the test sections)."""
    import yaml

    from src.visualization.charts import (
        plot_confusion,
        plot_history,
        plot_per_class,
        plot_snr_sweep,
    )
    from src.visualization.report import build_report as render_report
    from src.visualization.report import load_metrics

    run_name = resolve_run(run_name)
    _, report_dir = run_dirs(run_name)
    config_path = report_dir / "config.yaml"
    if config is not None:
        raw = config.to_dict()
    elif config_path.exists():
        raw = yaml.safe_load(config_path.read_text())
    else:
        raise FileNotFoundError(f"No config for run {run_name!r} at {config_path}")

    figures_dir = report_dir / "figures"
    figures: dict[str, Path] = {}
    for name, path in (("errors", errors), ("augmentation", report_dir / "figures" / "augmentation.png")):
        if path is not None and Path(path).exists():
            figures[name] = Path(path)

    history = report_dir / "history.csv"
    if history.exists():
        figures["history"] = plot_history(history, figures_dir / "history.png")

    metrics = load_metrics(report_dir)
    if metrics:
        import numpy as np

        figures["confusion"] = plot_confusion(
            np.array(metrics["confusion_matrix"]), metrics["labels"], figures_dir / "confusion.png"
        )
        figures["per_class"] = plot_per_class(metrics["per_class"], figures_dir / "per_class.png")
        if len(metrics.get("snr_sweep", [])) > 1:
            points = [(p["snr_db"], p["accuracy"]) for p in metrics["snr_sweep"]]
            figures["snr"] = plot_snr_sweep(points, figures_dir / "snr.png")

    out = render_report(report_dir, raw, metrics, figures)
    logger.info(f"Report: {out}")
    return out


def sweep_names(names: list[str], experiments_dir: Path = EXPERIMENTS_DIR) -> list[str]:
    """Which experiments a sweep covers. Explicit names win — including the skipped
    ones, which are skipped by default rather than forbidden. Otherwise: every config
    in `experiments_dir` minus SWEEP_SKIP, and `baseline` (the defaults, no experiment
    file) only when no baseline.yaml exists, so the control is in every sweep exactly
    once."""
    if names:
        return list(names)
    found = sorted(p.stem for p in Path(experiments_dir).glob("*.yaml") if p.stem not in SWEEP_SKIP)
    if not found:
        raise FileNotFoundError(f"No experiments found in {experiments_dir}")
    return found if "baseline" in found else ["baseline", *found]


def sweep(
    names: list[str],
    entry: Path,
    overrides: list[str] | None = None,
    experiments_dir: Path = EXPERIMENTS_DIR,
    log_dir: Path = SWEEP_LOG_DIR,
) -> int:
    """Train, evaluate and report every experiment, then one summary page over all of
    them. Each experiment becomes a run named after its config file, so reports/<name>/
    and checkpoints/<name>/ never collide.

    One child process per experiment: a crash or an OOM in the second config must not
    throw away the rest of a six-hour sweep, and a fresh process also gives each run
    clean CUDA/MPS state. A failing experiment is recorded and the sweep continues.
    """
    experiments_dir, log_dir = Path(experiments_dir), Path(log_dir)
    selected = sweep_names(names, experiments_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Sweep: {' '.join(selected)}")
    logger.info(f"Logs:  {log_dir}")

    failed: list[str] = []
    started_all = time.monotonic()

    for name in selected:
        argv = [sys.executable, str(entry), "run"]
        if name != "baseline":
            config_path = experiments_dir / f"{name}.yaml"
            if not config_path.is_file():
                logger.error(f"!! {name}: no such config ({config_path}), skipping")
                failed.append(name)
                continue
            argv += ["-c", str(config_path)]
        argv += ["--set", f"train.run_name={name}"]
        for override in overrides or []:
            argv += ["--set", override]

        logger.info(f"── {name} ──────────────────────────────────────────────")
        started = time.monotonic()
        log_path = log_dir / f"{name}.log"
        code = _tee(argv, log_path)
        elapsed = time.monotonic() - started
        if code == 0:
            logger.info(f"   ok in {elapsed:.0f}s")
        else:
            logger.error(f"!! {name} failed after {elapsed:.0f}s (exit {code}) — see {log_path}")
            failed.append(name)

    logger.info("── summary ────────────────────────────────────────────")
    try:
        summary()
    except FileNotFoundError as exc:
        logger.error(f"No summary page: {exc}")

    logger.info(f"Sweep finished in {time.monotonic() - started_all:.0f}s")
    if failed:
        logger.error(f"Failed: {' '.join(failed)}")
        return 1
    return 0


def _tee(argv: list[str], log_path: Path) -> int:
    """Runs a child, writing its output to `log_path` and to our stdout as it arrives.
    A plain redirect would leave a multi-hour sweep looking hung."""
    with log_path.open("w") as log, subprocess.Popen(
        argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    ) as child:
        assert child.stdout is not None
        for line in child.stdout:
            sys.stdout.write(line)
            log.write(line)
        return child.wait()
