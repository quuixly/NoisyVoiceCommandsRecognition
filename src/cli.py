"""Single entry point for the whole project.

    nvcr prepare
    nvcr train -c configs/experiments/logmel.yaml --set train.epochs=40
    nvcr evaluate --run baseline
    nvcr report --run baseline
    nvcr compare baseline logmel crnn
    nvcr preview

Every subcommand takes the same `-c/--config` and repeatable `--set key.path=value`,
so any knob in configs/default.yaml is reachable from the command line without the
CLI having to mirror it as a flag.
"""

import argparse
import logging
import sys
from pathlib import Path

from src.config import Config
from src.data.manifest import MANIFEST_PATH
from src.logging_config import setup_logging
from src.training.trainer import REPORTS_DIR, run_dirs

logger = logging.getLogger(__name__)


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-c", "--config", default=None, help="experiment YAML, deep-merged onto configs/default.yaml")
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        metavar="KEY=VALUE", help="override any config key, e.g. --set train.lr=1e-3")
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)


def _config(args) -> Config:
    return Config.load(args.config, args.overrides)


def cmd_prepare(args) -> int:
    from src.data.prepare import prepare_dataset

    prepare_dataset(_config(args), args.manifest)
    return 0


def cmd_train(args) -> int:
    from src.data.dataset import get_dataloaders
    from src.training.seed import set_seed
    from src.training.trainer import Trainer

    config = _config(args)
    set_seed(config.seed)
    loaders = get_dataloaders(config, args.manifest, splits=("train", "val"))
    Trainer(config).fit(loaders["train"], loaders["val"])
    if not config.train.overfit_batch and not args.no_report:
        _build_report(config.train.run_name, config)
    return 0


def cmd_evaluate(args) -> int:
    from src.evaluation import evaluate_run

    evaluate_run(args.run, device_name=args.device, manifest_path=args.manifest)
    if not args.no_report:
        _build_report(args.run, errors=_error_figure(args.run, args.manifest, args.device))
    return 0


def _error_figure(run_name: str, manifest_path: Path, device_name: str | None) -> Path | None:
    """Renders the confidently-wrong test clips. A confident error points at a data
    or feature problem, which a confusion matrix alone will not show you."""
    from src.data.dataset import SpeechCommandsDataset
    from src.data.manifest import read_manifest
    from src.evaluation import load_run, worst_errors
    from src.viz.audio import plot_examples

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


def cmd_run(args) -> int:
    """prepare (if stale) -> train -> evaluate -> report, for one config.

    The whole point of a run being reproducible from its config is that this can be a
    single command. Preparation is skipped when the manifest on disk was already built
    from this config's data/split settings, so repeated runs over the same corpus pay
    the decode cost once.
    """
    from src.data.dataset import get_dataloaders
    from src.data.prepare import needs_prepare, prepare_dataset
    from src.evaluation import evaluate_run
    from src.training.seed import set_seed
    from src.training.trainer import Trainer

    config = _config(args)
    run_name = config.train.run_name

    if args.skip_prepare:
        logger.info("Skipping prepare (--skip-prepare)")
    elif needs_prepare(config, args.manifest):
        logger.info("Corpus/split config differs from the manifest on disk — preparing")
        prepare_dataset(config, args.manifest)
    else:
        logger.info(f"Reusing {args.manifest} — matches this config's data and split settings")

    logger.info(f"=== train: {run_name} ===")
    set_seed(config.seed)
    loaders = get_dataloaders(config, args.manifest, splits=("train", "val"))
    Trainer(config).fit(loaders["train"], loaders["val"])

    logger.info(f"=== evaluate: {run_name} ===")
    evaluate_run(run_name, device_name=config.train.device, manifest_path=args.manifest)

    report = _build_report(run_name, config, errors=_error_figure(run_name, args.manifest, config.train.device))
    logger.info(f"=== done: {run_name} -> {report} ===")
    return 0


def cmd_summary(args) -> int:
    from src.viz.charts import plot_run_comparison, plot_snr_comparison
    from src.viz.summary import build_summary, collect_runs

    runs = collect_runs(REPORTS_DIR, args.runs or None)
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

    out = build_summary(runs, Path(args.out), figures)
    logger.info(f"Summary of {len(runs)} runs: {out}")
    return 0


def cmd_report(args) -> int:
    _build_report(args.run)
    return 0


def cmd_compare(args) -> int:
    from src.viz.charts import plot_run_comparison

    runs = {}
    for name in args.runs:
        history = REPORTS_DIR / name / "history.csv"
        if not history.exists():
            raise FileNotFoundError(f"No history for run {name!r} at {history}")
        runs[name] = history
    out = plot_run_comparison(runs, Path(args.out))
    logger.info(f"Wrote {out}")
    return 0


def cmd_preview(args) -> int:
    """Augmentation preview straight from the config — the fast way to see whether a
    transform range is sensible before spending an epoch on it."""
    import numpy as np

    from src.data.manifest import read_manifest
    from src.data.waveforms import WaveformStore
    from src.viz.audio import plot_augmentation_preview

    config = _config(args)
    rows = read_manifest(args.manifest)
    pack = WaveformStore(config.data).load("train")
    clips = []
    seen: set[str] = set()
    for row in rows:
        if row["split"] != "train" or row["command"] in seen:
            continue
        seen.add(row["command"])
        clips.append((row["command"], np.asarray(pack[int(row["wave_idx"])]).copy()))
        if len(clips) >= args.count:
            break
    out = plot_augmentation_preview(config, clips, Path(args.out), n_variants=args.variants)
    logger.info(f"Wrote {out}")
    return 0


def _build_report(run_name: str, config: Config | None = None, errors: Path | None = None) -> Path:
    """Renders whatever artifacts exist for a run into one HTML file. Safe to call
    after training (history only) or after evaluation (adds the test sections)."""
    import yaml

    from src.viz.charts import (
        plot_confusion,
        plot_history,
        plot_per_class,
        plot_snr_sweep,
    )
    from src.viz.report import build_report, load_metrics

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

    out = build_report(report_dir, raw, metrics, figures)
    logger.info(f"Report: {out}")
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nvcr", description="Noisy voice-command recognition pipeline.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="download audio, split by speaker, pack waveforms")
    _add_common(prepare)
    prepare.set_defaults(func=cmd_prepare)

    run = subparsers.add_parser("run", help="the full chain: prepare (if stale) + train + evaluate + report")
    _add_common(run)
    run.add_argument("--skip-prepare", action="store_true",
                     help="assume the manifest is current even if the config says otherwise")
    run.set_defaults(func=cmd_run)

    summary = subparsers.add_parser("summary", help="one comparison page over several runs")
    summary.add_argument("runs", nargs="*", help="run names; default is every run under reports/")
    summary.add_argument("--out", default="reports/summary.html")
    summary.set_defaults(func=cmd_summary)

    train = subparsers.add_parser("train", help="train a model")
    _add_common(train)
    train.add_argument("--no-report", action="store_true", help="skip the HTML report afterwards")
    train.set_defaults(func=cmd_train)

    evaluate = subparsers.add_parser("evaluate", help="test-set metrics + SNR sweep for a trained run")
    _add_common(evaluate)
    evaluate.add_argument("--run", required=True, help="run name under checkpoints/")
    evaluate.add_argument("--device", default=None, choices=["auto", "cuda", "mps", "cpu"])
    evaluate.add_argument("--no-report", action="store_true")
    evaluate.set_defaults(func=cmd_evaluate)

    report = subparsers.add_parser("report", help="rebuild reports/<run>/report.html from existing artifacts")
    report.add_argument("--run", required=True)
    report.set_defaults(func=cmd_report)

    compare = subparsers.add_parser("compare", help="overlay validation curves from several runs")
    compare.add_argument("runs", nargs="+")
    compare.add_argument("--out", default="reports/comparison.png")
    compare.set_defaults(func=cmd_compare)

    preview = subparsers.add_parser("preview", help="plot clean vs augmented audio and features")
    _add_common(preview)
    preview.add_argument("--count", type=int, default=3, help="how many clips")
    preview.add_argument("--variants", type=int, default=2, help="augmented versions per clip")
    preview.add_argument("--out", default="reports/augmentation_preview.png")
    preview.set_defaults(func=cmd_preview)

    return parser


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
