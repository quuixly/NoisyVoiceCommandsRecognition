#!/usr/bin/env python3
"""Single entry point for the whole project.

    python main.py prepare
    python main.py run -c configs/experiments/logmel.yaml --set train.epochs=40
    python main.py train --no-report
    python main.py evaluate --run baseline
    python main.py report --run reports/baseline    # a name or a path to it, either way
    python main.py sweep logmel crnn --set train.epochs=40
    python main.py summary
    python main.py compare baseline logmel crnn
    python main.py preview

Every subcommand that builds a config takes the same `-c/--config` and repeatable
`--set key.path=value`, so any knob in configs/default.yaml is reachable from the
command line without this file having to mirror it as a flag. The stages themselves
live in src/pipeline.py; this file only parses arguments and dispatches.
"""

import argparse
import sys
from pathlib import Path

from src import pipeline
from src.config import Config
from src.data.manifest import MANIFEST_PATH
from src.logging_config import setup_logging

ENTRY = Path(__file__).resolve()


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-c", "--config", default=None, help="experiment YAML, deep-merged onto configs/default.yaml")
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        metavar="KEY=VALUE", help="override any config key, e.g. --set train.lr=1e-3")
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)


def _config(args) -> Config:
    return Config.load(args.config, args.overrides)


def cmd_prepare(args) -> int:
    pipeline.prepare(_config(args), args.manifest)
    return 0


def cmd_train(args) -> int:
    pipeline.train(_config(args), args.manifest, report=not args.no_report)
    return 0


def cmd_evaluate(args) -> int:
    pipeline.evaluate(args.run, args.manifest, device_name=args.device, report=not args.no_report)
    return 0


def cmd_run(args) -> int:
    pipeline.run_all(_config(args), args.manifest, skip_prepare=args.skip_prepare)
    return 0


def cmd_sweep(args) -> int:
    return pipeline.sweep(
        args.experiments, ENTRY, args.overrides,
        experiments_dir=args.experiments_dir, log_dir=args.log_dir,
    )


def cmd_summary(args) -> int:
    pipeline.summary(args.runs, Path(args.out))
    return 0


def cmd_report(args) -> int:
    pipeline.build_report(args.run)
    return 0


def cmd_compare(args) -> int:
    pipeline.compare(args.runs, Path(args.out))
    return 0


def cmd_preview(args) -> int:
    pipeline.preview(_config(args), args.manifest, Path(args.out), count=args.count, variants=args.variants)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="main.py", description="Noisy voice-command recognition pipeline.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="download audio, split by speaker, pack waveforms")
    _add_common(prepare)
    prepare.set_defaults(func=cmd_prepare)

    run = subparsers.add_parser("run", help="the full chain: prepare (if stale) + train + evaluate + report")
    _add_common(run)
    run.add_argument("--skip-prepare", action="store_true",
                     help="assume the manifest is current even if the config says otherwise")
    run.set_defaults(func=cmd_run)

    sweep = subparsers.add_parser("sweep", help="run every experiment in its own process, then the summary page")
    sweep.add_argument("experiments", nargs="*",
                       help="experiment names (config stems); default is baseline + every config except smoke")
    sweep.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                       help="override forwarded to every experiment, e.g. --set train.epochs=40")
    sweep.add_argument("--experiments-dir", type=Path, default=pipeline.EXPERIMENTS_DIR)
    sweep.add_argument("--log-dir", type=Path, default=pipeline.SWEEP_LOG_DIR)
    sweep.set_defaults(func=cmd_sweep)

    summary = subparsers.add_parser("summary", help="one comparison page over several runs")
    summary.add_argument("runs", nargs="*", metavar="RUN",
                         help="run names or paths; default is every run under reports/")
    summary.add_argument("--out", default=pipeline.SUMMARY_PATH)
    summary.set_defaults(func=cmd_summary)

    train = subparsers.add_parser("train", help="train a model")
    _add_common(train)
    train.add_argument("--no-report", action="store_true", help="skip the HTML report afterwards")
    train.set_defaults(func=cmd_train)

    evaluate = subparsers.add_parser("evaluate", help="test-set metrics + SNR sweep for a trained run")
    _add_common(evaluate)
    evaluate.add_argument("--run", required=True, metavar="RUN",
                          help="run name, or any path inside it (reports/<name>, checkpoints/<name>/best.pt)")
    evaluate.add_argument("--device", default=None, choices=["auto", "cuda", "mps", "cpu"])
    evaluate.add_argument("--no-report", action="store_true")
    evaluate.set_defaults(func=cmd_evaluate)

    report = subparsers.add_parser("report", help="rebuild reports/<run>/report.html from existing artifacts")
    report.add_argument("--run", required=True, metavar="RUN",
                        help="run name, or any path inside it (reports/<name>, reports/<name>/report.html)")
    report.set_defaults(func=cmd_report)

    compare = subparsers.add_parser("compare", help="overlay validation curves from several runs")
    compare.add_argument("runs", nargs="+", metavar="RUN", help="run names or paths")
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
