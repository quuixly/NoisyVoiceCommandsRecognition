"""Cross-run comparison page.

One run's report answers "how did this do". This answers "which of these should I
write up" — the same metrics for every run in one table, plus the two charts that
actually separate candidates: validation curves and noise robustness.
"""

import json
from pathlib import Path

import yaml

from src.config import corpus_fingerprint_of
from src.visualization.report import CSS, _img, _table

EXTRA_CSS = """
.rank { color:var(--ink-3); font-variant-numeric:tabular-nums; }
td.best { font-weight:700; color:var(--ink); }
.note { color:var(--ink-3); font-size:.85rem; margin:.5rem 0 0; }
.warn { background:#fdf1ec; border:1px solid #eb6834; border-left-width:4px; border-radius:8px;
        padding:.85rem 1rem; margin:1.25rem 0; color:var(--ink); font-size:.92rem; }
.warn strong { color:#b2431c; }
.warn code { font-size:.85em; }
"""


def collect_runs(reports_dir: Path, names: list[str] | None = None) -> dict[str, dict]:
    """Gathers what each run left behind. A run that has trained but not been
    evaluated still appears — with its curve but no test numbers — rather than being
    silently dropped, so a half-finished sweep is visible as such."""
    reports_dir = Path(reports_dir)
    candidates = sorted(p.name for p in reports_dir.iterdir() if p.is_dir()) if names is None else names

    runs: dict[str, dict] = {}
    for name in candidates:
        run_dir = reports_dir / name
        config_path, history_path = run_dir / "config.yaml", run_dir / "history.csv"
        if not config_path.exists() or not history_path.exists():
            continue
        metrics_path = run_dir / "test_metrics.json"
        config = yaml.safe_load(config_path.read_text())
        runs[name] = {
            "config": config,
            "history": history_path,
            "metrics": json.loads(metrics_path.read_text()) if metrics_path.exists() else None,
            "corpus": corpus_fingerprint_of(config),
        }
    return runs


def _rows(runs: dict[str, dict]) -> tuple[list[str], list[list], list[int]]:
    headers = ["run", "corpus", "model", "features", "params", "epochs", "best val acc",
               "test acc", "macro F1", "acc @ worst SNR"]
    rows, accuracies = [], []
    for name, data in sorted(runs.items()):
        config, metrics = data["config"], data["metrics"]
        epochs = sum(1 for _ in data["history"].read_text().splitlines()) - 1
        sweep = (metrics or {}).get("snr_sweep") or []
        noisy = [p for p in sweep if p["snr_db"] is not None]
        worst = min(noisy, key=lambda p: p["snr_db"]) if noisy else None
        rows.append([
            name,
            data["corpus"],
            config["model"]["name"],
            f"{config['features']['type']} {tuple(metrics['feature_shape'])}" if metrics else config["features"]["type"],
            f"{metrics['parameters']:,}" if metrics else "—",
            epochs,
            f"{_best_val(data['history']):.4f}",
            f"{metrics['accuracy']:.4f}" if metrics else "not evaluated",
            f"{metrics['macro_f1']:.4f}" if metrics else "—",
            f"{worst['accuracy']:.4f} @ {worst['snr_db']:g} dB" if worst else "—",
        ])
        accuracies.append(metrics["accuracy"] if metrics else -1.0)

    best = [i for i, a in enumerate(accuracies) if a == max(accuracies) and a >= 0]
    return headers, rows, best


def _best_val(history_path: Path) -> float:
    import csv

    with Path(history_path).open(newline="") as f:
        values = [float(row["val_acc"]) for row in csv.DictReader(f)]
    return max(values) if values else 0.0


def _corpus_warning(runs: dict[str, dict]) -> str:
    """Runs trained on different clips or different splits are not comparable, and
    nothing about a chart makes that visible — two curves look equally authoritative
    whether or not they measure the same thing. Say so at the top of the page."""
    groups: dict[str, list[str]] = {}
    for name, data in sorted(runs.items()):
        groups.setdefault(data["corpus"], []).append(name)
    if len(groups) < 2:
        return ""
    lines = "".join(
        f"<li><code>{fingerprint}</code> — {', '.join(names)}</li>" for fingerprint, names in sorted(groups.items())
    )
    return (
        f'<div class="warn"><strong>These runs do not share one corpus.</strong> '
        f"{len(groups)} different data/split configurations are on this page, so their "
        f"accuracies are not directly comparable and the charts below put them on one axis anyway:"
        f"<ul>{lines}</ul>"
        "Pass explicit run names to <code>main.py summary</code> to compare within one group.</div>"
    )


def build_summary(runs: dict[str, dict], out_path: Path, figures: dict[str, Path]) -> Path:
    headers, rows, best = _rows(runs)
    marked = [
        [f'<span class="best">{cell}</span>' if i in best and j == 7 else cell for j, cell in enumerate(row)]
        for i, row in enumerate(rows)
    ]

    parts = [_corpus_warning(runs), _table(headers, marked)]
    evaluated = sum(1 for d in runs.values() if d["metrics"])
    if evaluated < len(runs):
        parts.append(
            f'<p class="note">{len(runs) - evaluated} of {len(runs)} runs have not been '
            "evaluated yet — their test columns are blank. Run <code>main.py evaluate --run &lt;name&gt;</code>.</p>"
        )
    if "curves" in figures:
        parts.append("<h2>Validation curves</h2>" + _img(figures["curves"]))
    if "snr" in figures:
        parts.append(
            "<h2>Noise robustness</h2>" + _img(figures["snr"])
            + '<p class="note">Diamonds are clean test accuracy; circles are accuracy with '
            "white noise mixed in at the marked SNR. A flatter curve is the more robust model, "
            "regardless of where it starts.</p>"
        )

    html = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>Run comparison</title><style>{CSS}{EXTRA_CSS}</style></head><body><div class=\"wrap\">"
        f'<h1>Run comparison</h1><p class="sub">{len(runs)} runs · '
        f"{evaluated} evaluated · best test accuracy highlighted</p>"
        + "".join(parts)
        + "</div></body></html>"
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html)
    return out_path
