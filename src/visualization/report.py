"""Self-contained HTML report for one run.

Images are inlined as data URIs so `reports/<run>/report.html` is a single portable
file — openable from disk, attachable to a thesis draft, with nothing to break when
it moves. Every figure is paired with a table view of the same numbers: three of the
categorical slots sit below 3:1 against the surface, and the table is what makes them
legible for readers who can't separate them.
"""

import base64
import json
from pathlib import Path

import yaml

CSS = """
:root { color-scheme: light; --surface:#fcfcfb; --card:#ffffff; --line:#e6e5e1;
        --ink:#0b0b0b; --ink-2:#52514e; --ink-3:#8a8983; --accent:#2a78d6; }
* { box-sizing: border-box; }
body { margin:0; padding:2.5rem 1.5rem 4rem; background:var(--surface); color:var(--ink);
       font:15px/1.6 -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; }
.wrap { max-width: 1080px; margin: 0 auto; }
h1 { font-size:1.75rem; margin:0 0 .25rem; letter-spacing:-.01em; }
h2 { font-size:1.1rem; margin:2.5rem 0 .75rem; letter-spacing:-.005em; }
.sub { color:var(--ink-3); margin:0 0 2rem; font-variant-numeric: tabular-nums; }
.tiles { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:.75rem; }
.tile { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:1rem 1.1rem; }
.tile .k { color:var(--ink-3); font-size:.75rem; text-transform:uppercase; letter-spacing:.06em; }
.tile .v { font-size:1.6rem; font-weight:650; margin-top:.35rem; font-variant-numeric:tabular-nums; }
figure { margin:0; background:var(--card); border:1px solid var(--line); border-radius:10px;
         padding:1rem; overflow-x:auto; }
figure img { display:block; width:100%; height:auto; }
table { border-collapse:collapse; width:100%; font-size:.9rem; font-variant-numeric:tabular-nums; }
th, td { text-align:right; padding:.4rem .6rem; border-bottom:1px solid var(--line); }
th:first-child, td:first-child { text-align:left; }
th { color:var(--ink-2); font-weight:600; font-size:.78rem; text-transform:uppercase; letter-spacing:.05em; }
details { margin-top:.75rem; }
summary { cursor:pointer; color:var(--accent); font-size:.9rem; }
.scroll { overflow-x:auto; margin-top:.75rem; }
pre { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:1rem;
      overflow-x:auto; font-size:.82rem; line-height:1.5; margin:0; }
"""


def _img(path: Path) -> str:
    encoded = base64.b64encode(Path(path).read_bytes()).decode()
    return f'<figure><img alt="" src="data:image/png;base64,{encoded}"></figure>'


def _table(headers: list[str], rows: list[list]) -> str:
    head = "".join(f"<th>{h}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows)
    return f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def _tiles(items: list[tuple[str, str]]) -> str:
    cells = "".join(f'<div class="tile"><div class="k">{k}</div><div class="v">{v}</div></div>' for k, v in items)
    return f'<div class="tiles">{cells}</div>'


def build_report(run_dir: Path, config: dict, metrics: dict | None, figures: dict[str, Path]) -> Path:
    """Assembles whatever exists: a report can be built mid-training from history
    alone, and gains the test sections once `main.py evaluate` has run."""
    run_dir = Path(run_dir)
    parts: list[str] = []
    run_name = config.get("train", {}).get("run_name", run_dir.name)

    tiles: list[tuple[str, str]] = []
    if metrics:
        tiles += [
            ("test accuracy", f"{metrics['accuracy']:.3f}"),
            ("macro F1", f"{metrics['macro_f1']:.3f}"),
            ("checkpoint epoch", str(metrics["checkpoint_epoch"])),
            ("parameters", f"{metrics['parameters']:,}"),
        ]
    tiles += [
        ("model", config["model"]["name"]),
        ("features", f"{config['features']['type']} {tuple(metrics['feature_shape']) if metrics else ''}".strip()),
        ("split by", config["split"]["by"]),
        ("augment", "on" if config["augment"]["enabled"] else "off"),
    ]
    parts.append(_tiles(tiles))

    if "history" in figures:
        parts.append("<h2>Training</h2>" + _img(figures["history"]))
    if "confusion" in figures and metrics:
        labels = metrics["labels"]
        cm = metrics["confusion_matrix"]
        rows = [[labels[i]] + [str(v) for v in row] for i, row in enumerate(cm)]
        parts.append(
            "<h2>Test set</h2>" + _img(figures["confusion"])
            + "<details><summary>Confusion matrix as a table (rows = actual, columns = predicted)</summary>"
            + _table(["actual \\ predicted", *labels], rows) + "</details>"
        )
    if "per_class" in figures and metrics:
        rows = [
            [name, f"{m['precision']:.3f}", f"{m['recall']:.3f}", f"{m['f1']:.3f}", str(m.get("support", ""))]
            for name, m in metrics["per_class"].items()
        ]
        parts.append(
            _img(figures["per_class"])
            + _table(["command", "precision", "recall", "F1", "support"], rows)
        )
    if "snr" in figures and metrics and metrics.get("snr_sweep"):
        rows = [
            ["clean" if p["snr_db"] is None else f"{p['snr_db']:g} dB", f"{p['accuracy']:.4f}"]
            for p in metrics["snr_sweep"]
        ]
        parts.append(
            "<h2>Noise robustness</h2>" + _img(figures["snr"]) + _table(["condition", "accuracy"], rows)
        )
    if "errors" in figures:
        parts.append("<h2>Worst misclassifications</h2>" + _img(figures["errors"]))
    if "augmentation" in figures:
        parts.append("<h2>Augmentation preview</h2>" + _img(figures["augmentation"]))

    parts.append(
        "<h2>Config</h2><details open><summary>The exact configuration this run used</summary>"
        f"<pre>{yaml.safe_dump(config, sort_keys=False)}</pre></details>"
    )

    subtitle = config["data"]["commands"]
    html = (
        f"<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        f"<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
        f"<title>{run_name} — run report</title><style>{CSS}</style></head><body><div class=\"wrap\">"
        f"<h1>{run_name}</h1><p class=\"sub\">{config['model']['name']} · "
        f"{config['features']['type']} features · {len(subtitle)} commands: {', '.join(subtitle)}</p>"
        + "".join(parts)
        + "</div></body></html>"
    )

    out_path = run_dir / "report.html"
    out_path.write_text(html)
    return out_path


def load_metrics(run_dir: Path) -> dict | None:
    path = Path(run_dir) / "test_metrics.json"
    return json.loads(path.read_text()) if path.exists() else None
