"""Nested-dataclass config loaded from YAML.

One `Config` object carries every knob any stage needs, so a run is fully described
by the YAML it was launched with. `Config.load()` deep-merges an experiment file onto
`configs/default.yaml`, then applies `--set a.b=c` overrides, so an experiment file
only has to name what it changes.
"""

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

# Resolved against the package, not the working directory: `configs/` ships with the
# repository, so main.py works from any directory instead of only the repository root.
# A local override is what `-c` is for.
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"


@dataclass
class DataConfig:
    kaggle_dataset: str = "yashdogra/speech-commands"
    data_dir: str = "data"
    commands: list[str] = field(default_factory=lambda: ["up", "down", "left", "right", "stop", "go"])
    sample_rate: int = 16_000
    clip_seconds: float = 1.0
    max_per_command: int | None = None

    @property
    def clip_samples(self) -> int:
        return int(self.sample_rate * self.clip_seconds)


@dataclass
class SplitConfig:
    by: str = "speaker"  # speaker | file
    train: float = 0.7
    val: float = 0.15
    test: float = 0.15
    seed: int = 42


@dataclass
class FeatureConfig:
    type: str = "mfcc"  # mfcc | logmel
    n_fft: int = 512
    hop_length: int = 160
    n_mels: int = 64
    fmin: int = 20
    fmax: int = 8000
    n_mfcc: int = 20
    deltas: bool = True
    pre_emphasis: float = 0.97
    trim_top_db: float | None = None
    normalize: str = "per_file"  # per_file | none
    max_frames: int = 101

    @property
    def shape(self) -> tuple[int, int, int]:
        """(channels, rows, frames) of one extracted example."""
        if self.type == "logmel":
            return (1, self.n_mels, self.max_frames)
        if self.type == "mfcc":
            return (3 if self.deltas else 1, self.n_mfcc, self.max_frames)
        raise ValueError(f"Unknown features.type={self.type!r} (expected 'mfcc' or 'logmel')")

    def fingerprint(self) -> str:
        """Short hash of every field that changes the extracted values. Names the
        feature cache, so changing any knob here yields a different cache file
        instead of silently reusing stale arrays."""
        payload = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha1(payload.encode()).hexdigest()[:10]


@dataclass
class AugmentConfig:
    enabled: bool = True
    min_k: int = 1
    max_k: int = 2
    transforms: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass
class ModelConfig:
    name: str = "baseline_cnn"
    channels: list[int] = field(default_factory=lambda: [32, 64, 96, 128])
    dropout: float = 0.3
    rnn_hidden: int = 96

    def kwargs(self) -> dict[str, Any]:
        """Everything except `name`, forwarded to the model constructor."""
        return {f.name: getattr(self, f.name) for f in fields(self) if f.name != "name"}


@dataclass
class TrainConfig:
    # Empty means "not chosen yet"; Config.resolve() fills in a timestamp. Typed as a
    # plain str rather than str | None so everything downstream — run_dirs(), the
    # report builder — can take it as a str without a None check that resolve()
    # has already made impossible.
    run_name: str = ""
    epochs: int = 30
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 1e-4
    label_smoothing: float = 0.05
    warmup_epochs: int = 2
    grad_clip: float = 1.0
    patience: int = 7
    device: str = "auto"
    amp: bool = False
    num_workers: int = 7
    overfit_batch: bool = False
    resume: bool = False


@dataclass
class EvalConfig:
    batch_size: int = 128
    snr_db: list[float] = field(default_factory=list)
    noise: str = "gaussian"


@dataclass
class Config:
    seed: int = 42
    data: DataConfig = field(default_factory=DataConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    augment: AugmentConfig = field(default_factory=AugmentConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    @classmethod
    def load(
        cls,
        path: str | Path | None = None,
        overrides: list[str] | None = None,
        base: str | Path = DEFAULT_CONFIG_PATH,
    ) -> "Config":
        raw = _read_yaml(Path(base))
        if path is not None:
            _deep_merge(raw, _read_yaml(Path(path)))
        for override in overrides or []:
            _apply_override(raw, override)
        config = _from_dict(cls, raw)
        config.resolve()
        return config

    def resolve(self) -> None:
        """Fills in derived defaults that shouldn't be hardcoded in the YAML."""
        if not self.train.run_name:
            # .astimezone() makes the local time explicitly tz-aware; run names are
            # read by a human next to their own clock, so local beats UTC here.
            self.train.run_name = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")

    def to_dict(self) -> dict:
        return asdict(self)

    def corpus_fingerprint(self) -> str:
        """Short hash of everything that decides which clips exist and where they go:
        the data selection and the split. `main.py run` compares it against the stamp
        left by the last prepare, so changing `data.max_per_command` or `split.by`
        re-prepares instead of training on a manifest that no longer matches."""
        return corpus_fingerprint_of(self.to_dict())

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))


def corpus_fingerprint_of(raw: dict) -> str:
    """Corpus hash from a plain config dict — so a saved `reports/<run>/config.yaml`
    can be fingerprinted without rebuilding a `Config`. Two runs sharing this hash
    were trained and tested on the same clips in the same splits, which is the
    precondition for their numbers being comparable at all."""
    payload = json.dumps({"data": raw.get("data", {}), "split": raw.get("split", {})}, sort_keys=True)
    return hashlib.sha1(payload.encode()).hexdigest()[:10]


def _read_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    return yaml.safe_load(path.read_text()) or {}


def _deep_merge(base: dict, override: dict) -> dict:
    """In-place recursive merge. Dicts merge key-by-key; every other type replaces
    wholesale, so a list in an experiment file overrides rather than appends."""
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def _apply_override(raw: dict, override: str) -> None:
    """`--set train.lr=1e-3` -> raw["train"]["lr"] = 0.001. Values go through the YAML
    scalar parser, so ints/floats/bools/null/lists are typed, not left as strings."""
    if "=" not in override:
        raise ValueError(f"--set expects key.path=value, got {override!r}")
    key_path, _, value_str = override.partition("=")
    node = raw
    keys = key_path.strip().split(".")
    for key in keys[:-1]:
        node = node.setdefault(key, {})
        if not isinstance(node, dict):
            raise TypeError(f"--set {override!r}: '{key}' is not a section")
    node[keys[-1]] = _parse_value(value_str)


def _parse_value(text: str):
    """YAML 1.1 only recognizes a float in exponent form when it carries an explicit
    decimal point, so `--set train.lr=1e-3` would otherwise arrive as the string
    "1e-3" and reach the optimizer intact. Try Python's numeric literals first."""
    text = text.strip()
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return yaml.safe_load(text)


def _from_dict(cls: Any, raw: dict) -> Any:
    """Builds nested dataclasses from plain dicts, rejecting unknown keys so a typo
    in a YAML file fails loudly instead of being silently ignored.

    `cls` is annotated `Any` rather than `type`: this walks dataclass fields
    reflectively, and `Field.type` is only known to be a dataclass at runtime — a
    static annotation here would be a claim the type checker cannot verify and every
    call site would need a cast to satisfy.
    """
    known = {f.name: f for f in fields(cls)}
    unknown = set(raw) - set(known)
    if unknown:
        raise ValueError(f"Unknown config key(s) for {cls.__name__}: {sorted(unknown)}")
    kwargs = {}
    for name, f in known.items():
        if name not in raw:
            continue
        value = raw[name]
        if is_dataclass(f.type) and isinstance(value, dict):
            kwargs[name] = _from_dict(f.type, value)
        else:
            kwargs[name] = value
    return cls(**kwargs)
