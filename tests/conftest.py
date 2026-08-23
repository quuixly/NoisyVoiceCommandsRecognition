import numpy as np
import pytest
import soundfile as sf

from src.config import Config

COMMANDS = ["up", "down", "left", "right", "stop", "go"]


@pytest.fixture
def corpus(tmp_path):
    """A miniature Speech Commands corpus: 12 speakers, each saying every command
    twice, with the real `<speaker>_nohash_<n>.wav` naming so the speaker-aware
    splitter has something to key off."""
    rng = np.random.default_rng(0)
    data_dir = tmp_path / "data"
    for command in COMMANDS:
        (data_dir / command).mkdir(parents=True)
        for speaker in range(12):
            for take in range(2):
                y = rng.normal(0, 0.1, 16_000).astype(np.float32)
                sf.write(data_dir / command / f"spk{speaker:02d}_nohash_{take}.wav", y, 16_000)
    return data_dir


@pytest.fixture
def config(corpus, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return Config.load(overrides=[
        f"data.data_dir={corpus.as_posix()}",
        "data.commands=[up, down, left, right, stop, go]",
        "train.epochs=1",
        "train.batch_size=8",
        "train.num_workers=0",
        "eval.batch_size=8",
        "eval.snr_db=[10]",
    ])
