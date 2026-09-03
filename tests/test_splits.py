from collections import defaultdict

from src.config import SplitConfig
from src.data.splits import assign_splits, speaker_of


def _rows():
    return [
        {"file_path": f"data/{cmd}/spk{s:02d}_nohash_{t}.wav", "command": cmd, "speaker": f"spk{s:02d}"}
        for cmd in ["up", "down", "left"]
        for s in range(12)
        for t in range(2)
    ]


def test_speaker_of_parses_speech_commands_names():
    assert speaker_of("00b01445_nohash_1.wav") == "00b01445"


def test_no_speaker_appears_in_two_splits():
    rows = assign_splits(_rows(), SplitConfig(by="speaker"))
    splits_per_speaker = defaultdict(set)
    for row in rows:
        splits_per_speaker[row["speaker"]].add(row["split"])
    assert all(len(s) == 1 for s in splits_per_speaker.values())


def test_every_split_is_non_empty_for_every_command():
    rows = assign_splits(_rows(), SplitConfig(by="speaker"))
    seen = defaultdict(set)
    for row in rows:
        seen[row["command"]].add(row["split"])
    assert all(s == {"train", "val", "test"} for s in seen.values())


def test_split_is_reproducible_under_a_fixed_seed():
    a = {r["file_path"]: r["split"] for r in assign_splits(_rows(), SplitConfig())}
    b = {r["file_path"]: r["split"] for r in assign_splits(_rows(), SplitConfig())}
    assert a == b
