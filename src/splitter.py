import random
from collections import defaultdict

from src.constants import SPLIT_SEED, TEST_RATIO, TRAIN_RATIO, VAL_RATIO


def assign_splits(
    rows: list[dict],
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    seed: int = SPLIT_SEED,
) -> list[dict]:
    """Group-aware, per-command-balanced train/val/test split.

    A raw source file and every augmented file derived from it share one group_id
    (assigned in Augmenter.augment_audio) and always land in the same split — this
    is what prevents near-duplicate leakage between train and test.
    """
    groups_by_command: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        groups_by_command[row["command"]].add(row["group_id"])

    split_by_group: dict[str, str] = {}
    rng = random.Random(seed)
    for command, group_set in groups_by_command.items():
        groups = sorted(group_set)  # sorted first: set order isn't stable, would break the seed
        rng.shuffle(groups)

        n = len(groups)
        if n < 3:
            raise ValueError(
                f"Command '{command}': only {n} group(s) available — need at least 3 "
                "(one per train/val/test). Increase sources_per_command."
            )

        # round() instead of int()-floor: floor sends every remainder to test and
        # can zero out val/test entirely at small n (e.g. n=5, val_ratio=0.15 -> 0).
        # The rebalance below still guarantees >=1 group in every split.
        n_train = max(1, round(n * train_ratio))
        n_val = max(1, round(n * val_ratio))
        if n_train + n_val > n - 1:
            n_train, n_val = n - 2, 1

        for g in groups[:n_train]:
            split_by_group[g] = "train"
        for g in groups[n_train : n_train + n_val]:
            split_by_group[g] = "val"
        for g in groups[n_train + n_val :]:
            split_by_group[g] = "test"

    for row in rows:
        row["split"] = split_by_group[row["group_id"]]

    return rows
