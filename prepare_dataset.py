import logging

from src.augmenter import Augmenter
from src.constants import MANIFEST_PATH
from src.data_loader import DatasetLoader
from src.feature_extractor import FeatureExtractor
from src.labels import save_label_map
from src.logging_config import setup_logging
from src.manifest import write_manifest
from src.splitter import assign_splits

setup_logging()


def prepare_dataset(sources_per_command: int = 100) -> None:
    DatasetLoader().load_dataset()

    augmenter = Augmenter(sources_per_command=sources_per_command)
    augmenter.augment_audio()
    save_label_map()

    # split before extraction: FeatureExtractor packs one array per split
    rows = assign_splits(augmenter.manifest_rows)
    rows = FeatureExtractor().extract_all(rows)
    write_manifest(rows, MANIFEST_PATH)

    logging.info(f"Dataset preparation complete. Manifest: {MANIFEST_PATH} ({len(rows)} rows)")


if __name__ == "__main__":
    prepare_dataset()
