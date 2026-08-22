from src.augmenter import Augmenter
from src.constants import AUG_DIR
from src.data_loader import DatasetLoader
from src.logging_config import setup_logging
from src.visualization import generate_mfcc

setup_logging()


def main() -> None:
    DatasetLoader().load_dataset()

    # Small, fast debug run — see prepare_dataset.py for the full training-prep pipeline.
    augmenter = Augmenter(interactive=False, num_preview_files=5, sources_per_command=5)
    augmenter.augment_audio()

    generate_mfcc(AUG_DIR)


if __name__ == "__main__":
    main()
