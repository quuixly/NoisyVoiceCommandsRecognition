import logging
import shutil
from pathlib import Path

import kagglehub
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


DATA_DIR = Path("data")
COMMANDS_TO_PROCESS = ["up", "down", "left", "right", "stop", "go"]


def check_dataset() -> bool:
    for command in COMMANDS_TO_PROCESS:
        if not (DATA_DIR / command).exists():
            return False
    return True


def clear_cache(cache_dir: Path) -> None:
    try:
        shutil.rmtree(cache_dir)
        logging.info("Base cache was cleared successfully")
        dataset_base_dir = cache_dir.parent.parent
        if dataset_base_dir.exists():
            shutil.rmtree(dataset_base_dir)
    except Exception as e:
        logging.warning(f"Could not clear the cache automatically. {e}")


def load_dataset() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if check_dataset():
        logging.info("Dataset already exists, skipping download")
        return

    logging.info("Dataset not found")
    cache_path = kagglehub.dataset_download("yashdogra/speech-commands")
    cache_dir = Path(cache_path)
    logging.info(f"Dataset cached by kagglehub at: {cache_dir}")

    logging.info(f"Copying files to {DATA_DIR} folder")
    for item in cache_dir.iterdir():
        if item.is_dir():
            if item.name in COMMANDS_TO_PROCESS:
                shutil.copytree(item, DATA_DIR / item.name, dirs_exist_ok=True)

    logging.info(
        f"Specified parts from dataset successfully copied to {DATA_DIR} folder"
    )

    # clear_cache(cache_dir)


MFCC_DIR = Path("mfcc")


def generate_mfcc() -> None:
    MFCC_DIR.mkdir(parents=True, exist_ok=True)
    for command in COMMANDS_TO_PROCESS:
        logging.info(f"Processing {DATA_DIR}/{command}")
        command_dir = DATA_DIR / command
        figures_dir = MFCC_DIR / command
        figures_dir.mkdir(parents=True, exist_ok=True)
        for number, file in enumerate(command_dir.iterdir()):
            logging.info(f"Processing {command} {number}")
            if file.is_file():
                y, sr = librosa.load(file)
                mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

                plt.figure(figsize=(10, 5))
                librosa.display.specshow(mfccs, x_axis="time")
                plt.colorbar()
                plt.title(f"MFCC - {command} - {number}")
                plt.xlabel("Time")
                plt.ylabel("MFCC Coefficients")
                plt.savefig(figures_dir / f"{command}_{number}.png")
                plt.close()


def main():
    load_dataset()
    generate_mfcc()


if __name__ == "__main__":
    main()
