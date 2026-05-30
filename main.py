import logging
import shutil
from itertools import combinations
from pathlib import Path

import kagglehub
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from audiomentations import (
    AddBackgroundNoise,
    AddGaussianNoise,
    ClippingDistortion,
    Compose,
    Gain,
    HighPassFilter,
    LowPassFilter,
    PitchShift,
    RoomSimulator,
    Shift,
    TimeStretch,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

N_MFCC = 40
MAX_PAD_LEN = 128
PRE_EMPHASIS = 0.97
TOP_DB = 30

DATA_DIR = Path("data")
MFCC_DIR = Path("mfcc")
AUG_DIR = Path("augmented")

COMMANDS_TO_PROCESS = ["up", "down", "left", "right", "stop", "go"]


SINGLE_TRANSFORMS = {
    "gaussian_noise": AddGaussianNoise(min_amplitude=0.005, max_amplitude=0.02, p=1.0),
    "time_stretch": TimeStretch(min_rate=0.8, max_rate=1.2, p=1.0),
    "pitch_shift": PitchShift(min_semitones=-3, max_semitones=3, p=1.0),
    "shift": Shift(min_shift=-0.2, max_shift=0.2, p=1.0),
    "gain": Gain(min_gain_db=-6, max_gain_db=6, p=1.0),
    "low_pass": LowPassFilter(min_cutoff_freq=1000, max_cutoff_freq=4000, p=1.0),
    "high_pass": HighPassFilter(min_cutoff_freq=100, max_cutoff_freq=1000, p=1.0),
    "clipping": ClippingDistortion(
        min_percentile_threshold=0, max_percentile_threshold=5, p=1.0
    ),
}

# Combination sizes to generate
COMBO_SIZES = [2, 3, 4, 5]


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


def augment_in_lib():
    # TODO: Batches?
    AUG_DIR.mkdir(parents=True, exist_ok=True)

    transform = AddGaussianNoise(min_amplitude=0.005, max_amplitude=0.02, p=1.0)

    limit = 10
    for command in COMMANDS_TO_PROCESS:
        (AUG_DIR / command).mkdir(parents=True, exist_ok=True)
        for idx, item in enumerate(
            list((DATA_DIR / command).iterdir())[:limit]
        ):  # list cast only for limit
            samples, sample_rate = sf.read(item)
            augmented_samples = transform(
                samples=samples.astype(np.float32), sample_rate=sample_rate
            )

            sf.write(
                AUG_DIR / command / (f"{command}_{idx}_noised.wav"),
                augmented_samples,
                sample_rate,
            )


def generate_mfcc(start_dir: Path) -> None:
    MFCC_DIR.mkdir(parents=True, exist_ok=True)
    for command in COMMANDS_TO_PROCESS:
        logging.info(f"Processing {start_dir}/{command}")
        command_dir = start_dir / command
        figures_dir = (
            MFCC_DIR / start_dir / command
        )  # TODO : REMOVE START DIR, only for comparison
        figures_dir.mkdir(parents=True, exist_ok=True)
        for number, file in enumerate(
            list(command_dir.iterdir())[:10]
        ):  # TODO : REMOVE LIMIT
            logging.info(f"Processing {start_dir}/{command} {number}")
            if file.is_file():
                y, sr = librosa.load(file, sr=16_000, mono=True)
                y, _ = librosa.effects.trim(y, top_db=TOP_DB)
                mfccs = librosa.feature.mfcc(
                    y=y,
                    sr=sr,
                    n_fft=512,
                    hop_length=160,
                    n_mels=40,
                    fmin=20,
                    fmax=sr // 2,
                )

                plt.figure(figsize=(10, 5))
                librosa.display.specshow(mfccs, x_axis="time")
                plt.colorbar()
                plt.title(f"MFCC - {command} - {start_dir.stem} - {number}")
                plt.xlabel("Time")
                plt.ylabel("MFCC Coefficients")
                plt.savefig(figures_dir / f"{command}_{number}.png")
                plt.close()

                # np.save(figures_dir / f"{command}_{number}.npy", features)


def main():
    # load_dataset()
    generate_mfcc(DATA_DIR)
    augment_in_lib()
    generate_mfcc(AUG_DIR)


if __name__ == "__main__":
    main()
