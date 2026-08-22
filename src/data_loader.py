import logging
import shutil
from pathlib import Path

import kagglehub

from src.constants import COMMANDS_TO_PROCESS, DATA_DIR


class DatasetLoader:
    """Owns dataset acquisition: presence checks, download, and cache cleanup."""

    def __init__(
        self,
        data_dir: Path = DATA_DIR,
        commands: list[str] | None = None,
        kaggle_dataset: str = "yashdogra/speech-commands",
    ) -> None:
        self.data_dir = data_dir
        self.commands = commands if commands is not None else COMMANDS_TO_PROCESS
        self.kaggle_dataset = kaggle_dataset

    def check_dataset(self) -> bool:
        for command in self.commands:
            if not (self.data_dir / command).exists():
                return False
        return True

    def clear_cache(self, cache_dir: Path) -> None:
        try:
            shutil.rmtree(cache_dir)
            logging.info(f"Cleared kagglehub cache dir: {cache_dir}")
        except Exception as e:
            logging.warning(f"Could not clear the cache automatically. {e}")

    def load_dataset(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        if self.check_dataset():
            logging.info("Dataset already exists, skipping download")
            return

        logging.info("Dataset not found")
        cache_path = kagglehub.dataset_download(self.kaggle_dataset)
        cache_dir = Path(cache_path)
        logging.info(f"Dataset cached by kagglehub at: {cache_dir}")

        logging.info(f"Copying files to {self.data_dir} folder")
        for item in cache_dir.iterdir():
            if item.is_dir():
                if item.name in self.commands:
                    shutil.copytree(item, self.data_dir / item.name, dirs_exist_ok=True)

        logging.info(
            f"Specified parts from dataset successfully copied to {self.data_dir} folder"
        )

        # self.clear_cache(cache_dir)
