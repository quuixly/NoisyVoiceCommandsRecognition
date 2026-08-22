import logging


def setup_logging(level: int = logging.INFO) -> None:
    """Shared basicConfig for every entry point (main.py, prepare_dataset.py, ...)
    so the format/datefmt can't drift between them."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
