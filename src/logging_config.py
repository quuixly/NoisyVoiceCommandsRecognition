import logging


def setup_logging(level: int = logging.INFO) -> None:
    """Shared setup for every entry point, so the format can't drift between them."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
