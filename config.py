from pathlib import Path

from loguru import logger

logger.add("proj.log", rotation="5 MB")


WORK_DIR = Path(__file__).parent
DATA_DIR = WORK_DIR / "data"
