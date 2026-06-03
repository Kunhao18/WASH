import os
import logging

from logging.handlers import RotatingFileHandler


LEVEL_DICT = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}

global_logger = logging.getLogger("watermark_ensemble.stream")
file_logger = logging.getLogger("watermark_ensemble.file")


def setup_logger(
    log_level: str = "info",
    log_file: str = "blend.log",
    log_root: str = "./logs"
) -> None:
    os.makedirs(log_root, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    level = LEVEL_DICT[log_level.lower()]

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    stream_handler.setLevel(level)

    log_path = os.path.join(log_root, log_file)
    if os.path.exists(log_path):
        os.remove(log_path)
    file_handler = RotatingFileHandler(
        log_path, maxBytes=10 * 1024 * 1024, backupCount=1
    )
    file_handler.setFormatter(fmt)
    file_handler.setLevel(level)

    global_logger.addHandler(stream_handler)
    global_logger.setLevel(level)
    file_logger.addHandler(file_handler)
    file_logger.setLevel(level)
