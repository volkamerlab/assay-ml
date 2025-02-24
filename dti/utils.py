from typing import Union
import time
import logging
import sys

import torch
import numpy as np
import random

from .constants import OUTPUT

device = "cuda" if torch.cuda.is_available() else "cpu"


def set_random_seeds(seed: int):
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    random.seed(seed)
    np.random.seed(seed)


class LoggerWriter:
    # https://stackoverflow.com/questions/19425736/how-to-redirect-stdout-and-stderr-to-logger-in-python
    def __init__(self, level):
        self.level = level

    def write(self, message):
        if message != '\n':
            self.level(message)

    def flush(self):
        self.level(sys.stderr)


def init_logging(run_name: Union[str, None] = str(time.time())):
    (OUTPUT / run_name).mkdir(exist_ok=True, parents=True)
    log_file = OUTPUT / run_name / "output.log"
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(log_file)

    console_handler.setLevel(logging.INFO)
    file_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    # sys.stdout = LoggerWriter(logger.debug)
    # sys.stderr = LoggerWriter(logger.warning)

    logger.info(f"logging run {run_name} to {log_file}")


def write_header(run_name: str):
    write_info(
        run_name,
        [
            "model_type",
            "index",
            "epoch",
            "train_loss",
            "val_loss",
            "train_rank_corr",
            "val_rank_corr",
        ],
    )


def write_info(run_name: str, fields: list):
    """Write optimization data to a CSV file."""
    (OUTPUT / run_name).mkdir(exist_ok=True, parents=True)
    with open(OUTPUT / run_name / "optimization.csv", "a") as f:
        f.write(",".join(map(str, fields)) + "\n")
