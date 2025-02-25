from typing import Union
import subprocess
import time
import logging
import tarfile
from pathlib import Path
from enum import StrEnum, auto

import torch
import numpy as np
import random

from .constants import OUTPUT

device = "cuda" if torch.cuda.is_available() else "cpu"

logger = logging.getLogger(__name__)


class Method(StrEnum):
    IC50 = auto()
    HODGE = auto()
    PAIRS = auto()
    ALLPAIRS = auto()
    SETS = auto()
    ALLSETS = auto()

    @staticmethod
    def from_string(m: str):
        match m.upper().replace("_", ""):
            case "IC50":
                return Method.IC50
            case "HODGE":
                return Method.HODGE
            case "ALLPAIRS" | "PAIRALL":
                return Method.ALLPAIRS
            case "PAIR" | "PAIRS":
                return Method.PAIRS
            case "SET" | "SETS":
                return Method.SETS
            case "SETALL" | "ALLSETS":
                return Method.ALLSETS
            case _:
                raise ValueError(f"Unknown method '{m}'")

    @property
    def on_sets(self) -> bool:
        return self in [Method.SETS, Method.ALLSETS]

    @property
    def on_pairs(self) -> bool:
        return self in [Method.PAIRS, Method.ALLPAIRS]


def set_random_seeds(seed: int):
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    random.seed(seed)
    np.random.seed(seed)


def output_dir(run_name: str) -> Path:
    out_dir = OUTPUT / run_name
    out_dir.mkdir(exist_ok=True, parents=True)
    return out_dir


def get_tracked_files():
    try:
        result = subprocess.run(
            ["git", "ls-files"], capture_output=True, text=True, check=True
        )
        files = result.stdout.strip().split("\n")
        return [Path(f) for f in files if f]
    except subprocess.CalledProcessError:
        print("Error: Not a valid Git repository or issue running 'git ls-files'")
        return []


def save_code_snapshot(run_name):
    archive_name = output_dir(run_name) / "code.tar.gz"
    python_files = get_tracked_files()
    if not python_files:
        logger.warn("No tracked Python files found.")
        return

    with tarfile.open(archive_name, "w:gz") as tar:
        for py_file in python_files:
            tar.add(py_file, arcname=py_file)

    logger.info(f"code archive created: {archive_name}")


def init_logging(run_name: Union[str, None] = str(time.time())):
    log_file = output_dir(run_name) / "output.log"
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
    with open(output_dir(run_name) / "optimization.csv", "a") as f:
        f.write(",".join(map(str, fields)) + "\n")
