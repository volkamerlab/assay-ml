from typing import Union
import time
import logging
from pathlib import Path

import torch

DATA = Path(".") / "data"
OUTPUT = DATA / "output"
SMILES = "compound_structures.canonical_smiles"
ACT = "activities.standard_value"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def init_logging(run_name: Union[str, None] = str(time.time())):
    (OUTPUT / run_name).mkdir(exist_ok=True, parents=True)
    log_file = OUTPUT / run_name / "output.log"
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(log_file)

    console_handler.setLevel(logging.INFO)
    file_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
