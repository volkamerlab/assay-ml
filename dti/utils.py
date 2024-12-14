import time
import logging
from pathlib import Path

import torch

DATA = Path(".") / "data"
SMILES = "compound_structures.canonical_smiles"
ACT = "activities.standard_value"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def init_logging():
    log_dir = DATA / "logs"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / (str(time.time()) + ".log")
    logging.basicConfig(
        filename=log_file,
        filemode="w",
        format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        level=logging.DEBUG,
    )
