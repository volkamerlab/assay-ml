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
    OUTPUT.mkdir(exist_ok=True)
    log_file = OUTPUT / (run_name + ".log")
    logging.basicConfig(
        filename=log_file,
        filemode="w",
        format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        level=logging.DEBUG,
    )
