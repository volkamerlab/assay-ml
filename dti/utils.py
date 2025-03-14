from typing import Union, Tuple
import functools
import subprocess
import time
import logging
import tarfile
from pathlib import Path
from enum import Enum
import random
from multiprocessing import Pool
import tempfile
import shutil

import torch
import pandas as pd
import numpy as np
import tqdm.auto as tqdm
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import rdFingerprintGenerator
from sklearn.cluster import AgglomerativeClustering

from .constants import OUTPUT, SMILES

device = "cuda" if torch.cuda.is_available() else "cpu"

logger = logging.getLogger(__name__)


class Method(Enum):
    IC50 = "ic50"
    IC50CORR = "ic50corr"
    HODGE = "hodge"
    PAIRS = "pairs"
    ALLPAIRS = "allpairs"
    SETS = "sets"
    ALLSETS = "allsets"
    IC50SETS = "ic50sets"
    IC50ALLSETS = "ic50allsets"

    @staticmethod
    def from_string(m: str):
        m_cleaned = m.upper().replace("_", "")
        match m_cleaned:
            case "IC50":
                return Method.IC50
            case "IC50CORR":
                return Method.IC50CORR
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
            case "IC50SETS":
                return Method.IC50SETS
            case "IC50ALLSETS":
                return Method.IC50ALLSETS
            case _:
                raise ValueError(f"Unknown method '{m_cleaned}'")

    @property
    def on_sets(self) -> bool:
        return self in [
            Method.SETS,
            Method.ALLSETS,
            Method.IC50SETS,
            Method.IC50ALLSETS,
        ]

    @property
    def on_pairs(self) -> bool:
        return self in [Method.PAIRS, Method.ALLPAIRS]

    @property
    def assay_based(self) -> bool:
        return self in [Method.PAIRS, Method.SETS, Method.HODGE, Method.IC50SETS]

    @property
    def point_prediction(self):
        return self in [Method.IC50, Method.HODGE, Method.IC50CORR]

    def __str__(self):
        return self.name.lower()

    def __repr__(self):
        return self.name.lower()


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
    if shutil.which("git") is None:
        logger.error("git not installed; no code snapshot")
        return
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


@functools.cache
def get_scaffold(smiles: str, generic: bool = True) -> str:
    try:
        mol = Chem.MolFromSmiles(smiles)
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        if generic:
            scaffold = MurckoScaffold.MakeScaffoldGeneric(scaffold)
        return Chem.CanonSmiles(Chem.MolToSmiles(scaffold))
    except Exception as e:
        logger.error(f"error processing SMILES {smiles}: {e}")
        return None


def compute_fp(smi: str, radius: int = 3, fp_dim: int = 2048):
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=fp_dim)
    try:
        return mfpgen.GetFingerprintAsNumPy(Chem.MolFromSmiles(smi))
    except TypeError:
        logger.warn(f"No fp for SMILES={smi}")
        return None


def scaffold_split(
    data: pd.DataFrame, proportion: float = 0.8, seed: int = 0, progress: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    np.random.seed(seed)
    smiles_it = tqdm.tqdm(data[SMILES], desc="scaffold") if progress else data[SMILES]
    data["scaffold"] = [get_scaffold(smi) for smi in smiles_it]

    data = data[~data["scaffold"].isna()]

    all_scaffolds = data["scaffold"].unique()
    np.random.shuffle(all_scaffolds)

    train_scaffolds = all_scaffolds[: int(len(all_scaffolds) * 0.8)]
    df_train = data[data["scaffold"].isin(train_scaffolds)]
    df_test = data[~data["scaffold"].isin(train_scaffolds)]

    return df_train, df_test


def umap_clusters(
    data: pd.DataFrame,
    proportions: float = 0.8,
    seed: int = 0,
    progress: bool = True,
    n_clusters: int = 10,
    n_jobs: int = 6,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Adapted from Pat Walter's useful rdkit utils
    """
    with tempfile.TemporaryDirectory() as tempdir:
        ac = AgglomerativeClustering(
            n_clusters=n_clusters, memory=tempdir, compute_full_tree=False
        )
    with Pool(n_jobs) as p:
        fp_list = p.map(compute_fp, data[SMILES].values)
    ac.fit_predict(np.stack(fp_list))
    return ac.labels_
