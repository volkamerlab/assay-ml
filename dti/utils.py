from typing import Union, Iterable
import functools
import subprocess
import time
import logging
import tarfile
from pathlib import Path
from enum import Enum
import random
from multiprocessing import Pool
import shutil

import torch
import pandas as pd
import numpy as np
import tqdm.auto as tqdm
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import rdFingerprintGenerator
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA

from .constants import OUTPUT, SMILES

device = "cuda" if torch.cuda.is_available() else "cpu"

logger = logging.getLogger(__name__)


@functools.total_ordering
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
        match self:
            case Method.IC50:
                return "IC50"
            case Method.IC50CORR:
                return "IC50 corr."
            case Method.HODGE:
                return "Hodge"
            case Method.PAIRS:
                return "Assay PairModel"
            case Method.ALLPAIRS:
                return "Random PairModel"
            case Method.SETS:
                return "Assay SetRank"
            case Method.ALLSETS:
                return "Random SetRank"
            case Method.IC50SETS:
                return "Assay IC50 SetRank"
            case Method.IC50ALLSETS:
                return "Random IC50 SetRank"
            case _:
                return self.name.lower()

    def __repr__(self):
        return self.name.lower()

    def __lt__(self, other):
        return str(self) < str(other)


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


def par_compute_fp(smiles: Iterable[str], n_jobs=16):
    with Pool(n_jobs) as p:
        return p.map(compute_fp, data[SMILES].values)


def scaffold_split(
    data: pd.DataFrame, proportion: float = 0.8, seed: int = 0, progress: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame]:
    np.random.seed(seed)
    smiles_it = tqdm.tqdm(data[SMILES], desc="scaffold") if progress else data[SMILES]
    scaffold_key = "_scaffold"
    data[_scaffold_key] = [get_scaffold(smi) for smi in smiles_it]

    data = data[~data[_scaffold_key].isna()]

    all_scaffolds = data[_scaffold_key].unique()
    np.random.shuffle(all_scaffolds)

    train_scaffolds = all_scaffolds[: int(len(all_scaffolds) * 0.8)]
    df_train = data[data[_scaffold_key].isin(train_scaffolds)]
    df_test = data[~data[_scaffold_key].isin(train_scaffolds)]

    return df_train, df_test


def umap_split(
    data: pd.DataFrame,
    n_jobs: int = 6,
    pca_args: dict = dict(n_components=20),
    umap_args: dict = dict(n_components=2, n_neighbors=15, min_dist=0.1),
    cluster_args: dict = dict(n_clusters=5),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Adapted from Pat Walters' useful rdkit utils."""
    import umap

    logger.info("umap split: compute fingerprints")
    fp_list = par_compute_fp(data[SMILES].values, n_jobs=6)

    data = data[[fp is not None for fp in fp_list]]

    logger.info("umap split: PCA dimensionality reduction")
    pca = PCA(**pca_args)
    pcs = pca.fit_transform(np.stack(fp_list))

    logger.info("umap split: UMAP dimensionality reduction")
    reducer = umap.UMAP(**umap_args)
    embedding = reducer.fit_transform(pcs)

    logger.info("umap split: Agglomerative clustering")
    ac = AgglomerativeClustering(**cluster_args)
    ac.fit_predict(embedding)
    data["_umap"] = ac.labels_

    return data[data["_umap"] != 0], data[data["_umap"] == 0]
