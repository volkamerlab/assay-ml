from typing import Union, Iterable
import functools
import subprocess
import time
import logging
import tarfile
from pathlib import Path
from enum import Enum, unique
import random
from multiprocessing import Pool
import shutil
import os, multiprocessing as mp

import torch
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import rdFingerprintGenerator
from rdkit import DataStructs
from rdkit.ML.Cluster import Butina
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA

from .constants import OUTPUT, SMILES

device = "cuda" if torch.cuda.is_available() else "cpu"

logger = logging.getLogger(__name__)


@functools.total_ordering
@unique
class Method(Enum):
    IC50 = "ic50"
    IC50CORR = "ic50corr"
    HODGE = "hodge"
    ALLPAIRS = "allpairs"
    PAIRS = "pairs"
    ALLSETS = "allsets"
    SETS = "sets"
    IC50ALLSETS = "ic50allsets"
    IC50SETS = "ic50sets"

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

    def _value(self):
        return list(__class__).index(self)

    def __lt__(self, other):
        assert isinstance(other, __class__)
        return self._value < other._value


def set_random_seeds(seed: int):
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    random.seed(seed)
    np.random.seed(seed)


def output_dir(run_name: str) -> Path:
    out_dir = OUTPUT / run_name
    out_dir.mkdir(exist_ok=True, parents=True)
    return out_dir


def get_tracked_files() -> list[Path]:
    try:
        result = subprocess.run(
            ["git", "ls-files"], capture_output=True, text=True, check=True
        )
        files = result.stdout.strip().split("\n")
        return [Path(f) for f in files if f]
    except subprocess.CalledProcessError:
        print("Error: Not a valid Git repository or issue running 'git ls-files'")
        return []


def save_code_snapshot(run_name: str):
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


@functools.cache
def compute_fp(smi: str, radius: int = 3, fp_dim: int = 2048, target: str = "numpy"):
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=fp_dim)
    try:
        match target:
            case "numpy":
                return mfpgen.GetFingerprintAsNumPy(Chem.MolFromSmiles(smi))
            case "native":
                return mfpgen.GetFingerprint(Chem.MolFromSmiles(smi))
            case _:
                raise ValueError(f"unkown fingerpritn target: '{target}'")
    except TypeError:
        logger.warn(f"No fp for SMILES={smi}")
        return None


def par_compute_fp(smiles: Iterable[str], n_jobs=16, target: str = "numpy"):
    with Pool(n_jobs) as p:
        return p.map(functools.partial(compute_fp, target=target), smiles)


def add_scaffold_col(
    data: pd.DataFrame, name: str = "_scaffold", progress: bool = True
):
    smiles_it = tqdm(data[SMILES], desc="scaffold") if progress else data[SMILES]
    data[name] = [get_scaffold(smi) for smi in smiles_it]


def scaffold_split(
    data: pd.DataFrame, proportion: float = 0.8, seed: int = 0, progress: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame]:
    np.random.seed(seed)
    _scaffold_key = "_scaffold"
    add_scaffold_col(data, _scaffold_key, progress=progress)
    data = data[~data[_scaffold_key].isna()]

    all_scaffolds = data[_scaffold_key].unique()
    np.random.shuffle(all_scaffolds)

    train_scaffolds = all_scaffolds[: int(len(all_scaffolds) * 0.8)]
    df_train = data[data[_scaffold_key].isin(train_scaffolds)]
    df_test = data[~data[_scaffold_key].isin(train_scaffolds)]

    return df_train, df_test


def butina_clusters(
    data: pd.DataFrame, cutoff: float = 0.2, label_col: str = "_butina"
):
    logger.info("Butina split: compute fingerprints")
    fp_list = par_compute_fp(data[SMILES].values, target="native")
    clusters = cluster_fingerprints(fp_list, cutoff=cutoff)
    labels = -np.ones(len(data), dtype=np.int64)
    for i, cluster in enumerate(clusters):
        labels[list(cluster)] = i
    data[label_col] = labels


_FP_LIST = None  # will become read‑only global inside each process


def _init_pool(fps):
    global _FP_LIST
    _FP_LIST = fps  # no pickling → shared (copy‑on‑write) memory


def _row_dissim_worker(i):
    sims = DataStructs.BulkTanimotoSimilarity(_FP_LIST[i], _FP_LIST[:i])
    return [1.0 - s for s in sims]


def tanimoto_distance_matrix(fp_list, n_processes=None, chunksize=20):
    n_processes = n_processes or os.cpu_count()

    with mp.get_context("fork").Pool(
        processes=n_processes, initializer=_init_pool, initargs=(fp_list,)
    ) as pool:
        work = pool.imap_unordered(
            _row_dissim_worker, range(1, len(fp_list)), chunksize
        )
        results = [None] * (len(fp_list) - 1)
        with tqdm(total=len(fp_list) - 1, desc="Tanimoto (mp‑fork)") as bar:
            for i, row in enumerate(work, 1):
                results[i - 1] = row
                bar.update()

    return [d for row in results for d in row]


def cluster_fingerprints(fingerprints, cutoff=0.2):
    """Cluster fingerprints
    Parameters:
        fingerprints
        cutoff: threshold for the clustering
    """
    # Calculate Tanimoto distance matrix
    distance_matrix = tanimoto_distance_matrix(fingerprints)
    # Now cluster the data with the implemented Butina algorithm:
    clusters = Butina.ClusterData(
        distance_matrix, len(fingerprints), cutoff, isDistData=True
    )
    clusters = sorted(clusters, key=len, reverse=True)
    return clusters


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
    fp_list = par_compute_fp(data[SMILES].values)

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


def read_predictions(
    p: Path,
    datasets: list[str] = ["omnivore", "landrum", "kinodata"],
    methods: list[str] = [Method.IC50, Method.ALLSETS, Method.SETS],
) -> pd.DataFrame:
    predictions = list()
    for run in tqdm(p.iterdir()):
        preds = run / "predictions.csv.gz"
        if not preds.exists():
            preds = run / "predictions.csv"
        if not preds.exists():
            logger.warn(f"no predictions file in {run}")
            continue
        parts = run.name.split("_")
        dataset, fold, method = parts[:3]
        if dataset not in datasets:
            continue
        method = Method.from_string(method)
        if method not in methods:
            continue
        fold = int(fold)
        df = pd.read_csv(preds, index_col=0)
        df["method"] = method
        df["fold"] = fold
        df["dataset"] = dataset
        df["coldtgt"] = parts[-1] == "coldtgt"
        predictions.append(df)
    return pd.concat(predictions)
