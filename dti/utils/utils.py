import os
import re
import uuid
import shutil
import tarfile
from typing import Union, List, Tuple
import functools
import subprocess
import time
import logging
from pathlib import Path
from enum import unique, StrEnum, auto
import random
import multiprocessing
import platform

import torch
import numpy as np
import polars as pl
from tqdm.auto import tqdm
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import DataStructs
from rdkit.ML.Cluster import Butina
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from rdkit.DataStructs.cDataStructs import TanimotoSimilarity
from rdkit import RDLogger
import yaml

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

from .constants import OUTPUT, SMILES

device = "cuda" if torch.cuda.is_available() else "cpu"

logger = logging.getLogger(__name__)


@functools.total_ordering
@unique
class Method(StrEnum):
    IC50 = auto()
    HODGE = auto()
    ALLPAIRS = auto()
    PAIRS = auto()
    ALLSETS = auto()
    SETS = auto()
    IC50ALLSETS = auto()
    IC50SETS = auto()
    PFN = auto()

    @property
    def on_sets(self) -> bool:
        return self in {
            Method.PFN,
            Method.SETS,
            Method.ALLSETS,
            Method.IC50SETS,
            Method.IC50ALLSETS,
        }

    @property
    def on_pairs(self) -> bool:
        return self in [Method.PAIRS, Method.ALLPAIRS]

    @property
    def assay_based(self) -> bool:
        return self in [
            Method.PFN,
            Method.PAIRS,
            Method.SETS,
            Method.HODGE,
            Method.IC50SETS,
        ]

    @property
    def point_prediction(self):
        return self in [Method.IC50, Method.HODGE]

    def __str__(self):
        match self:
            case Method.PFN:
                return "PFN"
            case Method.IC50:
                return "IC50"
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
        return self._value() < other._value()


def set_random_seeds(seed: int):
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    random.seed(seed)
    np.random.seed(seed)


_KNOWN_ACTIVATIONS = [
    torch.nn.modules.activation.Sigmoid,
    torch.nn.modules.activation.Tanh,
    torch.nn.modules.activation.ReLU,
    torch.nn.modules.activation.SiLU,
    torch.nn.modules.activation.GELU,
    torch.nn.modules.activation.ELU,
    torch.nn.modules.activation.SELU,
]


def activation_instance_representer(dumper, data):
    name = data.__class__.__name__
    return dumper.represent_scalar("!activation", name)


def activation_class_representer(dumper, data):
    if data in _KNOWN_ACTIVATIONS:
        name = data.__name__
        return dumper.represent_scalar("!activation_cls", name)
    return None


def activation_class_constructor(loader, node):
    class_name = loader.construct_scalar(node)
    try:
        act_class = getattr(torch.nn, class_name)
    except AttributeError as e:
        raise ValueError(f"Unknown activation class: {class_name}") from e
    return act_class


def activation_instance_constructor(loader, node):
    act_class = activation_class_constructor(loader, node)
    return act_class()


def register_torch_activation_with_yaml():
    for act_cls in _KNOWN_ACTIVATIONS:
        yaml.SafeDumper.add_multi_representer(act_cls, activation_instance_representer)
    yaml.SafeDumper.add_multi_representer(type, activation_class_representer)
    yaml.SafeLoader.add_constructor("!activation_cls", activation_class_constructor)
    yaml.SafeLoader.add_constructor("!activation", activation_instance_constructor)


def setup_multiprocessing_method():
    current_os = platform.system()

    # Determine the desired method
    if current_os in ["Windows", "Darwin"]:  # 'Darwin' is macOS
        method = "spawn"
    else:
        method = "fork"

    try:
        multiprocessing.set_start_method(method, force=True)
        logger.info(f"Multiprocessing start method set to: {method}")
    except RuntimeError:
        # This occurs if the start method has already been set elsewhere
        actual_method = multiprocessing.get_start_method()
        logger.warning(
            f"Could not set start method to {method}. "
            f"It is already set to {actual_method}."
        )


def output_dir(run_name: str) -> Path:
    out_dir = OUTPUT / run_name
    out_dir.mkdir(exist_ok=True, parents=True)
    return out_dir


def store_opts(opts: dict, run_name: str | None = None):
    run_name = run_name or opts["run_name"]
    opts["run_name"] = run_name
    with open((opts_file := output_dir(run_name) / "training_opts.yaml"), "w") as f:
        yaml.dump(opts, f, yaml.SafeDumper)
    logger.info(f"Training options saved to {opts_file}")
    return opts


def load_opts(opts_file: Path) -> dict:
    with open(opts_file, "r") as f:
        opts = yaml.load(f, Loader=yaml.SafeLoader)
    return opts


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


def get_condor_job_id():
    job_ad_path = os.getenv("_CONDOR_JOB_AD")
    if job_ad_path and os.path.exists(job_ad_path):
        with open(job_ad_path) as f:
            text = f.read()
        cluster = re.search(r"(?m)^ClusterId\s*=\s*(\d+)", text)
        proc = re.search(r"(?m)^ProcId\s*=\s*(\d+)", text)
        if cluster and proc:
            return f"{cluster.group(1)}.{proc.group(1)}"

    starter_pid = os.getenv("CONDOR_STARTER_PID")
    if starter_pid:
        return f"starter{starter_pid}"

    return uuid.uuid4().hex[:4]


def init_logging(run_name: Union[str, None] = str(time.time())):
    log_file = output_dir(run_name) / "output.log"
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(log_file)

    console_handler.setLevel(logging.DEBUG)
    file_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        f"%(asctime)s [{run_name}] [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    logger.info(f"logging run {run_name} to {log_file}")


@functools.cache
def get_scaffold(smiles: str, generic: bool = True) -> Union[str, None]:
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        if generic:
            scaffold = MurckoScaffold.MakeScaffoldGeneric(scaffold)
        return Chem.CanonSmiles(Chem.MolToSmiles(scaffold))
    except Exception as e:
        logger.error(f"error processing SMILES {smiles}: {e}")
        return None


def check_smi_valid(smi: str) -> bool:
    """Checks if a SMILES string is valid."""
    if smi is None or not isinstance(smi, str):
        return False
    return Chem.MolFromSmiles(smi) is not None


def add_scaffold_col(
    data: pl.DataFrame, name: str = "_scaffold", progress: bool = True
) -> pl.DataFrame:
    smiles_list = data[SMILES].to_list()
    if progress:
        scaffolds = [get_scaffold(smi) for smi in tqdm(smiles_list, desc="scaffold")]
    else:
        scaffolds = [get_scaffold(smi) for smi in smiles_list]
    return data.with_columns(pl.Series(name=name, values=scaffolds))


def scaffold_split(
    data: pl.DataFrame, proportion: float = 0.8, seed: int = 0, progress: bool = True
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """Splits a Polars DataFrame based on Murcko scaffolds."""
    np.random.seed(seed)
    _scaffold_key = "_scaffold"

    data = add_scaffold_col(data, _scaffold_key, progress=progress)

    data = data.filter(pl.col(_scaffold_key).is_not_null())

    all_scaffolds = data[_scaffold_key].unique().to_list()
    np.random.shuffle(all_scaffolds)

    split_idx = int(len(all_scaffolds) * proportion)
    train_scaffolds = set(all_scaffolds[:split_idx])

    df_train = data.filter(pl.col(_scaffold_key).is_in(train_scaffolds))
    df_test = data.filter(~pl.col(_scaffold_key).is_in(train_scaffolds))

    return df_train, df_test


def butina_clusters(
    data: pl.DataFrame,
    cutoff: float = 0.2,
    label_col: str = "_butina",
    smiles_col: str = SMILES,
) -> pl.DataFrame:
    uniq_smiles = data.select(pl.col(smiles_col)).unique()[smiles_col].to_numpy()

    logger.info(f"Butina: compute {len(uniq_smiles)} fingerprints")

    fp_list = par_compute_fp(uniq_smiles, target="native")

    valid_fp_indices = [i for i, fp in enumerate(fp_list) if fp is not None]
    valid_fps = [fp_list[i] for i in valid_fp_indices]
    valid_smiles = [uniq_smiles[i] for i in valid_fp_indices]

    clusters = cluster_fingerprints(valid_fps, cutoff=cutoff)

    uniq_labels = np.full(len(valid_smiles), -1, dtype=np.int64)
    for cid, cluster in enumerate(clusters):
        uniq_labels[list(cluster)] = cid

    smiles_to_cluster = pl.DataFrame({smiles_col: valid_smiles, label_col: uniq_labels})

    data = data.join(smiles_to_cluster, on=smiles_col, how="left")

    return data.with_columns(pl.col(label_col).fill_null(-1).cast(pl.Int64))


def umap_split(
    data: pl.DataFrame,
    n_jobs: int = 6,
    pca_args: dict = dict(n_components=20),
    umap_args: dict = dict(n_components=2, n_neighbors=15, min_dist=0.1),
    cluster_args: dict = dict(n_clusters=5),
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    import umap

    logger.info("umap split: compute fingerprints")

    smiles_array = data[SMILES].to_numpy()
    fp_list = par_compute_fp(smiles_array)

    valid_indices = [i for i, fp in enumerate(fp_list) if fp is not None]

    data = data.take(valid_indices)

    fp_list = [fp_list[i] for i in valid_indices]

    logger.info("umap split: PCA dimensionality reduction")
    pca = PCA(**pca_args)
    pcs = pca.fit_transform(np.stack(fp_list))

    logger.info("umap split: UMAP dimensionality reduction")
    reducer = umap.UMAP(**umap_args)
    embedding = reducer.fit_transform(pcs)

    logger.info("umap split: Agglomerative clustering")
    ac = AgglomerativeClustering(**cluster_args)
    ac.fit_predict(embedding)

    data = data.with_columns(pl.Series(name="_umap", values=ac.labels_))

    return data.filter(pl.col("_umap") != 0), data.filter(pl.col("_umap") == 0)


def read_predictions(
    p: Path,
    datasets: List[str] = ["omnivore", "landrum", "kinodata"],
    methods: List[Method] = [Method.IC50, Method.ALLSETS, Method.SETS],
) -> pl.DataFrame:
    """Reads prediction files from a directory structure into a single Polars DataFrame."""
    predictions = []

    read_fn = pl.read_csv

    for run in tqdm(p.iterdir()):
        preds = run / "predictions.csv.gz"
        if not preds.exists():
            preds = run / "predictions.csv"
        if not preds.exists():
            logger.warn(f"no predictions file in {run}")
            continue

        parts = run.name.split("_")

        if len(parts) < 3:
            logger.warn(f"Run name {run.name} has too few parts; skipping.")
            continue

        dataset, fold, method_str = parts[:3]

        if dataset not in datasets:
            continue

        try:
            method = Method.from_string(method_str)
        except AttributeError:
            logger.warn(f"Method {method_str} not recognized; skipping.")
            continue

        if method not in methods:
            continue

        try:
            fold = int(fold)
        except ValueError:
            logger.warn(f"Fold part {fold} is not an integer; skipping.")
            continue

        try:
            df = pl.read_csv(preds)

        except Exception as e:
            logger.error(f"Error reading prediction file {preds}: {e}")
            continue

        df = df.with_columns(
            [
                pl.lit(method).alias("method"),
                pl.lit(fold).alias("fold"),
                pl.lit(dataset).alias("dataset"),
                pl.lit(parts[-1] == "coldtgt").alias("coldtgt"),
            ]
        )

        predictions.append(df)

    if not predictions:
        return pl.DataFrame({})

    return pl.concat(predictions)


def tanimoto_distance_vector(fp_list):
    """Compute upper-triangle distance matrix as a flat list (Butina-compatible)"""
    distances = []
    for i in tqdm(range(1, len(fp_list)), desc="Tanimoto"):
        sims = DataStructs.BulkTanimotoSimilarity(fp_list[i], fp_list[:i])
        distances.extend(1.0 - s for s in sims)
    return distances


def tanimoto(x, y):
    return TanimotoSimilarity(x, y)


def cluster_fingerprints(fingerprints, cutoff=0.2):
    logger.info("Butina: Clustering")
    clusters = Butina.ClusterData(
        fingerprints,
        len(fingerprints),
        cutoff,
        isDistData=False,
        distFunc=TanimotoSimilarity,
    )
    return sorted(clusters, key=len, reverse=True)
