import logging
import uuid
import sys
import random
from functools import partial
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.preprocessing import StandardScaler
from scipy.stats import kendalltau, spearmanr

from dti.model import (
    CombinedModel,
    MolecularModel,
    PairCombinedModel,
    PairMolecularModel,
)
from dti.data import (
    ActivityDataset,
    PairDataset,
    prepare_datasets,
    load_landrum,
    load_kinodata,
    load_atcc,
    load_split,
)
from dti.training import train_and_evaluate_model, AssayRankAccuracy
from dti.utils import (
    init_logging,
    set_random_seeds,
    write_header,
)
from dti.constants import DATA, ASSAY, COMPOUND

logger = logging.getLogger(__name__)


def model_and_dataset(method: str, mol_only: bool) -> Tuple[type, type]:
    match method:
        case "pair" if mol_only:
            return PairMolecularModel, PairDataset
        case "ic50" if mol_only:
            return MolecularModel, ActivityDataset
        case "pair":
            return PairCombinedModel, PairDataset
        case "ic50":
            return CombinedModel, ActivityDataset
        case _:
            logger.error(f"Unknown method: {method}")
            sys.exit(1)


def setup(method: str, dataset: str) -> Tuple[type, type, pd.DataFrame]:
    match dataset:
        case "kinodata":
            data = load_kinodata()
            model_cls, dataset_cls = model_and_dataset(method, False)
        case "landrum":
            data = load_landrum()
            model_cls, dataset_cls = model_and_dataset(method, False)
        case "large_landrum":
            data = load_landrum(DATA / "raw" / "landrum_large.csv")
            model_cls, dataset_cls = model_and_dataset(method, False)
        case "atcc":
            data = load_atcc()
            model_cls, dataset_cls = model_and_dataset(method, True)
        case _:
            logger.error(f"Unknown dataset: {dataset}")
            sys.exit(1)

    return model_cls, dataset_cls, data


def run_split(
    method: str,
    dataset: str,
    fold: int,
    seed: int,
    run_name: str,
    model_cls: type,
    dataset_cls: type,
    data: pd.DataFrame,
):
    batch_size = 512
    num_epochs = 50_000  # early stopping in place
    info_cols = [COMPOUND, ASSAY]
    data_dir = DATA / "processed" / dataset
    tgt_name = "scaled_ic50"

    if not (data_dir / f"split_{fold}.csv").exists():
        prepare_datasets(data, data_dir, tgt_name, 5, random_valset=False)

    train_data, val_data, test_data = load_split(fold, data_dir, tgt_name)
    val_dataset = dataset_cls(val_data, target=tgt_name, info_cols=info_cols)
    test_dataset = dataset_cls(test_data, target=tgt_name, info_cols=info_cols)

    scaler = StandardScaler()
    train_data[tgt_name] = scaler.fit_transform(
        train_data[tgt_name].values.reshape(-1, 1)
    )
    train_dataset = dataset_cls(train_data, target=tgt_name, info_cols=info_cols)

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(seed + fold)

    sampler = WeightedRandomSampler(
        train_dataset.weights, len(train_dataset), generator=g
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    rstat = partial(kendalltau, nan_policy="omit", variant="c")
    assay_rank = AssayRankAccuracy(data, method == "pair", rank_statistic=rstat)

    train_and_evaluate_model(
        model_cls,
        run_name,
        train_loader,
        val_loader,
        test_loader,
        method,
        fold,
        rank_corr_fn=assay_rank,
        embedding_size=512,
        num_epochs=num_epochs,
        cosine_agg=True,
        patience_termination=1000 if method == "pair" else 100,
        patience_lr=100 if method == "pair" else 10,
    )

    logger.info(f"{run_name} finished")


def main():
    seed = int(sys.argv[1])
    dataset = sys.argv[2]
    method = sys.argv[3]
    fold = int(sys.argv[4])

    run_name = f"{dataset}_{method}_{fold}_" + uuid.uuid4().hex[:4]
    init_logging(run_name)
    logger = logging.getLogger(run_name)
    logger.info(f"seed={seed} method={method} dataset={dataset} fold={fold}")

    set_random_seeds(seed)
    model_cls, dataset_cls, data = setup(method, dataset)
    write_header(run_name)

    run_split(method, dataset, fold, seed, run_name, model_cls, dataset_cls, data)


if __name__ == "__main__":
    main()
