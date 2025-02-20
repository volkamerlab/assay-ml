import logging
import uuid
import sys
import random
from functools import partial
from typing import Tuple, Callable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.preprocessing import StandardScaler
from scipy.stats import kendalltau

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
from dti.training import train_and_evaluate_model, AssayRankAccuracy, batch_pair_loss
from dti.utils import (
    init_logging,
    set_random_seeds,
    write_header,
)
from dti.constants import DATA, ASSAY, COMPOUND

logger = logging.getLogger(__name__)


def model_and_dataset(method: str, mol_only: bool) -> Tuple[type, type, type]:
    match method:
        case "pair" if mol_only:
            return PairMolecularModel, PairDataset, PairDataset
        case "pair":
            return PairCombinedModel, PairDataset, PairDataset
        case "pair_all" if mol_only:
            return PairMolecularModel, ActivityDataset, PairDataset
        case "pair_all":
            return PairCombinedModel, ActivityDataset, PairDataset
        case "hodge" | "ic50" if mol_only:
            return MolecularModel, ActivityDataset, ActivityDataset
        case "hodge" | "ic50":
            return CombinedModel, ActivityDataset, ActivityDataset
        case _:
            logger.error(f"Unknown method: {method}")
            sys.exit(1)


def setup(method: str, dataset: str) -> Tuple[type, type, type, Callable]:
    match dataset:
        case "kinodata":
            data, mol_only = load_kinodata, False
        case "landrum":
            data, mol_only = load_landrum, False
        case "large_landrum":
            data_path = DATA / "raw" / "landrum_large.csv"
            data, mol_only = partial(load_landrum, data_path), False
        case "omnivore":
            data, mol_only = partial(load_landrum, DATA / "raw" / "omnivore.csv"), False
        case "atcc":
            data, mol_only = load_atcc, True
        case _:
            logger.error(f"Unknown dataset: {dataset}")
            sys.exit(1)

    model_cls, dataset_cls, val_dataset_cls = model_and_dataset(method, mol_only)
    return model_cls, dataset_cls, val_dataset_cls, data


def run_split(
    run_name: str,
    method: str,
    dataset_name: str,
    fold: int,
    seed: int,
):
    batch_size = 512
    num_epochs = 50_000  # early stopping in place
    info_cols = [COMPOUND, ASSAY]
    data_dir = DATA / "processed" / dataset_name
    tgt_name = "hodge_score" if method == "hodge" else "scaled_ic50"

    model_cls, dataset_cls, val_dataset_cls, load_data = setup(method, dataset_name)
    data = load_data()

    if not (data_dir / f"{fold}").exists():
        prepare_datasets(
            data, data_dir, 5, random_valset=False, aggregate=method == "hodge"
        )

    inter_assay_weight = 0.0 if method == "hodge" else None
    train_data, val_data, test_data = load_split(
        fold, data_dir, tgt_name, inter_assay_weight=inter_assay_weight
    )
    val_dataset = val_dataset_cls(val_data, target=tgt_name, info_cols=info_cols)
    test_dataset = val_dataset_cls(test_data, target=tgt_name, info_cols=info_cols)

    scaler = StandardScaler()
    train_data[tgt_name] = scaler.fit_transform(
        train_data[tgt_name].values.reshape(-1, 1)
    )
    train_tgt = tgt_name if method != "hodge" else "hodge_score"
    logger.info(f"training target: {train_tgt}")
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
    # 23 ** 2 ~ 512
    train_batch = 23 if method == "pair_all" else batch_size
    train_loader = DataLoader(
        train_dataset, batch_size=train_batch, sampler=sampler, drop_last=True
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    rstat = partial(kendalltau, nan_policy="omit", variant="c")
    assay_rank = AssayRankAccuracy(
        data, method in ["pair", "pair_all"], rank_statistic=rstat
    )
    training_loss = (
        partial(batch_pair_loss, criterion=nn.HuberLoss())
        if method == "pair_all"
        else nn.HuberLoss()
    )
    train_short = method in ["pair", "pair_all"]

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
        training_loss=training_loss,
        patience_termination=500 if train_short else 1000,
        patience_lr=50 if train_short else 100,
    )

    logger.info(f"{run_name} finished")


def main():
    seed = int(sys.argv[1])
    dataset_name = sys.argv[2]
    method = sys.argv[3]
    fold = int(sys.argv[4])

    run_name = f"{dataset_name}_{fold}_{method}_" + uuid.uuid4().hex[:4]
    init_logging(run_name)
    logger = logging.getLogger(run_name)
    logger.info(f"seed={seed} method={method} dataset={dataset_name} fold={fold}")

    set_random_seeds(seed)
    write_header(run_name)

    run_split(run_name, method, dataset_name, fold, seed)


if __name__ == "__main__":
    main()
