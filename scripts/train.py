import logging
import traceback
import uuid
import sys
from functools import partial
from typing import Tuple, Callable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from scipy.stats import spearmanr, pearsonr

from dti.model import (
    CombinedModel,
    MolecularModel,
    PairCombinedModel,
    PairMolecularModel,
    SetRankModel,
    MoleculeSetRank,
)
from dti.data import (
    ActivityDataset,
    SetActivityDataset,
    MultiSetActivityDataset,
    PairDataset,
    prepare_datasets,
    load_landrum,
    load_kinodata,
    load_nci,
    load_split,
)
from dti.training import (
    AssayRankAccuracy,
    train_and_evaluate_model,
    batch_pair_loss,
    corr_loss,
)
from dti.utils import (
    Method,
    init_logging,
    set_random_seeds,
    save_code_snapshot,
)
from dti.constants import DATA, ASSAY, COMPOUND, HODGE, TID

logger = logging.getLogger(__name__)


def setup(method: str, dataset: str) -> Tuple[type, type, type, Callable]:
    match dataset.lower():
        case "kinodata":
            data, mol_only = load_kinodata, False
        case "landrum":
            data, mol_only = load_landrum, False
        case "large_landrum":
            data_path = DATA / "raw" / "landrum_large.csv"
            data, mol_only = partial(load_landrum, data_path), False
        case "omnivore":
            data, mol_only = partial(load_landrum, DATA / "raw" / "omnivore.csv"), False
        case "atcc" | "ovcar":
            data, mol_only = (
                partial(load_nci, DATA / "raw" / f"{dataset.lower()}.csv"),
                True,
            )
        case _:
            logger.error(f"Unknown dataset: {dataset}")
            sys.exit(1)

    model_cls, dataset_cls, val_dataset_cls = model_and_dataset(method, mol_only)
    return model_cls, dataset_cls, val_dataset_cls, data


def model_and_dataset(method: Method, mol_only: bool) -> Tuple[type, type, type]:
    shuffled_multiset = partial(MultiSetActivityDataset, inter_assay=True)
    match method:
        case Method.PAIRS if mol_only:
            return PairMolecularModel, PairDataset, PairDataset
        case Method.PAIRS:
            return PairCombinedModel, PairDataset, PairDataset
        case Method.ALLPAIRS if mol_only:
            return PairMolecularModel, ActivityDataset, PairDataset
        case Method.ALLPAIRS:
            return PairCombinedModel, ActivityDataset, PairDataset
        case Method.IC50CORR if mol_only:
            return MolecularModel, SetActivityDataset, MultiSetActivityDataset
        case Method.IC50CORR:
            return CombinedModel, SetActivityDataset, MultiSetActivityDataset
        case Method.HODGE | Method.IC50 if mol_only:
            return MolecularModel, ActivityDataset, MultiSetActivityDataset
        case Method.HODGE | Method.IC50:
            return CombinedModel, ActivityDataset, MultiSetActivityDataset
        case Method.IC50SETS | Method.SETS if mol_only:
            return MoleculeSetRank, MultiSetActivityDataset, MultiSetActivityDataset
        case Method.IC50SETS | Method.SETS:
            return SetRankModel, MultiSetActivityDataset, MultiSetActivityDataset
        case Method.IC50ALLSETS | Method.ALLSETS if mol_only:
            return MoleculeSetRank, shuffled_multiset, MultiSetActivityDataset
        case Method.IC50ALLSETS | Method.ALLSETS:
            return SetRankModel, shuffled_multiset, MultiSetActivityDataset
        case _:
            logger.error(f"No model and dataset configuration for method: {method}")
            sys.exit(1)


def train_batch(method: Method, default: int) -> int:
    if method == Method.ALLPAIRS:
        return int(np.sqrt(default))
    elif method.on_sets or method == Method.IC50CORR:
        return 1
    else:
        return default


def loss_fn(method: Method, default: Callable) -> Callable:
    match method:
        case Method.IC50CORR | Method.SETS | Method.ALLSETS:
            return corr_loss
        case Method.ALLPAIRS:
            return partial(batch_pair_loss, criterion=default)
        case Method.IC50ALLSETS | Method.IC50SETS:
            return nn.SmoothL1Loss()
        case _:
            return nn.SmoothL1Loss(reduction="none")


def test_batch(method: Method, default: int) -> int:
    return default if method.on_pairs else 1


def run_split(
    run_name: str,
    method: Method,
    dataset_name: str,
    fold: int,
    seed: int,
):
    batch_size = 512
    num_epochs = 50_000  # early stopping in place
    info_cols = [COMPOUND, ASSAY]
    data_dir = DATA / "processed_assay_target" / dataset_name
    train_tgt = tgt_name = "scaled_ic50"
    aggregate = True
    inter_assay_weight = None

    if method == Method.HODGE:
        inter_assay_weight = 0.0
        train_tgt = HODGE

    model_cls, dataset_cls, val_dataset_cls, load_data = setup(method, dataset_name)
    data = load_data()

    if not (data_dir / f"{fold}").exists():
        prepare_datasets(
            data,
            data_dir,
            5,
            random_valset=False,
            aggregate=aggregate,
            columns=[ASSAY, TID],
        )

    train_data, val_data, test_data = load_split(
        fold, data_dir, tgt_name, inter_assay_weight=inter_assay_weight
    )
    val_dataset = val_dataset_cls(val_data, target=tgt_name, info_cols=info_cols)
    test_dataset = val_dataset_cls(test_data, target=tgt_name, info_cols=info_cols)

    logger.info(f"training target: {train_tgt}")
    train_dataset = dataset_cls(train_data, target=train_tgt, info_cols=info_cols)

    assert len(train_dataset) > 0
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch(method, batch_size),
        shuffle=True,
        num_workers=0,
        drop_last=method.on_pairs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=test_batch(method, batch_size),
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch(method, batch_size),
        shuffle=False,
        num_workers=0,
    )

    # rstat = partial(spearmanr, nan_policy="raise")  # , variant="c")
    rstat = pearsonr
    assay_rank = AssayRankAccuracy(data, method.on_pairs, rank_statistic=rstat)
    multi_batch = method in [Method.SETS, Method.IC50CORR]
    training_loss = loss_fn(method, nn.SmoothL1Loss(reduction="none"))
    train_short = method.on_sets or method.on_pairs
    train_and_evaluate_model(
        model_cls,
        run_name,
        train_loader,
        val_loader,
        test_loader,
        method,
        fold,
        multi_batch=multi_batch,
        batch_size=batch_size,
        rank_corr_fn=assay_rank,
        embedding_size=512,
        num_epochs=num_epochs,
        cosine_agg=True,
        training_loss=training_loss,
        patience_termination=100 if train_short else 1000,
        patience_lr=10 if train_short else 100,
        normalize_training_batches=False,  # method.on_sets,
        lr=1e-4,
        fisher_transform=method not in [Method.IC50SETS, Method.IC50ALLSETS],
    )

    logger.info(f"{run_name} finished")


def main():
    torch.cuda.empty_cache()
    seed = int(sys.argv[1])
    dataset_name = sys.argv[2].lower()
    method = Method.from_string(sys.argv[3])
    fold = int(sys.argv[4])

    run_name = f"{dataset_name}_{fold}_{repr(method)}_" + uuid.uuid4().hex[:4]
    init_logging(run_name)
    logger = logging.getLogger(run_name)
    logger.info(f"seed={seed} method={repr(method)} dataset={dataset_name} fold={fold}")
    save_code_snapshot(run_name)

    set_random_seeds(seed)

    run_split(run_name, method, dataset_name, fold, seed)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        for line in traceback.format_exc().split("\n"):
            logger.error(line)
        logger.error(e)
