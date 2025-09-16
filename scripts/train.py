import argparse
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
from scipy.stats import pearsonr

from dti.model import (
    CombinedModel,
    MolecularModel,
    PairCombinedModel,
    PairMolecularModel,
    ComplexSetRank,
    MoleculeSetRank,
    MoleculeBayesianSetRankModel,
    ComplexBayesianSetRankModel,
)
from dti.data import (
    ActivityDataset,
    SetActivityDataset,
    MultiSetActivityDataset,
    PairDataset,
    prepare_datasets,
    load_landrum,
    load_chembl_endpoints,
    load_kinodata,
    load_nci,
    load_solubility,
    load_lipo,
    load_clearance,
    load_activities,
    load_split,
)
from dti.featurization import MolFingerprint
from dti.training import (
    AssayRankAccuracy,
    train_and_evaluate_model,
    train_and_evaluate_pfn_model,
    batch_pair_loss,
    corr_loss,
)
from dti.utils import (
    Method,
    init_logging,
    set_random_seeds,
    save_code_snapshot,
)
from dti.constants import DATA, ASSAY, COMPOUND, HODGE

logger = logging.getLogger(__name__)


def setup(method: Method, dataset: str) -> Tuple[type, type, type, Callable]:
    match dataset.lower():
        case "chembl":
            data, mol_only = load_chembl_endpoints, True
        case "kinodata":
            data, mol_only = load_kinodata, False
        case "landrum":
            data, mol_only = load_landrum, False
        case "large_landrum":
            data_path = DATA / "raw" / "landrum_large.csv"
            data, mol_only = partial(load_landrum, data_path), False
        case "omnivore":
            data, mol_only = partial(load_landrum, DATA / "raw" / "omnivore.csv"), False
        case "activities":
            data, mol_only = load_activities, False
        case "solubility":
            data, mol_only = load_solubility, True
        case "lipo":
            data, mol_only = load_lipo, True
        case "clearance":
            data, mol_only = load_clearance, True
        case "cell_line":
            data, mol_only = partial(load_lipo, DATA / "raw" / "cell_line.csv"), True
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
    msa = partial(
        MultiSetActivityDataset,
        sets_per_batch=(10 if method == Method.PFN else 20),
        max_set_size=(500 if method == Method.PFN else 1000)
    )
    shuffled_multiset = partial(msa, inter_assay=True)
    match method:
        case Method.PFN if mol_only:
            return MoleculeBayesianSetRankModel, msa, msa
        case Method.PFN:
            return ComplexBayesianSetRankModel, msa, msa
        case Method.PAIRS if mol_only:
            return PairMolecularModel, PairDataset, PairDataset
        case Method.PAIRS:
            return PairCombinedModel, PairDataset, PairDataset
        case Method.ALLPAIRS if mol_only:
            return PairMolecularModel, ActivityDataset, PairDataset
        case Method.ALLPAIRS:
            return PairCombinedModel, ActivityDataset, PairDataset
        case Method.IC50CORR if mol_only:
            return MolecularModel, SetActivityDataset, msa
        case Method.IC50CORR:
            return CombinedModel, SetActivityDataset, msa
        case Method.HODGE | Method.IC50 if mol_only:
            return MolecularModel, ActivityDataset, msa
        case Method.HODGE | Method.IC50:
            return CombinedModel, ActivityDataset, msa
        case Method.IC50SETS | Method.SETS if mol_only:
            return MoleculeSetRank, msa, msa
        case Method.IC50SETS | Method.SETS:
            return ComplexSetRank, msa, msa
        case Method.IC50ALLSETS | Method.ALLSETS if mol_only:
            return MoleculeSetRank, shuffled_multiset, msa
        case Method.IC50ALLSETS | Method.ALLSETS:
            return ComplexSetRank, shuffled_multiset, msa
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


def prepare_dataset_splits(
    dataset_name: str,
    fold: int,
    method: Method,
    mol_feat: str,
    info_cols: list[str],
    batch_size: int,
):
    """Prepare and return model class, dataloaders, ligand_dim, and raw data."""
    data_dir = DATA / "processed" / dataset_name
    train_tgt = tgt_name = "scaled_ic50"
    aggregate = dataset_name != "chembl"
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
        )

    train_data, val_data, test_data = load_split(
        fold, data_dir, tgt_name, inter_assay_weight=inter_assay_weight
    )
    mol_feat = MolFingerprint(mol_feat)
    data_kwargs = dict(mol_featurizer=mol_feat, info_cols=info_cols)

    train_dataset = dataset_cls(train_data, target=train_tgt, **data_kwargs)
    val_dataset = val_dataset_cls(val_data, target=tgt_name, **data_kwargs)
    test_dataset = val_dataset_cls(test_data, target=tgt_name, **data_kwargs)

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

    return (
        model_cls,
        train_loader,
        val_loader,
        test_loader,
        mol_feat.dim,
        data,
        train_tgt,
    )


def run_split(
    run_name: str,
    mol_feat: str,
    method: Method,
    dataset_name: str,
    fold: int,
    seed: int,
):
    batch_size = 512
    num_epochs = 50_000  # early stopping in place
    info_cols = [COMPOUND, ASSAY]

    (
        model_cls,
        train_loader,
        val_loader,
        test_loader,
        ligand_dim,
        data,
        train_tgt,
    ) = prepare_dataset_splits(
        dataset_name, fold, method, mol_feat, info_cols, batch_size
    )

    rstat = pearsonr
    assay_rank = AssayRankAccuracy(data, method.on_pairs, rank_statistic=rstat)
    training_loss = loss_fn(method, nn.SmoothL1Loss(reduction="none"))
    train_short = method.on_sets or method.on_pairs

    logger.info(f"training target: {train_tgt}")

    args = [model_cls, run_name, train_loader, val_loader, test_loader, method, fold]
    kwargs = dict(
        ligand_dim=ligand_dim,
        multi_batch=method in [Method.SETS, Method.IC50CORR],
        batch_size=batch_size,
        rank_corr_fn=assay_rank,
        num_epochs=num_epochs,
        training_loss=training_loss,
        patience_termination=100 if train_short else 1000,
        patience_lr=10 if train_short else 100,
        fisher_transform=method not in [Method.IC50SETS, Method.IC50ALLSETS],
    )
    if model_cls in [ComplexBayesianSetRankModel, MoleculeBayesianSetRankModel]:
        train_and_evaluate_pfn_model(*args, **kwargs)
    else:
        train_and_evaluate_model(*args, **kwargs)

    logger.info(f"{run_name} finished")


def main():
    torch.cuda.empty_cache()

    parser = argparse.ArgumentParser(
        description="Run experiment with given parameters."
    )
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--method", type=str, required=True, help="Method to use")
    parser.add_argument("--fold", type=int, required=True, help="Fold number")
    parser.add_argument("--seed", type=int, default=1, help="Random seed (default: 1)")
    parser.add_argument(
        "--mol-feat",
        type=str,
        default="morgan",
        help="Molecular features (default: morgan)",
    )

    args = parser.parse_args()

    dataset_name = args.dataset.lower()
    method = Method(args.method)
    mol_feat = args.mol_feat.lower()

    run_name = "_".join(
        map(
            str, [dataset_name, mol_feat, args.fold, repr(method), uuid.uuid4().hex[:4]]
        )
    )
    init_logging(run_name)
    logger = logging.getLogger(run_name)
    logger.info(
        f"seed={args.seed} method={repr(method)} dataset={dataset_name} fold={args.fold}"
    )
    save_code_snapshot(run_name)

    set_random_seeds(args.seed)
    run_split(run_name, mol_feat, method, dataset_name, args.fold, args.seed)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        for line in traceback.format_exc().split("\n"):
            logger.error(line)
        logger.error(e)
