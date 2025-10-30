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

from dti.model import MoleculeBayesianSetRankModel
from dti.data import (
    prepare_datasets,
    load_chembl_endpoints,
    load_split,
    MultiSetMapDataset,
    CostBasedBatchSampler,
    SetCollator,
)
from dti.featurization import MolFingerprint
from dti.training import train_and_evaluate_pfn_model
from dti.utils import (
    Method,
    device,
    init_logging,
    set_random_seeds,
    save_code_snapshot,
)
from dti.constants import ACT, DATA, ASSAY, COMPOUND, INTRA_ASSAY_TEST, IDENT

logger = logging.getLogger(__name__)


def get_model_and_dataset_types() -> Tuple[type, type, type]:
    """
    Returns the hard-coded model and dataset classes for PFN on chembl.
    """
    pfn_map_dataset = partial(
        MultiSetMapDataset,
        max_set_size=2048,
        shuffle_within_target=False,
        property_columns=[
            "mw_freebase",
            "alogp",
            "hba",
            "hbd",
            "psa",
            "rtb",
            "num_ro5_violations",
            "full_mwt",
            "aromatic_rings",
            "heavy_atoms",
            "qed_weighted",
            "np_likeness_score",
        ],
    )
    pfn_map_dataset_val = partial(
        MultiSetMapDataset,
        max_set_size=2048,
        shuffle_within_target=False,
        query_column=INTRA_ASSAY_TEST,
        property_columns=[],
        property_set_ratio=0.0,
    )
    return MoleculeBayesianSetRankModel, pfn_map_dataset, pfn_map_dataset_val


def prepare_dataset_splits(
    dataset_name: str,
    fold: int,
    mol_feat: str,
    info_cols: list[str],
    property_set_ratio: float,
):
    """Prepare and return model class, dataloaders, and ligand_dim."""
    data_dir = DATA / "processed" / dataset_name

    model_cls, dataset_cls, val_dataset_cls = get_model_and_dataset_types()

    if not (data_dir / f"{fold}").exists():
        data = load_chembl_endpoints()
        prepare_datasets(
            data,
            data_dir,
            5,
            random_valset=False,
            aggregate=False,
        )

    train_data, val_data, test_data = load_split(
        fold,
        data_dir,
        ACT,
        inter_assay_weight=None,
        scale_targets=False,
    )

    mol_feat = MolFingerprint(mol_feat)
    data_kwargs = dict(
        mol_featurizer=mol_feat,
        info_cols=info_cols,
        property_set_ratio=property_set_ratio,
    )

    val_dataset = val_dataset_cls(val_data, target=ACT, **data_kwargs)
    test_dataset = val_dataset_cls(test_data, target=ACT, **data_kwargs)
    train_dataset = dataset_cls(train_data, target=ACT, **data_kwargs)

    assert len(train_dataset) > 0

    logger.info("Using Map-style Dataset with CostBasedBatchSampler for PFN.")

    MAX_COST = 1 * 2048**2  # 1,048,576

    train_sampler = CostBasedBatchSampler(
        set_sizes=train_dataset.set_sizes, max_batch_cost=MAX_COST, shuffle=True
    )
    val_sampler = CostBasedBatchSampler(
        set_sizes=val_dataset.set_sizes, max_batch_cost=MAX_COST, shuffle=False
    )
    test_sampler = CostBasedBatchSampler(
        set_sizes=test_dataset.set_sizes, max_batch_cost=MAX_COST, shuffle=False
    )

    train_collator = SetCollator(
        query_ratio=train_dataset.query_ratio,
        determistic_queries=train_dataset.determistic_queries,
        info_accessor=train_dataset.info,
        device=device,
    )
    val_collator = SetCollator(
        determistic_queries=True,
        query_ratio=0.0,
        info_accessor=test_dataset.info,
        device=device,
    )


    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        collate_fn=train_collator,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_sampler=val_sampler,
        collate_fn=val_collator,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_sampler=test_sampler,
        collate_fn=val_collator,
        num_workers=0,  # No need for workers on test set
        pin_memory=True,
    )

    return (
        model_cls,
        train_loader,
        val_loader,
        test_loader,
        mol_feat.dim,
    )


def run_split(
    run_name: str,
    mol_feat: str,
    fold: int,
    seed: int,
    unmasked_weight: float,
    property_set_ratio: float,
    n_bins: int,
):
    method = Method.PFN
    dataset_name = "chembl"

    num_epochs = 50_000
    info_cols = [INTRA_ASSAY_TEST, IDENT, COMPOUND, ASSAY]

    (
        model_cls,
        train_loader,
        val_loader,
        test_loader,
        ligand_dim,
    ) = prepare_dataset_splits(
        dataset_name,
        fold,
        mol_feat,
        info_cols,
        property_set_ratio,
    )

    logger.info(f"training target: {ACT}")
    logger.info(f"test target: {ACT}")

    args = [model_cls, run_name, train_loader, val_loader, test_loader, method, fold]

    kwargs = dict(
        ligand_dim=ligand_dim,
        num_epochs=num_epochs,
        patience_termination=100,
        patience_lr=10,
        unmasked_weight=unmasked_weight,
        n_bins=n_bins,
    )

    train_and_evaluate_pfn_model(*args, **kwargs)
    logger.info(f"{run_name} finished")


def main():
    torch.cuda.empty_cache()

    parser = argparse.ArgumentParser(
        description="Run PFN experiment on ChEMBL dataset."
    )
    parser.add_argument("--fold", type=int, required=True, help="Fold number")
    parser.add_argument("--seed", type=int, default=1, help="Random seed (default: 1)")
    parser.add_argument(
        "--mol-feat",
        type=str,
        default="morgan",
        help="Molecular features (default: morgan)",
    )
    parser.add_argument(
        "--unmasked-weight",
        type=float,
        default=0.0,
        help="[PFN] Weight of reconstruction on unmasked samples. (default: 0.0)",
    )
    parser.add_argument(
        "--property-set-ratio",
        type=float,
        default=0.5,
        help="[PFN] Proportion of physiochemical property sets. (default: 0.5)",
    )
    parser.add_argument(
        "--n-bins",
        type=int,
        default=100,
        help="[PFN] Number of bins in distribution. (default: 100)",
    )

    args = parser.parse_args()

    dataset_name = "chembl"
    method = Method.PFN
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
    logger.info(f"device: {device}")
    save_code_snapshot(run_name)

    set_random_seeds(args.seed)
    run_split(
        run_name,
        mol_feat,
        args.fold,
        args.seed,
        args.unmasked_weight,
        args.property_set_ratio,
        args.n_bins,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(traceback.format_exc())
        logger.error(e)
