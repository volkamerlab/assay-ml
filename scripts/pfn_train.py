import argparse
import logging
from functools import partial
import traceback
from pathlib import Path
import os

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from dti.model.bayesian import (
    MoleculeBayesianSetRankModel,
    GraphMoleculeBayesianSetRankModel,
    AllMoleculeBayesianSetRankModel,
    EarlyFusionAllMoleculeBayesianSetRankModel,
)
from dti.data.dataset import (
    PropertySetDataset,
    ResettingBatchSampler,
    GraphPropertySetDataset,
    GraphAndFingerprintDataset,
)
from dti.data.processing import (
    prepare_datasets,
    load_chembl_endpoints,
    load_split,
)
from dti.data.featurization import MolFingerprint
from dti.training.bayesian import train_and_evaluate_pfn_model
from dti.utils import (
    Method,
    init_logging,
    set_random_seeds,
    get_condor_job_id,
)
from dti.utils.constants import (
    ACT,
    DATA,
    IDENT,
    COMPOUND,
    ASSAY,
    INTRA_ASSAY_TEST,
)


logger = logging.getLogger(__name__)

DATASET_NAME = "chembl"
METHOD = Method("pfn")
MOL_ONLY = True


def prepare_dataset_splits(
    dataset_name: str,
    fold: int,
    method: "Method",
    mol_feat: str,
    info_cols: list[str],
    property_set_ratio: float,
    n_jobs: int = 8,
):
    data_dir = DATA / "processed" / dataset_name

    if not (data_dir / f"{fold}").exists():
        data = load_chembl_endpoints()
        prepare_datasets(data, data_dir, 5, random_valset=False, aggregate=False)

    train_data, val_data, test_data = load_split(
        fold,
        data_dir,
        ACT,
        inter_assay_weight=None,
        scale_targets=False,
    )

    mol_feat_instance = MolFingerprint(mol_feat)

    common_dataset_kwargs = {
        "mol_featurizer": mol_feat_instance,
        "info_cols": info_cols,
        "target": ACT,
        "processed_dir": data_dir,
    }

    match mol_feat_instance:
        case MolFingerprint.GRAPH:
            dataset_cls = GraphPropertySetDataset
        case MolFingerprint.ALL:
            dataset_cls = GraphAndFingerprintDataset
            common_dataset_kwargs["graph_featurizer"] = MolFingerprint.GRAPH
            common_dataset_kwargs["mol_featurizer"] = MolFingerprint.ALLFP
        case _:
            dataset_cls = PropertySetDataset

    val_dataset_cls = partial(
        dataset_cls,
        max_batch_datapoints=2048,
        max_set_size=2048,
        shuffle_within_target=False,
        property_columns=[],
        property_set_ratio=0.0,
        query_col=INTRA_ASSAY_TEST,
        **common_dataset_kwargs,
    )

    cache_dir = (
        Path(os.environ.get("CACHE_DIR", data_dir))
        / dataset_name
        / str(fold)
        / mol_feat_instance.value
    )
    val_dataset = val_dataset_cls(val_data, cache_dir=cache_dir / "val")
    test_dataset = val_dataset_cls(test_data, cache_dir=cache_dir / "test")

    train_dataset = dataset_cls(
        train_data,
        max_batch_datapoints=2560,
        max_set_size=1024,
        shuffle_within_target=False,
        property_set_ratio=property_set_ratio,
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
        cache_dir=cache_dir / "train",
        estimate_deg=mol_feat_instance.graph_based,
        query_col=None,
        mask_fraction=0.2,
        **common_dataset_kwargs,
    )

    assert len(train_dataset) > 0

    if mol_feat_instance in (MolFingerprint.GRAPH, MolFingerprint.ALL):
        collate_fn = lambda data: data[0]
    else:
        collate_fn = None

    data_loader_cls = DataLoader

    loader_kwargs = {
        "batch_sampler": ResettingBatchSampler(train_dataset, batch_size=1),
        "shuffle": False,
        "num_workers": n_jobs,
        "collate_fn": collate_fn,
        "persistent_workers": True,
    }

    train_loader = data_loader_cls(train_dataset, drop_last=False, **loader_kwargs)

    val_loader = data_loader_cls(
        val_dataset,
        batch_sampler=ResettingBatchSampler(val_dataset, batch_size=1),
        shuffle=False,
        num_workers=n_jobs,
        collate_fn=collate_fn,
    )
    test_loader = data_loader_cls(
        test_dataset,
        batch_sampler=ResettingBatchSampler(test_dataset, batch_size=1),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    return (
        train_loader,
        val_loader,
        test_loader,
        mol_feat_instance.dim,
    )


def run_split(
    run_name: str,
    mol_feat: str,
    fold: int,
    property_set_ratio: float,
    n_bins: int,
    **kwargs,
):
    info_cols = [IDENT, COMPOUND, ASSAY]

    (
        train_loader,
        val_loader,
        test_loader,
        ligand_dim,
    ) = prepare_dataset_splits(
        DATASET_NAME,
        fold,
        METHOD,
        mol_feat,
        info_cols,
        property_set_ratio=property_set_ratio,
    )

    match MolFingerprint(mol_feat):
        case MolFingerprint.GRAPH:
            model_cls = GraphMoleculeBayesianSetRankModel
        case MolFingerprint.ALL:
            model_cls = AllMoleculeBayesianSetRankModel
        case _:
            model_cls = MoleculeBayesianSetRankModel

    args = [
        model_cls,
        run_name,
        train_loader,
        val_loader,
        test_loader,
        METHOD,
        fold,
    ]
    kwargs = dict(
        ligand_dim=ligand_dim,
        multi_batch=False,
        fisher_transform=False,
        n_bins=n_bins,
        deg=getattr(train_loader.dataset, "deg_histogram", None),
        **kwargs,
    )

    train_and_evaluate_pfn_model(*args, **kwargs)

    logger.info(f"{run_name} finished")


def main():
    torch.cuda.empty_cache()

    parser = argparse.ArgumentParser(
        description="Run PFN (MoleculeBayesianSetRankModel) training on ChEMBL."
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
        help="Weight of reconstruction on unmasked samples. (default: 0.0)",
    )
    parser.add_argument(
        "--property-set-ratio",
        type=float,
        default=0.5,
        help="Proportion of physiochemical property sets during training. (default: 0.5)",
    )
    parser.add_argument(
        "--n-bins",
        type=int,
        default=100,
        help="Number of bins in bin distribution. (default: 100)",
    )
    parser.add_argument(
        "--norm",
        type=str,
        default="minmax",
        help="Normalization mode either 'minmax' or 'zscore'. (default: 'minmax')",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default="1e-5",
        help="Initial learning rate. (default: 1e-5)",
    )
    valid_obj = ["nll", "crps", "wasserstein", "emd"]
    parser.add_argument(
        "--obj", type=str, default="nll", help=f"Training objective {valid_obj}"
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.05,
        help="Dropout probability. (default: 0.05)",
    )
    parser.add_argument(
        "--act", type=str, default="GELU", help="Activation function. (default: 'GELU')"
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=10_000,
        help="Maximum number of training epochs. (default: 10000)",
    )
    parser.add_argument(
        "--no-test",
        action="store_true",
        help="Do not apply the best model to the test set after optimization.",
    )
    parser.add_argument(
        "--min-lr",
        type=float,
        default=1e-6,
        help="Minimal learning rate at the final epoch. (default: 1e-6)",
    )

    args = parser.parse_args()

    mol_feat = args.mol_feat.lower()

    job_id = get_condor_job_id()
    run_name = f"chembl_{mol_feat}_{args.fold}_pfn_{job_id}"
    init_logging(run_name)
    logger = logging.getLogger(run_name)
    logger.info(f"CLI arguments: {' '.join(f'{k}={v}' for k, v in vars(args).items())}")

    objective = args.obj.lower()
    if objective not in valid_obj:
        raise ValueError(
            f"Invalid training objective '{objective}'. Must be one of {valid_obj}"
        )
    if objective == "wasserstein":
        objective = "emd"

    if args.norm not in ["minmax", "zscore"]:
        raise ValueError(
            f"Unknown normalization method: {args.norm}.Should be 'minmax' or 'zscore'"
        )

    if args.dropout < 0 or args.dropout > 1:
        raise ValueError(f"Invalid dropout probability {args.dropout}")
    set_random_seeds(args.seed)

    if not hasattr(nn, args.act):
        raise ValueError(f"Activation function '{args.act}' not found in `torch.nn`.")
    else:
        act = getattr(nn, args.act)

    if args.num_epochs < 1:
        raise ValueError(
            f"The number of training epochs ({args.num_epochs}) has to be greater than 1."
        )

    if args.min_lr < 0:
        raise ValueError(f"Minimum learning rate {args.min_lr:.2e} < 0.")

    run_split(
        run_name,
        mol_feat,
        args.fold,
        args.property_set_ratio,
        args.n_bins,
        unmasked_weight=args.unmasked_weight,
        normalization=args.norm,
        objective=objective,
        lr=args.lr,
        p_dropout=args.dropout,
        act=act,
        num_epochs=args.num_epochs,
        test=not args.no_test,
        min_lr=args.min_lr,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        for line in traceback.format_exc().split("\n"):
            logger.error(line)
        logger.error(e)
