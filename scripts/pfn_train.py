import argparse
import logging
from functools import partial
import traceback

import torch
from torch.utils.data import DataLoader

from dti.model import MoleculeBayesianSetRankModel
from dti.data.dataset import PropertySetDataset, ResettingBatchSampler
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
    method: Method,
    mol_feat: str,
    info_cols: list[str],
    property_set_ratio: float,
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
    mol_feat = MolFingerprint(mol_feat)
    data_kwargs = dict()
    val_dataset_cls = partial(
        PropertySetDataset,
        max_batch_datapoints=2048,
        max_set_size=2048,
        shuffle_within_target=False,
        mol_featurizer=mol_feat,
        info_cols=info_cols,
        property_columns=[],
        property_set_ratio=0.0,
        target=ACT,
    )

    val_dataset = val_dataset_cls(val_data)
    test_dataset = val_dataset_cls(test_data)
    train_dataset = PropertySetDataset(
        train_data,
        max_batch_datapoints=2048,
        max_set_size=1024,
        shuffle_within_target=False,
        property_set_ratio=property_set_ratio,
        mol_featurizer=mol_feat,
        info_cols=info_cols,
        target=ACT,
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

    assert len(train_dataset) > 0

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=ResettingBatchSampler(train_dataset, batch_size=1),
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_sampler=ResettingBatchSampler(val_dataset, batch_size=1),
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_sampler=ResettingBatchSampler(test_dataset, batch_size=1),
        shuffle=False,
        num_workers=0,
    )

    return (
        train_loader,
        val_loader,
        test_loader,
        mol_feat.dim,
    )


def run_split(
    run_name: str,
    mol_feat: str,
    fold: int,
    unmasked_weight: float,
    property_set_ratio: float,
    n_bins: int,
):
    num_epochs = 1_000
    info_cols = [INTRA_ASSAY_TEST, IDENT, COMPOUND, ASSAY]

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

    args = [
        MoleculeBayesianSetRankModel,
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
        num_epochs=num_epochs,
        patience_termination=30,
        patience_lr=10,
        fisher_transform=False,
        unmasked_weight=unmasked_weight,
        n_bins=n_bins,
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

    args = parser.parse_args()

    mol_feat = args.mol_feat.lower()

    job_id = get_condor_job_id()
    run_name = f"chembl_{mol_feat}_{args.fold}_pfn_{job_id}"
    init_logging(run_name)
    logger = logging.getLogger(run_name)
    logger.info(
        f"seed={args.seed} "
        f"fold={args.fold} "
        f"n_bins={args.n_bins} "
        f"prop-set-ratio={args.property_set_ratio} "
        f"unmasked-weight={args.unmasked_weight}"
    )

    set_random_seeds(args.seed)

    run_split(
        run_name,
        mol_feat,
        args.fold,
        args.unmasked_weight,
        args.property_set_ratio,
        args.n_bins,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        for line in traceback.format_exc().split("\n"):
            logger.error(line)
        logger.error(e)
