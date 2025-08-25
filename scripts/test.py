import logging
import sys
from functools import partial
from typing import Tuple, Callable

import torch
from torch import nn
from torch.utils.data import DataLoader
from scipy.stats import pearsonr

from dti.model import (
    CombinedModel,
    MolecularModel,
    PairCombinedModel,
    PairMolecularModel,
    SetRankModel,
    MoleculeSetRank,
)
from dti.data import (
    MultiSetActivityDataset,
    load_landrum,
    load_kinodata,
    load_nci,
    load_solubility,
    load_lipo,
    load_clearance,
    load_activities,
    load_split,
)
from dti.training import (
    AssayRankAccuracy,
    eval_with_batched_sets,
)
from dti.utils import (
    Method,
    init_logging,
    set_random_seeds,
    device,
)
from dti.constants import DATA, ASSAY, COMPOUND, OUTPUT


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

    model_cls = model_and_dataset(method, mol_only)
    return model_cls, data


def model_and_dataset(method: Method, mol_only: bool) -> Tuple[type, type, type]:
    match method:
        case Method.PAIRS | Method.ALLPAIRS if mol_only:
            return PairMolecularModel
        case Method.PAIRS | Method.ALLPAIRS:
            return PairCombinedModel
        case Method.IC50CORR | Method.HODGE | Method.IC50 if mol_only:
            return MolecularModel
        case Method.IC50CORR | Method.HODGE | Method.IC50:
            return CombinedModel
        case Method.IC50ALLSETS | Method.ALLSETS | Method.IC50SETS | Method.SETS if (
            mol_only
        ):
            return MoleculeSetRank
        case Method.IC50SETS | Method.SETS | Method.IC50ALLSETS | Method.ALLSETS:
            return SetRankModel
        case _:
            logger.error(f"No model and dataset configuration for method: {method}")
            sys.exit(1)


def test_batch(method: Method, default: int) -> int:
    return default if method.on_pairs else 1


def run_split(
    run_name: str,
    test_run_name: str,
    method: Method,
    dataset_name: str,
    fold: int,
    seed: int,
):
    batch_size = 512
    info_cols = [COMPOUND, ASSAY]
    data_dir = DATA / "processed" / dataset_name
    tgt_name = "scaled_ic50"

    model_cls, load_data = setup(method, dataset_name)
    msa = partial(MultiSetActivityDataset, max_set_size=1000)
    shuffled_msa = partial(msa, inter_assay=True)

    data = load_data()

    model = model_cls(
        ligand_input_size=2048,
        embedding_size=512,
        protein_input_size=1280,
        cosine_agg=True,
    ).to(device)
    model.load_state_dict(
        torch.load(
            OUTPUT / run_name / f"model{fold}.pt",
            weights_only=True,
            map_location=device,
        )
    )
    model.eval()

    _, _, test_data = load_split(fold, data_dir, tgt_name)

    for name, val_dataset_cls in [("shuffled", shuffled_msa), ("assays", msa)]:
        test_dataset = val_dataset_cls(test_data, target=tgt_name, info_cols=info_cols)

        test_loader = DataLoader(
            test_dataset,
            batch_size=test_batch(method, batch_size),
            shuffle=False,
            num_workers=0,
        )

        rstat = pearsonr
        assay_rank = AssayRankAccuracy(data, method.on_pairs, rank_statistic=rstat)

        _, test_rank_corr = eval_with_batched_sets(
            model,
            test_loader,
            criterion=nn.L1Loss(),
            rank_corr_fn=assay_rank,
            fisher_transform=True,
        )

        logger.info(
            f"[{test_run_name}] Epoch: 0 Fold: {fold} Test Rank Corr: {test_rank_corr:.4f} {name}"
        )

    logger.info(f"{test_run_name} finished")


def main():
    torch.cuda.empty_cache()
    ident = sys.argv[1]
    seed = int(sys.argv[2])
    dataset_name = sys.argv[3].lower()
    method = Method.from_string(sys.argv[4])
    fold = int(sys.argv[5])

    run_name = f"{dataset_name}_{fold}_{repr(method)}_{ident}"
    test_run_name = "test_" + run_name
    init_logging(test_run_name)

    logger = logging.getLogger(test_run_name)
    logger.info(f"seed={seed} method={repr(method)} dataset={dataset_name} fold={fold}")
    set_random_seeds(seed)

    run_split(run_name, test_run_name, method, dataset_name, fold, seed)


if __name__ == "__main__":
    main()
