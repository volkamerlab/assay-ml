from functools import partial
import logging
import os
from pathlib import Path
import sys
import traceback
from typing import Any, Callable, Dict, Tuple
import uuid

import numpy as np
import optuna
import torch
from torch import nn
from torch.utils.data import DataLoader
import yaml

from dti.constants import ASSAY, COMPOUND, DATA, DOC, HODGE
from dti.data import (
    ActivityDataset,
    MultiSetActivityDataset,
    PairDataset,
    SetActivityDataset,
    load_kinodata,
    load_landrum,
    load_split,
    prepare_datasets,
)
from dti.featurization import MolFingerprint
from dti.model import (
    CombinedModel,
    MolecularModel,
    MoleculeSetRank,
    PairCombinedModel,
    PairMolecularModel,
    SetRankModel,
)
from dti.training import (
    FisherPearsonLoss,
    eval_with_batched_sets,
    train_and_evaluate_model,
    train_epoch,
    train_with_batched_sets,
)
from dti.utils import (
    Method,
    device,
    init_logging,
    save_code_snapshot,
    set_random_seeds,
)

# Optuna logging setup
optuna.logging.set_verbosity(optuna.logging.INFO)
optuna_logger = optuna.logging.get_logger("optuna")
logger = logging.getLogger(__name__)


class YAMLRegistry:
    """Thread-safe YAML tracker mapping Optuna trials to parameters and run directories."""

    def __init__(self, filepath: Path):
        self.filepath = filepath
        if not self.filepath.exists():
            self.filepath.parent.mkdir(parents=True, exist_ok=True)
            with open(self.filepath, "w") as f:
                yaml.dump({"trials": []}, f)

    def log_trial(self, trial_info: Dict[str, Any]):
        try:
            with open(self.filepath, "r") as f:
                data = yaml.safe_load(f) or {"trials": []}

            data["trials"].append(trial_info)

            with open(self.filepath, "w") as f:
                yaml.dump(data, f, default_flow_style=False)
        except Exception as e:
            logger.error(f"Failed to write trial record to YAML registry: {e}")


def setup(method: Method, dataset: str) -> Tuple[type, type, type, Callable]:
    match dataset.lower():
        case "kinodata":
            data, mol_only = load_kinodata, False
        case "landrum":
            data, mol_only = load_landrum, False
        case "large_landrum":
            data_path = DATA / "raw" / "landrum_large.csv"
            data, mol_only = partial(load_landrum, data_path), False
        case "omnivore":
            data, mol_only = (
                partial(load_landrum, DATA / "raw" / "omnivore.csv"),
                False,
            )
        case _:
            logger.error(f"Unknown dataset: {dataset}")
            sys.exit(1)

    model_cls, dataset_cls, val_dataset_cls = model_and_dataset(method, mol_only)
    return model_cls, dataset_cls, val_dataset_cls, data


def model_and_dataset(method: Method, mol_only: bool) -> Tuple[type, type, type]:
    msa = partial(MultiSetActivityDataset, max_set_size=4096)
    shuffled_multiset = partial(msa, inter_assay=True)
    match method:
        case Method.IC50:
            return CombinedModel, ActivityDataset, msa
        case Method.SETS:
            return SetRankModel, msa, msa
        case Method.ALLSETS:
            return SetRankModel, shuffled_multiset, msa
        case _:
            logger.error(f"No model and dataset configuration for method: {method}")
            sys.exit(1)


def suggest_hyperparameters(trial: optuna.Trial, model_cls: type) -> Dict[str, Any]:
    """Generates a comparable search space between CombinedModel and SetRankModel."""
    params = {
        # Optimization hyperparams
        "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        "p_dropout": trial.suggest_float("p_dropout", 0.0, 0.4, step=0.05),
        # Architecture hyperparams
        "hidden_channels": trial.suggest_categorical(
            "hidden_channels", [128, 256, 512]
        ),
        "num_protein_layers": trial.suggest_int("num_protein_layers", 1, 4),
        "num_ligand_layers": trial.suggest_int("num_ligand_layers", 1, 4),
    }

    # Model specific search space tuning
    if issubclass(model_cls, SetRankModel):
        params.update(
            {
                "num_blocks": trial.suggest_int("num_blocks", 1, 6),
                "num_heads": trial.suggest_categorical("num_heads", [2, 4]),
            }
        )
    elif issubclass(model_cls, CombinedModel):
        params.update(
            {
                "num_output_layers": trial.suggest_int("num_output_layers", 1, 4),
            }
        )

    return params


def objective(
    trial: optuna.Trial,
    base_run_name: str,
    mol_feat_name: str,
    method: Method,
    dataset_name: str,
    fold: int,
    seed: int,
    registry: YAMLRegistry,
) -> float:
    set_random_seeds(seed + trial.number)

    # 1. Prepare data loaders
    batch_size = 1024
    num_epochs = 2000
    info_cols = [COMPOUND, ASSAY, DOC]
    data_dir = DATA / "processed" / dataset_name
    train_tgt = tgt_name = "scaled_ic50"

    model_cls, dataset_cls, val_dataset_cls, load_data = setup(method, dataset_name)
    data = load_data()

    if not (data_dir / f"{fold}").exists():
        prepare_datasets(
            data,
            data_dir,
            5,
            random_valset=False,
            aggregate=True,
            split_col=DOC,
        )

    train_data, val_data, test_data = load_split(
        fold, data_dir, tgt_name, inter_assay_weight=None
    )
    mol_feat = MolFingerprint(mol_feat_name)
    data_kwargs = dict(mol_featurizer=mol_feat, info_cols=info_cols)

    train_dataset = dataset_cls(train_data, target=train_tgt, **data_kwargs)
    val_dataset = val_dataset_cls(val_data, target=tgt_name, **data_kwargs)
    test_dataset = val_dataset_cls(test_data, target=tgt_name, **data_kwargs)

    train_loader = DataLoader(
        train_dataset,
        batch_size=1 if method.on_sets else batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    # 2. Get trial parameters
    hparams = suggest_hyperparameters(trial, model_cls)

    # 3. Formulate output directory run identity
    trial_uuid = uuid.uuid4().hex[:6]
    trial_run_name = f"{base_run_name}_trial_{trial.number}_{trial_uuid}"
    output_dir = Path("outputs") / trial_run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # 4. Construct execution settings
    settings = dict(
        ligand_dim=mol_feat.dim,
        multi_batch=method in [Method.SETS, Method.IC50CORR],
        batch_size=batch_size,
        num_epochs=num_epochs,
        cosine_agg=True,
        patience_termination=10,  # Tightened for faster optimization iterations
        patience_lr=5,
        eval_fn=eval_with_batched_sets,
        **hparams,
    )

    if method.on_sets:
        settings |= dict(
            criterion=FisherPearsonLoss().to(device),
            train_fn=train_with_batched_sets,
            eval_criterion=True,
        )
    else:
        settings |= dict(
            criterion=nn.MSELoss(), train_fn=train_epoch, eval_criterion=False
        )

    settings["model_weights"] = os.environ.get("ASSAY_ML_MODEL_WEIGHTS", None)

    logger.info(f"--- Starting Trial {trial.number} [{trial_run_name}] ---")
    logger.info(f"Parameters: {hparams}")
    settings |= hparams

    try:
        # Run training loop and return validation score
        val_loss = train_and_evaluate_model(
            model_cls,
            trial_run_name,
            train_loader,
            val_loader,
            test_loader,
            method,
            fold,
            trial=trial,
            test=False,
            **settings,
        )

        # Log trial setting registry
        registry.log_trial(
            {
                "trial_number": trial.number,
                "trial_id": trial_uuid,
                "output_dir": str(output_dir.resolve()),
                "run_name": trial_run_name,
                "val_loss": float(val_loss),
                "params": hparams,
            }
        )

        return val_loss

    except Exception as e:
        logger.error(f"Trial {trial.number} failed with error: {e}")
        logger.error(traceback.format_exc())
        raise e


def main():
    torch.cuda.empty_cache()

    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    dataset_name = sys.argv[2].lower() if len(sys.argv) > 2 else "kinodata"
    method = Method.from_string(sys.argv[3]) if len(sys.argv) > 3 else Method.IC50
    fold = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    mol_feat = sys.argv[5].lower() if len(sys.argv) > 5 else "morgan"
    n_trials = int(sys.argv[6]) if len(sys.argv) > 6 else 30

    base_run_name = os.environ.get(
        "ASSAY_ML_IDENT",
        "_".join(map(str, [dataset_name, mol_feat, fold, repr(method)])),
    )

    init_logging(base_run_name)
    save_code_snapshot(base_run_name)

    registry = YAMLRegistry(Path("hpo_registry.yaml"))

    db_path = Path("optuna_hpo.db").resolve()
    storage_url = f"sqlite:///{db_path}"

    study = optuna.create_study(
        study_name=f"hpo_{base_run_name}",
        storage=storage_url,
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
        load_if_exists=True,  # Allows resuming studies or concurrent workers
    )

    optuna_logger.info(
        f"Starting HPO Study '{study.study_name}' on storage {storage_url} ({n_trials} trials)"
    )

    obj_fn = partial(
        objective,
        base_run_name=base_run_name,
        mol_feat_name=mol_feat,
        method=method,
        dataset_name=dataset_name,
        fold=fold,
        seed=seed,
        registry=registry,
    )

    study.optimize(obj_fn, n_trials=n_trials)

    optuna_logger.info("--- Optimization Complete ---")
    optuna_logger.info(f"Best Trial Val Loss: {study.best_value}")
    optuna_logger.info(f"Best Parameters: {study.best_params}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        for line in traceback.format_exc().split("\n"):
            logger.error(line)
        logger.error(e)
