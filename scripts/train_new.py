import logging
import time
import uuid
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

from dti import utils
from dti.data import (
    ActivityDataset,
    process_kinodata_default_dti,
    split_kinodata,
)
from dti.utils import ACT, DEVICE, DATA, OUTPUT, init_logging
from dti.model import CombinedModel
from dti.training import model_epoch
from dti.hodge_ranking import parallel_hodge_rank

def write_info(run_name: str, fields: list):
    """Write optimization data to a CSV file."""
    (OUTPUT / run_name).mkdir(exist_ok=True, parents=True)
    with open(OUTPUT / run_name / "optimization.csv", "a") as f:
        f.write(",".join(map(str, fields)) + "\n")

def normalize_activity(data: pd.DataFrame, target_col: str, scaler: StandardScaler):
    """Normalize activity data to a standard normal distribution."""
    data[target_col] = scaler.transform(data[ACT].values.reshape(-1, 1))
    return data

def prepare_datasets(data_dir, split_dir, tgt_name, k, logger):
    """Prepare train, validation, and test datasets."""
    for index in range(0, k, 2):
        logger.info(f"Preparing datasets for split {index}")

        test_idcs = [index, (index + 1) % k]
        val_idx = (index + 2) % k
        train_idcs = [i for i in range(k) if i not in test_idcs and i != val_idx]

        val_data = pd.read_csv(split_dir / f"{val_idx}.csv", index_col=0)
        train_data = pd.concat(
            [pd.read_csv(split_dir / f"{i}.csv", index_col=0) for i in train_idcs]
        )
        test_data = pd.concat(
            [pd.read_csv(split_dir / f"{i}.csv", index_col=0) for i in test_idcs]
        )

        logger.info("Normalizing activity data")
        scaler = StandardScaler()
        train_data[tgt_name] = scaler.fit_transform(
            train_data[ACT].values.reshape(-1, 1)
        )
        test_data = normalize_activity(test_data, tgt_name, scaler)
        val_data = normalize_activity(val_data, tgt_name, scaler)

        yield train_data, val_data, test_data, index

def train_and_evaluate_model(run_name, train_loader, val_loader, test_loader, logger, target_name, index):
    """Train and evaluate the model."""
    logger.info(f"Training model for target: {target_name}")
    protein_dim = 1280
    ligand_dim = 2048
    embedding_size = 256
    num_epochs = 100
    batch_size = 256

    model = CombinedModel(protein_dim, ligand_dim, embedding_size).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_corr = 0
    for epoch in range(num_epochs):
        train_loss, train_rank_corr = model_epoch(model, train_loader, optimizer)
        val_loss, val_rank_corr = model_epoch(model, val_loader)

        logger.info(
            f"[{target_name}] epoch={epoch + 1} train_loss={train_loss:.4f} val_loss={val_loss:.4f} train_rank_corr={train_rank_corr:.4f} val_rank_corr={val_rank_corr:.4f}"
        )
        write_info(
            run_name,
            [
                target_name,
                index,
                epoch,
                train_loss,
                val_loss,
                train_rank_corr,
                val_rank_corr,
            ],
        )

        if val_rank_corr > best_corr:
            logger.info("[rank] Updating best rank corr.")
            best_corr = val_rank_corr
            test_loss, test_rank_corr = model_epoch(
                model,
                test_loader,
                prediction_file=OUTPUT / run_name / f"{target_name}_index{index}_preds.csv",
            )
            logger.info(
                f"[{target_name}] test_loss={test_loss:.4f} test_rank_corr={test_rank_corr:.4f}"
            )

def main():
    run_name = uuid.uuid4().hex[:8]
    init_logging(run_name)
    logger = logging.getLogger("main")
    
    write_info(
        run_name,
        [
            "model_type",
            "index",
            "epoch",
            "train_loss",
            "val_loss",
            "train_rank_corr",
            "val_rank_corr",
        ],
    )

    data_dir = DATA / "processed"
    split_dir = split_kinodata(data_dir, k=10)
    tgt_name = "scaled_ic50"

    for train_data, val_data, test_data, index in prepare_datasets(data_dir, split_dir, tgt_name, 10, logger):
        logger.info("Creating datasets")
        info_cols = ["activities.activity_id", "assay_id"]
        train_dataset = ActivityDataset(train_data, target=tgt_name, info_cols=info_cols)
        val_dataset = ActivityDataset(val_data, target=tgt_name, info_cols=info_cols)
        test_dataset = ActivityDataset(test_data, target=tgt_name, info_cols=info_cols)

        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

        train_and_evaluate_model(
            run_name, train_loader, val_loader, test_loader, logger, "ic50", index
        )

        logger.info("Performing Hodge ranking")
        hodge_file = data_dir / f"train_hodge_{index}.csv"
        if not hodge_file.exists():
            hodge_df = parallel_hodge_rank(train_data)
            hodge_kd = train_data.merge(
                hodge_df,
                on=["compound_structures.canonical_smiles", "UniprotID"],
                how="inner",
            )
            hodge_kd.to_csv(hodge_file)
        else:
            logger.info("Found cached Hodge ranking data")
            hodge_kd = pd.read_csv(hodge_file, index_col=0)

        train_dataset = ActivityDataset(hodge_kd, target="hodge_score")
        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

        train_and_evaluate_model(
            run_name, train_loader, val_loader, test_loader, logger, "rank", index
        )

    logger.info("Pipeline completed")

if __name__ == "__main__":
    main()

