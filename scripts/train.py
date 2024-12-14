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
    FingerprintFactory,
    split_kinodata,
)
from dti.utils import ACT, DEVICE, DATA, OUTPUT, init_logging
from dti.model import CombinedModel
from dti.training import train_model, val_model
from dti.hodge_ranking import parallel_hodge_rank


def write_info(run_name: str, fields: list):
    with open(OUTPUT / run_name / "optimization.csv", "a") as f:
        f.write(",".join(map(str, fields)) + "\n")


if __name__ == "__main__":
    run_name = uuid.uuid4().hex[:8]
    init_logging(run_name)
    logger = logging.getLogger("main")
    write_info(
        run_name,
        ["model_type", "index", "epoch", "train_loss", "val_loss", "mean_rank_corr"],
    )

    fp_gen = FingerprintFactory()

    k = 10
    target_dir = DATA / run_name
    split_dir = split_kinodata(target_dir, k=k)
    for index in range(0, k, 2):
        test_idcs = [index, (index + 1) % k]
        val_idx = (index + 2) % k
        train_idcs = [i for i in range(k) if i not in test_idcs and i != val_idx]
        assert (
            len(set(test_idcs) & set(train_idcs)) == 0 and val_idx not in test_idcs
        ), (train_idcs, test_idcs, val_idx)

        val_data = pd.read_csv(split_dir / f"{val_idx}.csv", index_col=0)
        train_data = pd.concat(
            [pd.read_csv(split_dir / f"{i}.csv", index_col=0) for i in train_idcs]
        )
        test_data = pd.concat(
            [pd.read_csv(split_dir / f"{i}.csv", index_col=0) for i in test_idcs]
        )

        scaler = StandardScaler()
        tgt_name = "scaled_ic50"
        train_data[tgt_name] = scaler.fit_transform(
            train_data[ACT].values.reshape(-1, 1)
        )
        test_data[tgt_name] = scaler.transform(test_data[ACT].values.reshape(-1, 1))
        val_data[tgt_name] = scaler.transform(val_data[ACT].values.reshape(-1, 1))

        info_cols = ["activities.activity_id", "assay_id"]
        train_dataset = ActivityDataset(
            train_data,
            fp_gen=fp_gen,
            target=tgt_name,
            info_cols=info_cols,
        )

        test_dataset = ActivityDataset(
            test_data,
            fp_gen=fp_gen,
            target=tgt_name,
            info_cols=info_cols,
        )
        val_dataset = ActivityDataset(
            val_data,
            fp_gen=fp_gen,
            target=tgt_name,
            info_cols=info_cols,
        )

        num_epochs = 100
        batch_size = 256

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        protein_dim = 1280
        ligand_dim = 2048
        embedding_size = 256

        model = CombinedModel(protein_dim, ligand_dim, embedding_size).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        best_corr = 0
        for epoch in range(num_epochs):
            train_loss = train_model(model, train_loader, optimizer)
            val_loss, mean_rank_corr = val_model(model, val_loader)
            if mean_rank_corr > best_corr:
                best_corr = mean_rank_corr
                val_loss, mean_rank_corr = val_model(
                    model,
                    test_loader,
                    prediction_file=target_dir / f"index{index}_epoch{epoch}_preds.csv",
                )
                f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, Mean Rank Correlation = {mean_rank_corr:.4f}"
            else:
                logger.info(
                    f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, Mean Rank Correlation = {mean_rank_corr:.4f}"
                )
            write_info(
                run_name, ["ic50", index, epoch, train_loss, val_loss, mean_rank_corr]
            )

        hodge_file = split_dir / f"train_hodge_{index}.csv"
        if not hodge_file.exists():
            hodge_df = parallel_hodge_rank(train_data)
            hodge_kd = train_data.merge(
                hodge_df,
                on=["compound_structures.canonical_smiles", "UniprotID"],
                how="inner",
            )
            hodge_kd.to_csv(hodge_file)
        else:
            hodge_kd = pd.read_csv(hodge_file, index_col=0)

        tgt_name = "hodge_score"
        train_dataset = ActivityDataset(
            hodge_kd,
            fp_gen=fp_gen,
            target=tgt_name,
            info_cols=info_cols,
        )
        test_dataset = ActivityDataset(
            test_data,
            fp_gen=fp_gen,
            target=tgt_name,
            info_cols=info_cols,
        )
        val_dataset = ActivityDataset(
            val_data,
            fp_gen=fp_gen,
            target=tgt_name,
            info_cols=info_cols,
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = CombinedModel(protein_dim, ligand_dim, embedding_size).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        best_corr = 0
        for epoch in range(num_epochs):
            train_loss = train_model(model, train_loader, optimizer)
            val_loss, mean_rank_corr = val_model(model, val_loader)
            if mean_rank_corr > best_corr:
                best_corr = mean_rank_corr
                val_loss, mean_rank_corr = val_model(
                    model,
                    test_loader,
                    prediction_file=target_dir / f"index{index}_epoch{epoch}_preds.csv",
                )
                f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, Mean Rank Correlation = {mean_rank_corr:.4f}"
            else:
                logger.info(
                    f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, Mean Rank Correlation = {mean_rank_corr:.4f}"
                )
            write_info(
                run_name, ["ic50", index, epoch, train_loss, val_loss, mean_rank_corr]
            )

