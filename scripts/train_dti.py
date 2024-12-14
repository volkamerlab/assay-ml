import logging
import time
import uuid

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from dti import utils
from pathlib import Path
from dti.data import ActivityDataset, process_kinodata_default_dti, FingerprintFactory
from dti.utils import ACT, DEVICE, DATA, OUTPUT, init_logging
from dti.model import CombinedModel
from dti.training import train_model, val_model
from dti.hodge_ranking import parallel_hodge_rank

def write_info(run_name: str, fields: list):
    with open(OUTPUT / (run_name + ".csv"), "w") as f:
        f.write(",".join(map(str, fields)) + "\n")

if __name__ == "__main__":
    run_name = uuid.uuid4().hex[:8]
    init_logging(run_name)
    logger = logging.getLogger("main")

    fp_gen = FingerprintFactory()

    target_dir = process_kinodata_default_dti()
    for index in range(5):
        split_dir = target_dir / str(index)
        train_file = split_dir / "train.csv"
        test_file = split_dir / "test.csv"
        train_data, val_data = pd.read_csv(train_file, index_col=0), pd.read_csv(
            test_file, index_col=0
        )

        tgt_name = "scaled_ic50"
        train_dataset = ActivityDataset(
            train_data, fp_gen=fp_gen, target=tgt_name, keep_cols=["assay_id"]
        )
        val_dataset = ActivityDataset(
            val_data, fp_gen=fp_gen, target=tgt_name, keep_cols=["assay_id"]
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        protein_input_size = 1280
        ligand_input_size = 2048
        embedding_size = 256
        batch_size = 256

        model = CombinedModel(protein_input_size, ligand_input_size, embedding_size).to(
            DEVICE
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        for epoch in range(30):
            train_loss = train_model(model, train_loader, optimizer)
            val_loss, mean_rank_corr = val_model(model, val_loader)
            logger.info(
                f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, Mean Rank Correlation = {mean_rank_corr:.4f}"
            )
            write_info(run_name, ["ic50", index, epoch, train_loss, val_loss, mean_rank_corr])

        hodge_file = split_dir / "train_hodge.csv"
        if not hodge_file.exists():
            hodge_df = parallel_hodge_rank(train_data)
            hodge_kd = train_data.merge(
                hodge_df,
                on=["compound_structures.canonical_smiles", "UniprotID"],
                how="inner",
            )
            hodge_kd.to_csv(hodge_file)
        else:
            pd.read_csv(hodge_file, index_col=0)

        train_dataset = ActivityDataset(
            hodge_kd, fp_gen=fp_gen, target="hodge_score", keep_cols=["assay_id"]
        )
        val_dataset = ActivityDataset(
            val_dataset, fp_gen=fp_gen, target=tgt_name, keep_cols=["assay_id"]
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = CombinedModel(protein_input_size, ligand_input_size, embedding_size).to(
            device
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        for epoch in range(30):
            train_loss = train_model(model, train_loader, optimizer)
            val_loss, mean_rank_corr = val_model(model, val_loader)
            logger.info(
                f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, Mean Rank Correlation = {mean_rank_corr:.4f}"
            )
            write_info(run_name, ["hodge", index, epoch, train_loss, val_loss, mean_rank_corr])
