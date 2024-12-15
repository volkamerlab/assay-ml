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
from dti.training import model_epoch
from dti.hodge_ranking import parallel_hodge_rank


def write_info(run_name: str, fields: list):
    (OUTPUT / run_name).mkdir(exist_ok=True, parents=True)
    with open(OUTPUT / run_name / "optimization.csv", "a") as f:
        f.write(",".join(map(str, fields)) + "\n")


if __name__ == "__main__":
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

    fp_gen = FingerprintFactory()

    k = 10
    data_dir = DATA / "processed"
    split_dir = split_kinodata(data_dir, k=k)
    for index in range(0, k, 2):
        logger.info(f"fit split {index}")
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

        logger.info("normalize ic50 data")
        scaler = StandardScaler()
        tgt_name = "scaled_ic50"
        train_data[tgt_name] = scaler.fit_transform(
            train_data[ACT].values.reshape(-1, 1)
        )
        test_data[tgt_name] = scaler.transform(test_data[ACT].values.reshape(-1, 1))
        val_data[tgt_name] = scaler.transform(val_data[ACT].values.reshape(-1, 1))

        info_cols = ["activities.activity_id", "assay_id"]
        logger.info("create training set")
        train_dataset = ActivityDataset(
            train_data,
            fp_gen=fp_gen,
            target=tgt_name,
            info_cols=info_cols,
        )
        logger.info("create test set")
        test_dataset = ActivityDataset(
            test_data,
            fp_gen=fp_gen,
            target=tgt_name,
            info_cols=info_cols,
        )
        logger.info("create validation set")
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

        logger.info("train pIC50 model")
        model = CombinedModel(protein_dim, ligand_dim, embedding_size).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        best_corr = 0
        for epoch in range(num_epochs):
            train_loss, train_rank_corr = model_epoch(model, train_loader, optimizer)
            val_loss, val_rank_corr = model_epoch(model, val_loader)
            logger.info(
                f"[ic50] epoch={epoch + 1} train_loss={train_loss:.4f} val_loss={val_loss:.4f} train_rank_corr={train_rank_corr:.4f} val_rank_corr={val_rank_corr:.4f}"
            )
            write_info(
                run_name,
                [
                    "ic50",
                    index,
                    epoch,
                    train_loss,
                    val_loss,
                    train_rank_corr,
                    val_rank_corr,
                ],
            )
            if val_rank_corr > best_corr:
                best_corr = val_rank_corr
                test_loss, test_rank_corr = model_epoch(
                    model,
                    test_loader,
                    prediction_file=OUTPUT / run_name / f"pIC50_index{index}_preds.csv",
                )
                logger.info(
                    f"[ic50] test_loss={test_loss:.4f} test_rank_corr={val_rank_corr:.4f}"
                )

        logger.info("hodge ranking")
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
            logger.info("found cached Hodge ranking data")
            hodge_kd = pd.read_csv(hodge_file, index_col=0)

        logger.info("create training set")
        train_dataset = ActivityDataset(
            hodge_kd,
            fp_gen=fp_gen,
            target="hodge_score",
            info_cols=info_cols,
        )
        logger.info("create test set")
        test_dataset = ActivityDataset(
            test_data,
            fp_gen=fp_gen,
            target=tgt_name,
            info_cols=info_cols,
        )
        logger.info("create validation set")
        val_dataset = ActivityDataset(
            val_data,
            fp_gen=fp_gen,
            target=tgt_name,
            info_cols=info_cols,
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        logger.info("train rank model")
        model = CombinedModel(protein_dim, ligand_dim, embedding_size).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        best_corr = 0
        for epoch in range(num_epochs):
            train_loss, train_rank_corr = model_epoch(model, train_loader, optimizer)
            val_loss, val_rank_corr = model_epoch(model, val_loader)
            logger.info(
                f"[rank] epoch={epoch + 1} train_loss={train_loss:.4f} val_loss={val_loss:.4f} train_rank_corr={train_rank_corr:.4f} val_rank_corr={val_rank_corr:.4f}"
            )
            write_info(
                run_name,
                [
                    "rank",
                    index,
                    epoch,
                    train_loss,
                    val_loss,
                    train_rank_corr,
                    val_rank_corr,
                ],
            )
            if val_rank_corr > best_corr:
                best_corr = val_rank_corr
                test_loss, test_rank_corr = model_epoch(
                    model,
                    test_loader,
                    prediction_file=OUTPUT / run_name / f"rank_index{index}_preds.csv",
                )
                logger.info(
                    f"[rank] test_loss={test_loss:.4f} test_rank_corr={val_rank_corr:.4f}"
                )
        logger.info("done")
