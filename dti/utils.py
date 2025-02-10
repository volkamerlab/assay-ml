from typing import Union
import time
import logging

import torch
import numpy as np
import random

from .training import model_epoch

from .constants import OUTPUT

device = lambda: "cuda" if torch.cuda.is_available() else "cpu"


def set_random_seeds(seed: int):
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    random.seed(seed)
    np.random.seed(seed)


def init_logging(run_name: Union[str, None] = str(time.time())):
    (OUTPUT / run_name).mkdir(exist_ok=True, parents=True)
    log_file = OUTPUT / run_name / "output.log"
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(log_file)

    console_handler.setLevel(logging.INFO)
    file_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    logger.info(f"logging run {run_name} to {log_file}")


def write_header(run_name: str):
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


def write_info(run_name: str, fields: list):
    """Write optimization data to a CSV file."""
    (OUTPUT / run_name).mkdir(exist_ok=True, parents=True)
    with open(OUTPUT / run_name / "optimization.csv", "a") as f:
        f.write(",".join(map(str, fields)) + "\n")


def train_and_evaluate_model(
    model_cls,
    run_name,
    train_loader,
    val_loader,
    test_loader,
    logger,
    target_name,
    index,
    **kwargs,
):
    """Train and evaluate the model."""
    logger.info(f"training model for target: {target_name}")
    opts = (
        dict(
            protein_dim=1280,
            ligand_dim=2048,
            embedding_size=256,
            num_epochs=500,
            cosine_agg=False,
        )
        | kwargs
    )

    model = model_cls(
        opts["protein_dim"],
        opts["ligand_dim"],
        opts["embedding_size"],
        cosine_agg=opts["cosine_agg"],
    ).to(device())
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    best_corr = 0
    for epoch in range(opts["num_epochs"]):
        train_loss, train_rank_corr = model_epoch(model, train_loader, optimizer)
        val_loss, val_rank_corr = model_epoch(model, val_loader)

        logger.info(
            " ".join(
                [
                    f"[{run_name}]",
                    f"epoch={epoch + 1}/{opts['num_epochs']}",
                    f"train_loss={train_loss:.4f}",
                    f"val_loss={val_loss:.4f}",
                    f"train_rank_corr={train_rank_corr:.4f}",
                    f"val_rank_corr={val_rank_corr:.4f}",
                ]
            )
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
            logger.info(f"[{run_name}] updating test set predictions")
            best_corr = val_rank_corr
            torch.save(model.state_dict(), OUTPUT / run_name / f"model{index}.pt")
            test_loss, test_rank_corr = model_epoch(
                model,
                test_loader,
                prediction_file=OUTPUT
                / run_name
                / f"{target_name}_index{index}_preds.csv",
            )
            logger.info(
                f"[{run_name}] epoch={epoch + 1}/{opts['num_epochs']} test_loss={test_loss:.4f} test_rank_corr={test_rank_corr:.4f}"
            )
