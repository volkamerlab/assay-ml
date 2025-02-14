from typing import Type, Any, Dict

import tqdm
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from scipy.stats import kendalltau
from scipy.special import binom

import logging

from .constants import ASSAY
from .utils import write_info, device

logger = logging.getLogger(__name__)


def rank_corr_pairs(prediction_data: pd.DataFrame) -> float:
    # misclassification rate
    return (
        np.sign(prediction_data["prediction"]) == np.sign(prediction_data["target"])
    ).mean()


def rank_corr(prediction_data: pd.DataFrame) -> float:
    overall_tau = 0
    total_weight = 0
    for _, group in prediction_data.groupby(ASSAY):
        if group["target"].nunique() == 1 or group["prediction"].nunique() == 1:
            continue
        assay_weight = binom(len(group), 2)
        tau = kendalltau(
            group["prediction"], group["target"], nan_policy="raise", variant="c"
        ).statistic
        if np.isnan(tau):
            continue
        total_weight += assay_weight
        overall_tau += assay_weight * tau
    overall_tau /= total_weight

    return overall_tau


def model_epoch(
    model,
    loader,
    optimizer=None,
    criterion=None,
    prediction_file=None,
    rank_corr_fn=rank_corr_pairs,
):
    if optimizer is not None:
        logger.debug("train model")
        criterion = nn.MSELoss() if criterion is None else criterion
        model.train()
    else:
        logger.debug("evaluate model")
        criterion = nn.L1Loss() if criterion is None else criterion
        model.eval()
    torch.set_grad_enabled(optimizer is not None)

    total_loss = 0
    all_preds = list()
    all_labels = list()
    all_info = list()
    for protein_features, ligand_features, labels, info in tqdm.tqdm(loader):
        protein_features, ligand_features, labels = (
            protein_features.to(device),
            ligand_features.to(device),
            labels.to(device),
        )
        if optimizer is not None:
            optimizer.zero_grad()
        predictions = model(protein_features, ligand_features).squeeze()
        loss = criterion(predictions, labels)
        if optimizer is not None:
            loss.backward()
            optimizer.step()
        total_loss += loss.item()

        all_preds.extend(list(predictions.detach().cpu().numpy().flatten()))
        all_labels.extend(list(labels.detach().cpu().numpy().flatten()))
        all_info.append(info.detach().cpu().numpy())

    torch.set_grad_enabled(True)

    content = {
        "prediction": all_preds,
        "target": all_labels,
    }
    all_info = np.concat(all_info)
    for i, col in enumerate(loader.dataset.info_cols):
        content[col] = list(all_info[:, i].flatten())
    prediction_data = pd.DataFrame(content)
    if prediction_file is not None:
        logger.info(f"writing predictions to {prediction_file}")
        prediction_data.to_csv(prediction_file)
    mean_rank_corr = rank_corr_fn(prediction_data)

    total_loss /= len(loader)

    return total_loss, mean_rank_corr


def train_and_evaluate_model(
    model_cls: Type[nn.Module],
    run_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    logger: Any,
    target_name: str,
    index: int,
    **kwargs: Dict[str, Any],
) -> None:
    """Train and evaluate the model with learning rate adjustment and early stopping."""
    logger.info(f"training model for target: {target_name}")
    opts: Dict[str, Any] = (
        dict(
            protein_dim=1280,
            ligand_dim=2048,
            embedding_size=512,
            num_epochs=500,
            patience=100,
            cosine_agg=False,
            rank_corr_fn=rank_corr_pairs,
        )
        | kwargs
    )

    model: nn.Module = model_cls(
        ligand_input_size=opts["ligand_dim"],
        embedding_size=opts["embedding_size"],
        protein_input_size=opts["protein_dim"],
        cosine_agg=opts["cosine_agg"],
    ).to(device)
    optimizer: torch.optim.Optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, verbose=True
    )

    best_corr: float = 0.0
    epochs_without_improvement: int = 0
    for epoch in range(opts["num_epochs"]):
        train_loss: float
        train_rank_corr: float
        train_loss, train_rank_corr = model_epoch(
            model, train_loader, optimizer, rank_corr_fn=rank_corr_pairs
        )
        val_loss: float
        val_rank_corr: float
        val_loss, val_rank_corr = model_epoch(
            model, val_loader, rank_corr_fn=rank_corr_pairs
        )

        scheduler.step(val_rank_corr)

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
            epochs_without_improvement = 0
            torch.save(model.state_dict(), OUTPUT / run_name / f"model{index}.pt")
            test_loss: float
            test_rank_corr: float
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
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= opts["patience"]:
                logger.info(
                    f"[{run_name}] early stopping triggered after {epoch + 1} epochs."
                )
                break
