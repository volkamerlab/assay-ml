from typing import Type, Any, Dict, Callable

import tqdm
import pandas as pd
import numpy as np
import torch
from torch import nn, Tensor
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from scipy.stats import spearmanr

import logging

from .utils import write_info, device
from .constants import ASSAY, OUTPUT, ACT, COMPOUND
from .hodge_ranking import assay_ranks

logger = logging.getLogger(__name__)


class AssayRankAccuracy:
    def __init__(
        self,
        reference_data: pd.DataFrame,
        pair_predictions: bool,
        rank_statistic: Callable = spearmanr,
    ):
        self.pair_predictions = pair_predictions
        self.rank_statistic = rank_statistic
        self.reference_data = reference_data.groupby([ASSAY, COMPOUND])[ACT].mean()

    def __call__(self, prediction_data: pd.DataFrame) -> float:
        count = 0
        corr_sum = 0

        key_sffx = "_a" if self.pair_predictions else ""
        for assay, data in prediction_data.groupby(ASSAY + key_sffx):
            if len(data) <= 1 or assay not in self.reference_data:
                continue

            scores = assay_ranks(data) if self.pair_predictions else data
            scores = scores.set_index(COMPOUND)
            reference = self.reference_data.loc[assay, scores.index]

            if len(scores) > 1 and reference.nunique() > 1:
                try:
                    prediction = scores["prediction"].values
                    ground_truth = reference.values
                    corr = self.rank_statistic(prediction, ground_truth).statistic
                except ValueError:
                    logger.warning(f"rank correlation failed (assay={assay})")
                    continue
                if np.isnan(corr):
                    logger.warning(f"rank correlation is nan (assay={assay})")
                    continue
                corr_sum += len(scores) * corr
                count += len(scores)

        return corr_sum / count


def batch_pair_loss(
    predictions: Tensor, labels: Tensor, criterion=nn.MSELoss()
) -> float:
    n = len(labels)
    target_delta = labels.view(n, 1) - labels.view(1, n)
    return criterion(predictions, target_delta.flatten())


def train_epoch(model, loader, optimizer, criterion=nn.MSELoss()):
    """Train the model for one epoch."""
    logger.debug("Training model")
    model.train()
    torch.set_grad_enabled(True)

    total_loss = 0

    for protein_features, ligand_features, labels, _ in tqdm.tqdm(
        loader, desc="training"
    ):
        protein_features, ligand_features, labels = (
            protein_features.to(device),
            ligand_features.to(device),
            labels.to(device),
        )

        optimizer.zero_grad()
        predictions = model(protein_features, ligand_features).squeeze()
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    total_loss /= len(loader)
    return total_loss


def evaluate_epoch(
    model,
    loader,
    criterion=nn.L1Loss(),
    prediction_file=None,
    rank_corr_fn=None,
):
    """Evaluate the model on validation or test data."""
    logger.debug("Evaluating model")
    model.eval()
    torch.set_grad_enabled(False)

    total_loss = 0
    all_preds, all_labels, all_info = [], [], []

    for protein_features, ligand_features, labels, info in tqdm.tqdm(
        loader, desc="evaluating"
    ):
        protein_features, ligand_features, labels = (
            protein_features.to(device),
            ligand_features.to(device),
            labels.to(device),
        )

        predictions = model(protein_features, ligand_features).squeeze()
        loss = criterion(predictions, labels)

        total_loss += loss.item()
        all_preds.extend(predictions.detach().cpu().numpy().flatten())
        all_labels.extend(labels.detach().cpu().numpy().flatten())
        all_info.append(info.detach().cpu().numpy())

    total_loss /= len(loader)
    torch.set_grad_enabled(True)

    all_info = np.concatenate(all_info)
    content = {"prediction": all_preds, "target": all_labels}
    for i, col in enumerate(loader.dataset.info_cols):
        content[col] = list(all_info[:, i].flatten())

    prediction_data = pd.DataFrame(content)
    if prediction_file is not None:
        logger.info(f"Writing predictions to {prediction_file}")
        prediction_data.to_csv(prediction_file)

    mean_rank_corr = -1 if rank_corr_fn is None else rank_corr_fn(prediction_data)
    return total_loss, mean_rank_corr


def train_and_evaluate_model(
    model_cls: Type[nn.Module],
    run_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
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
            patience_termination=100,
            patience_lr=20,
            rank_corr_fn=None,
            training_loss=nn.MSELoss(),
            cosine_agg=True,
        )
        | kwargs
    )

    logger.info("training options:")
    for k, v in opts.items():
        logger.info(f" - {k}={v}")

    model = model_cls(
        ligand_input_size=opts["ligand_dim"],
        embedding_size=opts["embedding_size"],
        protein_input_size=opts["protein_dim"],
        cosine_agg=opts["cosine_agg"],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=opts["patience_lr"]
    )

    best_corr = 0.0
    epochs_without_improvement = 0

    for epoch in range(opts["num_epochs"]):
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion=opts["training_loss"]
        )
        val_loss, val_rank_corr = evaluate_epoch(
            model, val_loader, rank_corr_fn=opts["rank_corr_fn"]
        )

        scheduler.step(val_rank_corr)
        logger.debug(f"Learning rate: {scheduler.get_last_lr()}")

        logger.info(
            f"[{run_name}] Epoch: {epoch + 1} "
            f"Fold: {index} "
            f"Train Loss: {train_loss:.4f} "
            f"Val Loss: {val_loss:.4f} "
            f"Val Rank Corr: {val_rank_corr:.4f}"
        )

        write_info(
            run_name, [target_name, index, epoch, train_loss, val_loss, val_rank_corr]
        )

        if val_rank_corr > best_corr:
            logger.info(f"[{run_name}] Updating test set predictions")
            best_corr = val_rank_corr
            epochs_without_improvement = 0
            torch.save(model.state_dict(), OUTPUT / run_name / f"model{index}.pt")
            pred_file = OUTPUT / run_name / f"{target_name}_index{index}_preds.csv"
            _, test_rank_corr = evaluate_epoch(
                model,
                test_loader,
                rank_corr_fn=opts["rank_corr_fn"],
                prediction_file=pred_file,
            )
            logger.info(
                f"[{run_name}] Epoch: {epoch + 1} "
                f"Fold: {index} "
                f"Test Rank Corr: {test_rank_corr:.4f}"
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= opts["patience_termination"]:
                logger.info(
                    f"[{run_name}] Early stopping triggered after {epoch + 1} epochs."
                )
                break
