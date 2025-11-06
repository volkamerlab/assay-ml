from typing import Type, Any, Dict
from pathlib import Path
from functools import partial

import tqdm
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import logging
from functools import namedtuple

from ..utils import device
from ..utils.constants import OUTPUT, PREDICTION, TID
from ..data.dataset import MultiSetActivityDataset

logger = logging.getLogger(__name__)

_defaults = dict(
    protein_dim=1280,
    ligand_dim=2048,
    embedding_size=512,
    hidden_channels=512,
    num_epochs=500,
    patience_termination=100,
    patience_lr=20,
    rank_corr_fn=None,
    training_loss=nn.MSELoss(),
    cosine_agg=True,
    normalize_training_batches=False,
    lr=5e-5,
    fisher_transform=True,
)


def eval_with_batched_sets(
    model,
    loader,
    criterion=nn.L1Loss(),
    rank_corr_fn=None,
    fisher_transform=True,
    normalize_training_batches=True,
    prediction_file: Path | str | None = None,
):
    """Call with batch size one only."""
    logger.debug("Evaluating model")
    model.eval()
    torch.set_grad_enabled(False)
    total_loss = 0
    steps = 0

    all_preds, all_labels, all_info = [], [], []

    for protein_features, ligand_features, labels, info, metadata in tqdm.tqdm(
        loader, desc="evaluate"
    ):
        set_boundaries = metadata["set_boundaries"].squeeze()
        num_sets = metadata["num_sets"].squeeze()
        labels = labels.squeeze()
        set_ids_tensor = metadata["set_ids_tensor"].to(device)

        predictions = model(
            protein_features.squeeze(),
            ligand_features.squeeze(),
            set_ids=set_ids_tensor,
        ).squeeze()

        all_preds.extend(predictions.detach().cpu().numpy().flatten())
        all_labels.extend(labels.detach().cpu().numpy().flatten())
        all_info.append(info.squeeze().detach().cpu().numpy())

        batch_loss = 0
        total_samples = 0

        for i in range(num_sets):
            start_idx = set_boundaries[i]
            end_idx = set_boundaries[i + 1]

            set_preds = predictions[start_idx:end_idx]
            set_labels = labels[start_idx:end_idx]

            if set_labels.std() < 1e-10:
                continue

            if normalize_training_batches:
                set_labels = (set_labels - set_labels.mean()) / set_labels.std()

            set_loss = criterion(set_preds, set_labels)

            if fisher_transform:
                set_loss = fisher_transform_torch(set_loss)

            set_size = end_idx - start_idx
            batch_loss += set_loss * set_size
            total_samples += set_size

        if total_samples > 0:
            batch_loss /= total_samples

            total_loss += batch_loss.item()
            steps += 1

    all_info = np.concatenate(all_info)
    content = {PREDICTION: all_preds, TID: all_labels}
    for i, col in enumerate(loader.dataset.info_cols):
        content[col] = list(all_info[:, i].flatten())

    prediction_data = pd.DataFrame(content)
    if prediction_file is not None:
        logger.info(f"writing predictions to {prediction_file}")
        prediction_data.to_csv(prediction_file)

    mean_rank_corr = -1 if rank_corr_fn is None else rank_corr_fn(prediction_data)

    return total_loss / max(1, steps), mean_rank_corr


def train_epoch(
    model,
    loader,
    optimizer,
    criterion=nn.MSELoss(),
    normalize_training_batches=False,
):
    """
    Train the model for one epoch.

    Process each batch from the loader, compute loss, and update model parameters.

    Args:
        model (nn.Module): The neural network model to train.
        loader (DataLoader): DataLoader providing batches of training data.
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        criterion (nn.Module, optional): Loss function. Defaults to nn.MSELoss().
        normalize_training_batches (bool, optional): Whether to normalize labels within each batch.
            Defaults to False.

    Returns:
        float: Average loss for the epoch.
    """
    logger.debug("Training model")
    model.train()
    torch.set_grad_enabled(True)

    total_loss = 0

    for protein_features, ligand_features, labels, _, weights in tqdm.tqdm(
        loader, desc="training"
    ):
        labels = labels.squeeze()
        if normalize_training_batches:
            if labels.std() < 1e-10:
                logger.warning("low label variance - skipping batch")
                continue
            labels = (labels - labels.mean()) / labels.std()

        optimizer.zero_grad()
        predictions = model(protein_features, ligand_features).squeeze()
        try:
            loss = (criterion(predictions, labels) * weights).sum() / weights.sum()
        except ValueError as e:
            logger.warning(f"exception in criterion: '{e}'")
            continue
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
    """
    Evaluate the model on validation or test data.

    Process each batch from the loader without updating model parameters, and
    optionally compute rank correlation and save predictions.

    Args:
        model (nn.Module): The neural network model to evaluate.
        loader (DataLoader): DataLoader providing batches of evaluation data.
        criterion (nn.Module, optional): Loss function for evaluation. Defaults to nn.L1Loss().
        prediction_file (str, optional): Path to save prediction results. Defaults to None.
        rank_corr_fn (Callable, optional): Function to compute rank correlation. Defaults to None.

    Returns:
        tuple: A tuple containing:
            - float: Average loss for the evaluation data.
            - float: Mean rank correlation if rank_corr_fn is provided, -1 otherwise.
    """
    logger.debug("Evaluating model")
    model.eval()
    torch.set_grad_enabled(False)

    total_loss = 0
    all_preds, all_labels, all_info = [], [], []

    for protein_features, ligand_features, labels, info, _ in tqdm.tqdm(
        loader, desc="evaluating"
    ):
        protein_features, ligand_features, labels, info = (
            protein_features.squeeze(0),
            ligand_features.squeeze(0),
            labels.squeeze(),
            info.squeeze(0),
        )

        predictions = model(protein_features, ligand_features).squeeze()
        loss = criterion(predictions, labels).mean()

        total_loss += loss.item()
        all_preds.extend(predictions.detach().cpu().numpy().flatten())
        all_labels.extend(labels.detach().cpu().numpy().flatten())
        all_info.append(info.detach().cpu().numpy())

    total_loss /= len(loader)
    torch.set_grad_enabled(True)

    all_info = np.concatenate(all_info)
    content = {PREDICTION: all_preds, TID: all_labels}
    for i, col in enumerate(loader.dataset.info_cols):
        content[col] = list(all_info[:, i].flatten())

    prediction_data = pd.DataFrame(content)
    if prediction_file is not None:
        logger.info(f"writing predictions to {prediction_file}")
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
    """
    Train and evaluate the model with learning rate adjustment and early stopping.

    This function handles the complete training pipeline including model instantiation,
    optimization, learning rate scheduling, early stopping, and model persistence.

    Args:
        model_cls (Type[nn.Module]): Model class to instantiate.
        run_name (str): Name of the training run for logging and file naming.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        test_loader (DataLoader): DataLoader for test data.
        target_name (str): Name of the target being predicted.
        index (int): Index/fold number for cross-validation.
        **kwargs (Dict[str, Any]): Additional configuration options, including:
            - protein_dim (int): Dimension of protein features.
            - ligand_dim (int): Dimension of ligand features.
            - embedding_size (int): Size of embeddings in the model.
            - num_epochs (int): Maximum number of training epochs.
            - patience_termination (int): Number of epochs without improvement before stopping.
            - patience_lr (int): Number of epochs without improvement before reducing learning rate.
            - rank_corr_fn (Callable): Function to compute rank correlation.
            - training_loss (nn.Module): Loss function for training.
            - cosine_agg (bool): Whether to use cosine similarity for aggregation.
            - normalize_training_batches (bool): Whether to normalize batches during training.
            - lr (float): Initial learning rate.
            - multi_batch (bool): Whether to use multi-batch training.
            - batch_size (int): Batch size for multi-batch training.
            - fisher_transform (bool): Fisher transform criterion values in set training before aggregation during training.

    Returns:
        None: The function saves the model and training statistics but doesn't return a value.
    """
    logger.info(f"training model for target: {target_name}")
    Epoch = namedtuple(
        "Epoch",
        [
            "epoch",
            "lr",
            "train_loss",
            "val_loss",
            "val_rank_corr",
        ],
    )

    opts: Dict[str, Any] = _defaults | kwargs

    logger.info("training options:")
    for k, v in opts.items():
        logger.info(f" - {k}={v}")

    model = model_cls(
        ligand_input_size=opts["ligand_dim"],
        protein_input_size=opts["protein_dim"],
        **opts,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=opts["lr"])
    scheduler = ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=opts["patience_lr"]
    )

    train_fn = train_epoch
    if isinstance(train_loader.dataset, MultiSetActivityDataset):
        train_fn = partial(
            train_with_batched_sets, fisher_transform=opts["fisher_transform"]
        )

    eval_fn = evaluate_epoch
    if isinstance(test_loader.dataset, MultiSetActivityDataset):
        eval_fn = eval_with_batched_sets

    best_corr = 0.0
    epochs_without_improvement = 0
    optimization = []

    for epoch in range(opts["num_epochs"]):
        train_loss = train_fn(
            model,
            train_loader,
            optimizer,
            criterion=opts["training_loss"],
            normalize_training_batches=opts["normalize_training_batches"],
        )
        val_loss, val_rank_corr = eval_fn(
            model, val_loader, rank_corr_fn=opts["rank_corr_fn"]
        )

        scheduler.step(val_rank_corr)
        lr = scheduler.get_last_lr()
        logger.debug(f"Learning rate: {lr}")

        logger.info(
            f"epoch: {epoch + 1} "
            f"fold: {index} "
            f"train loss: {train_loss:.4f} "
            f"val loss: {val_loss:.4f} "
            f"val rank corr: {val_rank_corr:.4f}"
        )

        optimization.append(Epoch(epoch, lr, train_loss, val_loss, val_rank_corr))
        pd.DataFrame(optimization).to_csv(
            OUTPUT / run_name / "optimization.csv", index=False
        )

        if val_rank_corr > best_corr:
            logger.info("updating test set predictions")
            best_corr = val_rank_corr
            epochs_without_improvement = 0
            torch.save(model.state_dict(), OUTPUT / run_name / f"model{index}.pt")
            pred_file = OUTPUT / run_name / "predictions.csv"
            _, test_rank_corr = eval_fn(
                model,
                test_loader,
                rank_corr_fn=opts["rank_corr_fn"],
                prediction_file=pred_file,
            )
            logger.info(
                f"epoch: {epoch + 1} fold: {index} test rank corr: {test_rank_corr:.4f}"
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= opts["patience_termination"]:
                logger.info(f"early stopping triggered after {epoch + 1} epochs.")
                break
