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
import traceback
from functools import namedtuple

from .utils import device
from .constants import ASSAY, OUTPUT, ACT, COMPOUND, PREDICTION, TID
from .hodge_ranking import assay_ranks

logger = logging.getLogger(__name__)


class AssayRankAccuracy:
    """
    Compute the intra-assay rank correlation weighted by assay size.

    This class evaluates the ranking performance of model predictions against reference data
    on an assay-by-assay basis, weighting the correlation by the number of compounds in each assay.

    Attributes:
        pair_predictions (bool): Whether predictions are made for pairs of compounds.
        rank_statistic (Callable): Function to calculate rank correlation (default: spearmanr).
        reference_data (pd.Series): Reference activity data grouped by assay and compound.
    """

    def __init__(
        self,
        reference_data: pd.DataFrame,
        pair_predictions: bool,
        rank_statistic: Callable = spearmanr,
    ):
        """
        Initialize the AssayRankAccuracy evaluator.

        Args:
            reference_data (pd.DataFrame): DataFrame containing reference activity data.
            pair_predictions (bool): Whether predictions are made for pairs of compounds.
            rank_statistic (Callable, optional): Function to calculate rank correlation.
                Defaults to spearmanr from scipy.stats.
        """
        self.pair_predictions = pair_predictions
        self.rank_statistic = rank_statistic
        self.reference_data = reference_data.groupby([ASSAY, COMPOUND])[ACT].mean()

    def __call__(self, prediction_data: pd.DataFrame) -> float:
        """
        Calculate weighted average rank correlation across assays.

        Args:
            prediction_data (pd.DataFrame): DataFrame containing model predictions.

        Returns:
            float: Weighted average rank correlation across all assays.
        """
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
                    prediction = scores[PREDICTION].values
                    ground_truth = reference.values
                    corr = self.rank_statistic(prediction, ground_truth).statistic
                except ValueError as e:
                    for line in traceback.format_exc().split("\n"):
                        logger.warning(line)
                    logger.warning(f"rank correlation failed (assay={assay}): {e}")
                    continue
                if np.isnan(corr):
                    logger.warning(f"rank correlation is nan (assay={assay})")
                    continue
                corr_sum += len(scores) * corr
                count += len(scores)

        return corr_sum / count

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.rank_statistic})"


def normBCE(pred_deltas: Tensor, true_deltas: Tensor):
    """
    Compute normalized Binary Cross Entropy loss.

    Applies sigmoid activation to the prediction deltas before computing BCE loss.

    Args:
        pred_deltas (Tensor): Predicted delta values.
        true_deltas (Tensor): True delta values.

    Returns:
        Tensor: Normalized BCE loss.
    """
    thres = nn.Sigmoid()
    return nn.BCELoss()(thres(pred_deltas), thres(true_deltas))


def batch_pair_loss(
    predictions: Tensor, labels: Tensor, criterion=nn.MSELoss()
) -> float:
    """
    Compute pairwise loss for a batch of predictions and labels.

    Creates a matrix of prediction differences and label differences, then applies
    the specified loss function.

    Args:
        predictions (Tensor): Model predictions.
        labels (Tensor): Ground truth labels.
        criterion (nn.Module, optional): Loss function. Defaults to nn.MSELoss().

    Returns:
        float: Computed pairwise loss.
    """
    n = len(labels)
    target_delta = labels.view(n, 1) - labels.view(1, n)
    if predictions.shape == (n,):
        predictions = predictions.view(n, 1) - predictions.view(1, n)
        predictions = predictions.flatten()
    return criterion(predictions, target_delta.flatten())


def corr_loss(x: Tensor, y: Tensor) -> float:
    """
    Compute negative Pearson correlation as a loss function.

    Higher correlation results in lower loss. The function handles centering
    and normalization internally.

    Args:
        x (Tensor): First tensor, typically predictions.
        y (Tensor): Second tensor, typically ground truth values.

    Returns:
        float: Negative Pearson correlation coefficient.

    Raises:
        ValueError: If either tensor has zero variance.
    """
    vx = x - torch.mean(x)
    vy = y - torch.mean(y)
    denom = torch.sqrt(torch.sum(vx**2)) * torch.sqrt(torch.sum(vy**2))
    if denom <= 0.0:
        raise ValueError("zero variance in batch")
    return -torch.sum(vx * vy) / denom


def train_multi_batch_epoch(
    model,
    loader,
    optimizer,
    count,
    criterion=nn.MSELoss(),
    normalize_training_batches=True,
    fisher_transform=True,
):
    """
    Train the model for one epoch using multiple batches for gradient accumulation.

    This function implements gradient accumulation, allowing effective batch sizes
    larger than what would fit in memory.

    Args:
        model (nn.Module): The neural network model to train.
        loader (DataLoader): DataLoader providing batches of training data.
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        count (int): Number of samples to accumulate before performing a parameter update.
        criterion (nn.Module, optional): Loss function. Defaults to nn.MSELoss().
        normalize_training_batches (bool, optional): Whether to normalize labels within each batch.
            Defaults to True.
        fisher_transform (bool, optional): Whether to apply Fisher transformation to the loss.
            Defaults to True.

    Returns:
        float: Average loss for the epoch.
    """
    logger.debug("Training model")
    model.train()
    torch.set_grad_enabled(True)

    total_loss = 0
    seen_samples = 0
    batch_loss = 0
    steps = 0

    for protein_features, ligand_features, labels, _, _ in tqdm.tqdm(
        loader, desc="training"
    ):
        protein_features, ligand_features, labels = (
            protein_features.to(device, non_blocking=True),
            ligand_features.to(device, non_blocking=True),
            labels.to(device, non_blocking=True).squeeze(),
        )
        if labels.std() < 1e-10:
            continue
        if normalize_training_batches:
            labels = (labels - labels.mean()) / labels.std()

        predictions = model(protein_features, ligand_features).squeeze()
        assay_size = len(labels)
        loss = criterion(predictions, labels)
        if fisher_transform:
            loss = torch.atanh(torch.clamp(loss, 1e-7 - 1, 1 - 1e-7))
        batch_loss += loss * assay_size
        seen_samples += assay_size

        if seen_samples >= count:
            batch_loss /= seen_samples
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            total_loss += batch_loss.item()
            seen_samples = 0
            batch_loss = 0
            steps += 1

    return total_loss / steps


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
        protein_features, ligand_features, labels, weights = (
            protein_features.to(device),
            ligand_features.to(device),
            labels.to(device).squeeze(),
            weights.to(device),
        )
        if labels.std() < 1e-10:
            logger.warning("low label variance - skipping batch")
        if normalize_training_batches:
            labels = (labels - labels.mean()) / labels.std()

        optimizer.zero_grad()
        assert not protein_features.isnan().any()
        assert not ligand_features.isnan().any()
        predictions = model(protein_features, ligand_features).squeeze()
        assert not predictions.isnan().any(), predictions
        loss = (criterion(predictions, labels) * weights).sum() / weights.sum()
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
            protein_features.to(device).squeeze(0),
            ligand_features.to(device).squeeze(0),
            labels.to(device).squeeze(),
            info.squeeze(0),
        )

        assert not protein_features.isnan().any()
        assert not ligand_features.isnan().any()
        predictions = model(protein_features, ligand_features).squeeze()
        assert not predictions.isnan().any()
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
            normalize_training_batches=False,
            lr=1e-4,
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

    optimizer = torch.optim.Adam(model.parameters(), lr=opts["lr"])
    scheduler = ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=opts["patience_lr"]
    )

    best_corr = 0.0
    epochs_without_improvement = 0
    optimization = []

    for epoch in range(opts["num_epochs"]):
        if opts.get("multi_batch", False):
            batch_size = opts.get("batch_size", 512)
            train_loss = train_multi_batch_epoch(
                model,
                train_loader,
                optimizer,
                batch_size,
                criterion=opts["training_loss"],
                normalize_training_batches=opts["normalize_training_batches"],
            )
        else:
            train_loss = train_epoch(
                model,
                train_loader,
                optimizer,
                criterion=opts["training_loss"],
                normalize_training_batches=opts["normalize_training_batches"],
            )
        val_loss, val_rank_corr = evaluate_epoch(
            model, val_loader, rank_corr_fn=opts["rank_corr_fn"]
        )

        scheduler.step(val_rank_corr)
        lr = scheduler.get_last_lr()
        logger.debug(f"Learning rate: {lr}")

        logger.info(
            f"[{run_name}] Epoch: {epoch + 1} "
            f"Fold: {index} "
            f"Train Loss: {train_loss:.4f} "
            f"Val Loss: {val_loss:.4f} "
            f"Val Rank Corr: {val_rank_corr:.4f}"
        )

        optimization.append(Epoch(epoch, lr, train_loss, val_loss, val_rank_corr))
        pd.DataFrame(optimization).to_csv(
            OUTPUT / run_name / "optimization.csv", index=False
        )

        if val_rank_corr > best_corr:
            logger.info(f"[{run_name}] updating test set predictions")
            best_corr = val_rank_corr
            epochs_without_improvement = 0
            torch.save(model.state_dict(), OUTPUT / run_name / f"model{index}.pt")
            pred_file = OUTPUT / run_name / "predictions.csv"
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
