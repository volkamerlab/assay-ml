from typing import Callable, Tuple, List
from joblib import Parallel, delayed

import polars as pl
import numpy as np
import torch
from torch import nn, Tensor
from scipy.stats import spearmanr
import logging

from ..utils.constants import ASSAY, ACT, COMPOUND, PREDICTION
from ..utils.hodge_ranking import assay_ranks

logger = logging.getLogger(__name__)


class AssayRankAccuracy:
    """
    Compute the intra-assay rank correlation weighted by assay size.

    This class evaluates the ranking performance of model predictions against reference data
    on an assay-by-assay basis, weighting the correlation by the number of compounds in each assay.

    Attributes:
        pair_predictions (bool): Whether predictions are made for pairs of compounds.
        rank_statistic (Callable): Function to calculate rank correlation (default: spearmanr).
        reference_data (pl.DataFrame): Reference activity data grouped by assay and compound.
    """

    def __init__(
        self,
        reference_data: pl.DataFrame,  # Changed to pl.DataFrame
        pair_predictions: bool,
        rank_statistic: Callable = spearmanr,
        fisher: bool = True,
    ):
        self.pair_predictions = pair_predictions
        self.rank_statistic = rank_statistic
        self.fisher = fisher

        self.reference_data = (
            reference_data.group_by(ASSAY, COMPOUND)
            .agg(pl.col(ACT).mean().alias(ACT))
            .sort(ASSAY, COMPOUND)
        )

    def __call__(self, prediction_data: pl.DataFrame) -> float:
        key_sffx = "_a" if self.pair_predictions else ""
        assay_col = ASSAY + key_sffx

        def process_assay(
            assay: str, data: pl.DataFrame, reference_subset: np.ndarray
        ) -> Tuple[float, float]:
            scores = assay_ranks(data) if self.pair_predictions else data

            compound_list = scores[COMPOUND].to_list()
            reference_df = self.reference_data.filter(
                (pl.col(ASSAY) == assay) & (pl.col(COMPOUND).is_in(compound_list))
            ).sort(COMPOUND)

            scores = scores.sort(COMPOUND)

            merged_data = scores.join(
                reference_df, on=COMPOUND, how="inner", suffixes=("_pred", "_ref")
            )

            if len(merged_data) <= 1 or merged_data[ACT].n_unique() <= 1:
                return 0, 0

            try:
                prediction = merged_data[PREDICTION].to_numpy()
                ground_truth = merged_data[ACT].to_numpy()

                corr = self.rank_statistic(prediction, ground_truth).statistic

                if np.isnan(corr):
                    logger.warning(f"rank correlation is nan (assay={assay})")
                    return 0, 0

                if self.fisher:
                    corr = fisher_transform_numpy(corr)
                    n = len(merged_data) - 3  # Degrees of freedom for Fisher transform
                else:
                    n = len(merged_data)

                return n * corr, n

            except ValueError as e:
                logger.warning(f"rank correlation failed (assay={assay}): {e}")
                return 0, 0

        valid_assays = self.reference_data[ASSAY].unique().to_list()

        prediction_data = prediction_data.filter(pl.col(assay_col).is_in(valid_assays))

        groups_list = []
        for assay, data in prediction_data.group_by(assay_col, maintain_order=True):
            if len(data) > 4:
                groups_list.append((assay, data))

        results: List[Tuple[float, float]] = Parallel(n_jobs=8)(
            delayed(process_assay)(assay, data, None) for assay, data in groups_list
        )

        if not results:
            return np.nan

        corr_sum, count = map(sum, zip(*results))

        if count <= 0:
            return np.nan

        if self.fisher:
            return np.tanh(corr_sum / count)

        return corr_sum / count

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.rank_statistic.__name__})"


def fisher_transform_numpy(corr: float) -> float:
    """Fisher transform a correlation using numpy."""
    return np.atanh(np.clip(corr, 1e-7 - 1, 1 - 1e-7))


def fisher_transform_torch(corr: Tensor) -> Tensor:
    """Fisher transform a correlation using pytorch."""
    return torch.atanh(torch.clamp(corr, 1e-7 - 1, 1 - 1e-7))


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

    Args:
        x (Tensor): First tensor, typically predictions.
        y (Tensor): Second tensor, typically ground truth values.

    Returns:
        float: Negative Pearson correlation coefficient.
    """
    vx = x - torch.mean(x)
    vy = y - torch.mean(y)
    denom = torch.sqrt(torch.sum(vx**2)) * torch.sqrt(torch.sum(vy**2))
    if denom <= 0.0:
        denom = torch.tensor(1e-8, dtype=torch.float32)
    return -torch.sum(vx * vy) / denom
