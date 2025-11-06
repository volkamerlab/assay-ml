from typing import Callable
from joblib import Parallel, delayed

import pandas as pd
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
        reference_data (pd.Series): Reference activity data grouped by assay and compound.
    """

    def __init__(
        self,
        reference_data: pd.DataFrame,
        pair_predictions: bool,
        rank_statistic: Callable = spearmanr,
        fisher: bool = True,
    ):
        """
        Initialize the AssayRankAccuracy evaluator.

        Args:
            reference_data (pd.DataFrame): DataFrame containing reference activity data.
            pair_predictions (bool): Whether predictions are made for pairs of compounds.
            rank_statistic (Callable, optional): Function to calculate rank correlation.
                Defaults to spearmanr from scipy.stats.
            fisher (bool, optional): Perform a Fisher transform before aggregation.
        """
        self.pair_predictions = pair_predictions
        self.rank_statistic = rank_statistic
        self.reference_data = reference_data.groupby([ASSAY, COMPOUND])[ACT].mean()
        self.fisher = fisher

    def __call__(self, prediction_data: pd.DataFrame) -> float:
        """
        Calculate weighted average rank correlation across assays in parallel.

        Args:
            prediction_data (pd.DataFrame): DataFrame containing model predictions.

        Returns:
            float: Weighted average rank correlation across all assays.
        """
        key_sffx = "_a" if self.pair_predictions else ""

        def process_assay(assay, data, reference_data):
            # if len(data) <= 4 or assay not in self.reference_data:
            #     return 0, 0  # No valid data for this assay

            scores = assay_ranks(data) if self.pair_predictions else data
            scores = scores.set_index(COMPOUND)
            reference = reference_data.reindex(scores.index)

            if len(scores) <= 1 or reference.nunique() <= 1:
                return 0, 0  # Skip invalid assays

            try:
                prediction = scores[PREDICTION].values
                ground_truth = reference.values
                corr = self.rank_statistic(prediction, ground_truth).statistic
                if np.isnan(corr):
                    logger.warning(f"rank correlation is nan (assay={assay})")
                    return 0, 0
                if self.fisher:
                    corr = fisher_transform_numpy(corr)
                    n = len(scores) - 3
                return n * corr, n
            except ValueError as e:
                logger.warning(f"rank correlation failed (assay={assay}): {e}")
                # logger.warning("\n".join(traceback.format_exc().split("\n")))
                return 0, 0

        results = Parallel(n_jobs=8)(
            delayed(process_assay)(assay, data, self.reference_data.loc[assay])
            for assay, data in [
                (a, d)
                for a, d in prediction_data.groupby(ASSAY + key_sffx)
                if a in self.reference_data and len(d) > 4
            ]
        )

        corr_sum, count = map(sum, zip(*results))
        if count <= 0:
            return np.nan
        if self.fisher:
            return np.tanh(corr_sum / count)
        return corr_sum / count

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.rank_statistic})"


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
