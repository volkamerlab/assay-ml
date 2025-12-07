from collections import defaultdict
from typing import Callable, Tuple, List
from joblib import Parallel, delayed

import polars as pl
import numpy as np
import torch
from torch import nn, Tensor
from scipy.stats import spearmanr
import logging

from ..model.bin_distribution import BinDistribution
from ..utils.constants import ASSAY, ACT, COMPOUND, PREDICTION
from ..utils.hodge_ranking import assay_ranks

logger = logging.getLogger(__name__)


class MetricTracker:
    def __init__(self, bin_dist: BinDistribution, compute_correlations: bool = False):
        self.bd = bin_dist
        self.compute_corr = compute_correlations

        self.sums = defaultdict(float)
        self.counts = defaultdict(float)

        self.total_sse = 0.0
        self.total_sst = 0.0

        self.z_rho_sum = 0.0
        self.z_r_sum = 0.0
        self.corr_valid_sets = 0

    @torch.no_grad()
    def update(
        self,
        preds: torch.Tensor,
        normed_labels: torch.Tensor,
        mask: torch.Tensor,
        set_boundaries: torch.Tensor,
        num_sets: int,
        loss_val: float | None = None,
    ):
        masked_preds = preds[mask]
        masked_normed = normed_labels[mask]
        n_masked = masked_preds.size(0)

        if n_masked == 0:
            return

        if loss_val is not None:
            self.sums["loss"] += loss_val * n_masked
            self.counts["loss"] += n_masked

        nll = -self.bd.log_prob(masked_normed, masked_preds).sum().item()
        self.sums["NLL"] += nll
        self.counts["NLL"] += n_masked

        wass = self.bd.emd(masked_normed, masked_preds).sum().item()
        self.sums["EMD"] += wass
        self.counts["EMD"] += n_masked

        crps = self.bd.crps(masked_normed, masked_preds).sum().item()
        self.sums["CRPS"] += crps
        self.counts["CRPS"] += n_masked

        pred_mean = self.bd.mean(masked_preds)
        mae = (pred_mean - masked_normed).abs().sum().item()
        self.sums["MAE"] += mae
        self.counts["MAE"] += n_masked

        probs = torch.softmax(masked_preds, dim=-1)
        true_bins = self.bd.labels(masked_normed)
        one_hot = self.bd.dist(true_bins)
        brier = (probs - one_hot).pow(2).sum(dim=-1).sum().item()
        self.sums["Brier"] += brier
        self.counts["Brier"] += n_masked

        full_pred_means = self.bd.mean(preds)

        batch_sse, batch_sst = self._compute_batch_variance(
            full_pred_means, normed_labels, mask, set_boundaries, num_sets
        )
        self.total_sse += batch_sse
        self.total_sst += batch_sst

        if self.compute_corr:
            z_rho, z_r, valid_sets = self._compute_batch_correlations(
                full_pred_means, normed_labels, mask, set_boundaries, num_sets
            )
            self.z_rho_sum += z_rho
            self.z_r_sum += z_r
            self.corr_valid_sets += valid_sets

    def compute(self) -> dict[str, float]:
        """Returns the averaged metrics."""
        results = {}

        for k in self.sums:
            if self.counts[k] > 0:
                results[k] = self.sums[k] / self.counts[k]
            else:
                results[k] = 0.0

        if self.total_sst > 1e-6:
            results["R2"] = 1.0 - (self.total_sse / self.total_sst)
        else:
            results["R2"] = 0.0

        if self.compute_corr:
            if self.corr_valid_sets > 0:
                avg_z_rho = self.z_rho_sum / self.corr_valid_sets
                avg_z_r = self.z_r_sum / self.corr_valid_sets
                results["Spearman"] = tanh(avg_z_rho)
                results["Pearson"] = tanh(avg_z_r)
            else:
                results["Spearman"] = 0.0
                results["Pearson"] = 0.0

        return results

    def _compute_batch_variance(
        self, pred_means, normed_labels, mask, boundaries, num_sets
    ):
        batch_sse = 0.0
        batch_sst = 0.0

        for i in range(num_sets):
            start, end = boundaries[i], boundaries[i + 1]
            set_mask = mask[start:end]

            masked_true = normed_labels[start:end][set_mask]
            masked_pred = pred_means[start:end][set_mask]

            unmasked_true = normed_labels[start:end][~set_mask]

            if masked_true.numel() == 0 or unmasked_true.numel() == 0:
                continue

            baseline_mean = unmasked_true.mean()

            batch_sse += ((masked_true - masked_pred) ** 2).sum().item()
            batch_sst += ((masked_true - baseline_mean) ** 2).sum().item()

        return batch_sse, batch_sst

    def _compute_batch_correlations(
        self, pred_means, normed_labels, mask, boundaries, num_sets
    ):
        total_z_rho = 0.0
        total_z_r = 0.0
        valid_sets = 0

        pred_np = pred_means.cpu().numpy()
        true_np = normed_labels.cpu().numpy()
        mask_np = mask.cpu().numpy()

        for i in range(num_sets):
            start, end = boundaries[i], boundaries[i + 1]
            set_mask = mask_np[start:end]

            p = pred_np[start:end][set_mask]
            t = true_np[start:end][set_mask]

            n = p.size
            if n < 4:
                continue

            rho, _ = spearmanr(p, t)
            if np.isfinite(rho):
                rho = np.clip(rho, -1 + 1e-6, 1 - 1e-6)
                total_z_rho += arctanh(rho) * (n - 3)

            r, _ = pearsonr(p, t)
            if np.isfinite(r):
                r = np.clip(r, -1 + 1e-6, 1 - 1e-6)
                total_z_r += arctanh(r) * (n - 3)

            valid_sets += n - 3

        return total_z_rho, total_z_r, valid_sets


class EnsembleMetricTracker:
    """Aggregates metrics across the ensemble for unified logging."""

    def __init__(self, trackers: List[MetricTracker]):
        self.trackers = trackers

    def update(self, *args, **kwargs):
        pass

    def compute_averaged(self) -> dict[str, float]:
        total_results = defaultdict(float)
        for t in self.trackers:
            res = t.compute()
            for k, v in res.items():
                total_results[k] += v

        n = len(self.trackers)
        return {k: v / n for k, v in total_results.items()}


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
