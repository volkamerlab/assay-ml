from typing import Type, Any, Dict, Callable
from joblib import Parallel, delayed
from pathlib import Path
from functools import partial

import tqdm
import pandas as pd
import numpy as np
import torch
from torch import nn, Tensor
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from scipy.stats import spearmanr
import logging
from functools import namedtuple
from torcheval.metrics import MulticlassAUROC

from .utils import device
from .constants import ASSAY, OUTPUT, ACT, COMPOUND, PREDICTION, TID
from .hodge_ranking import assay_ranks
from .data import MultiSetActivityDataset

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
    lr=1e-4,
    fisher_transform=True,
)


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


def create_set_attention_mask_from_ids(
    set_ids: torch.Tensor, num_heads: int = None
) -> torch.Tensor:
    """
    Create a mask where elements can only attend within their set.

    Args:
        set_ids: Tensor of shape (N,) with set identifier for each element
        num_heads: Number of attention heads (if None, returns 2D mask)

    Returns:
        Attention mask of shape (N, N) or (num_heads, N, N) where True means "mask out" (no attention)
    """
    set_ids = set_ids.flatten()
    mask = set_ids.unsqueeze(0) != set_ids.unsqueeze(1)  # (N, N)

    if num_heads is not None:
        mask = mask.unsqueeze(0).expand(num_heads, -1, -1)  # (num_heads, N, N)

    return mask


def train_with_batched_masked_sets(
    model,
    loader,
    optimizer,
    mask_fraction: float = 0.2,
    unmasked_weight: float = 1.0,
    **kwargs,
):
    """
    Training loop with one shared BinDistribution.
    Each set is min-max normalized using its unmasked samples before binning.

    - In each group, a fixed fraction of samples (mask_fraction) are masked.
    - Masked indices are chosen randomly each batch.
    - Reconstruction loss: unmasked samples get linear weight `unmasked_weight`.
    """
    model.train()
    logger.info(f"training with unmasked_weight={unmasked_weight}")
    total_loss = 0.0
    steps = 0

    for protein_features, ligand_features, labels, _, metadata in (
        pbar := tqdm.tqdm(loader, desc="training")
    ):
        set_boundaries = metadata["set_boundaries"].squeeze()
        num_sets = metadata["num_sets"].squeeze().item()
        set_ids_tensor = metadata["set_ids_tensor"].to(device)

        ligand_features = ligand_features.squeeze().to(device)
        protein_features = protein_features.squeeze().to(device)
        labels = labels.squeeze().to(device)
        batch_size = labels.size(0)

        sample_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
        normed_labels = torch.empty_like(labels)

        for i in range(num_sets):
            start_idx = set_boundaries[i]
            end_idx = set_boundaries[i + 1]
            set_size = end_idx - start_idx
            assert set_size >= 2

            n_masked = max(1, int(set_size * mask_fraction))
            perm = torch.randperm(set_size, device=device)
            mask_idx = perm[:n_masked]
            mask = torch.zeros(set_size, dtype=torch.bool, device=device)
            mask[mask_idx] = True

            sample_mask[start_idx:end_idx] = mask

            set_labels = labels[start_idx:end_idx]
            unmasked = set_labels[~mask]
            assert unmasked.numel() > 1, unmasked.numel()
            min_val = unmasked.min()
            max_val = unmasked.max()
            denom = (max_val - min_val).clamp_min(1e-6)
            normed_labels[start_idx:end_idx] = (set_labels - min_val) / denom

        preds = model(
            ligand_features,
            protein_features,
            normed_labels,
            sample_mask,
            set_ids_tensor,
        )  # (batch_size, n_bins)

        nll_losses = -model.bin_dist.log_prob(normed_labels, preds)

        weights = torch.where(
            sample_mask,
            torch.ones_like(labels),
            torch.full_like(labels, unmasked_weight),
        )
        batch_loss = (nll_losses * weights).mean()

        pbar.set_description(f"train batch loss={batch_loss:.4e}")

        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        total_loss += batch_loss.item()
        steps += 1

    return total_loss / max(1, steps)


@torch.no_grad()
def evaluate_with_batched_masked_sets(
    model,
    loader,
    mask_fraction: float = 0.2,
    predictions_file: Path | None = None,
):
    """
    Evaluate model on batched masked sets using the shared learnable BinDistribution.
    Each set is min-max normalized using its unmasked samples.

    - Masking comes from `info` (deterministic).
    - Computes average NLL, Brier score, Wasserstein distance, and Expected Value MAE
      separately for masked and unmasked samples.
    - Optionally writes predictions + info to a CSV.
    """
    model.eval()
    device = next(model.parameters()).device

    total_loss_masked = 0.0
    n_masked = 0
    brier_masked = 0.0
    wass_masked = 0.0
    mae_masked = 0.0
    pred_records = []

    bin_centers = model.bin_dist.bin_centers().to(device)

    for protein_features, ligand_features, labels, info, metadata in (
        pbar := tqdm.tqdm(loader, desc="evaluating")
    ):
        set_boundaries = metadata["set_boundaries"].squeeze()
        num_sets = metadata["num_sets"].squeeze()
        set_ids_tensor = metadata["set_ids_tensor"].to(device)

        ligand_features = ligand_features.squeeze().to(device)
        protein_features = protein_features.squeeze().to(device)
        labels = labels.squeeze().to(device)
        info = info.squeeze().to(device)
        batch_size = labels.size(0)

        sample_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
        normed_labels = torch.empty_like(labels)

        for i in range(num_sets):
            start_idx = set_boundaries[i]
            end_idx = set_boundaries[i + 1]
            set_size = end_idx - start_idx
            if set_size < 2:
                continue

            info_batch = info[start_idx:end_idx, :]
            if info_batch.size(1) == 3:
                mask = info_batch[:, 0].flatten().bool()
            else:
                mask = torch.rand(set_size, device=device) < mask_fraction
            sample_mask[start_idx:end_idx] = mask

            set_labels = labels[start_idx:end_idx]
            unmasked = set_labels[~mask]
            assert unmasked.numel() > 1, (
                f"Need at least 2 unmasked samples, got {unmasked.numel()}"
            )

            min_val = unmasked.min()
            max_val = unmasked.max()
            denom = (max_val - min_val).clamp_min(1e-6)
            normed_labels[start_idx:end_idx] = (set_labels - min_val) / denom

        preds = model(
            ligand_features,
            protein_features,
            normed_labels,
            sample_mask,
            set_ids_tensor,
        )  # (batch_size, n_bins)

        nll_losses = -model.bin_dist.log_prob(normed_labels, preds)
        probs = torch.softmax(preds, dim=-1)
        class_labels = model.bin_dist.labels(normed_labels)
        one_hot = model.bin_dist.dist(class_labels)
        expected_vals = (probs * bin_centers).sum(dim=-1)
        brier_scores = ((probs - one_hot) ** 2).sum(dim=-1)  # (batch_size,)
        wass_dists = torch.cumsum(probs, dim=-1) - torch.cumsum(one_hot, dim=-1)
        wass_dists = torch.abs(wass_dists).sum(dim=-1)  # discrete 1D Wasserstein
        mae_vals = torch.abs(expected_vals - normed_labels).clip(0, 1)

        masked_losses = nll_losses[sample_mask]

        if masked_losses.numel() > 0:
            total_loss_masked += masked_losses.sum().item()
            n_masked += masked_losses.numel()
            brier_masked += brier_scores[sample_mask].sum().item()
            wass_masked += wass_dists[sample_mask].sum().item()
            mae_masked += mae_vals[sample_mask].sum().item()

        if predictions_file is not None:
            probs_np = probs.cpu().numpy()
            labels_np = labels.cpu().numpy()
            normed_np = normed_labels.cpu().numpy()
            info_np = info.cpu().numpy()
            mask_np = sample_mask.cpu().numpy()

            for i in range(batch_size):
                record = {
                    **{f"info_{j}": info_np[i, j] for j in range(info_np.shape[1])},
                    "true_label": labels_np[i],
                    "normed_label": normed_np[i],
                    "is_masked": bool(mask_np[i]),
                }
                for b in range(model.n_bins):
                    record[f"prob_bin_{b}"] = probs_np[i, b]
                pred_records.append(record)

    avg_loss_masked = total_loss_masked / max(1, n_masked)
    brier_masked_value = brier_masked / max(1, n_masked)
    wass_masked_value = wass_masked / max(1, n_masked)
    mae_masked_value = mae_masked / max(1, n_masked)

    if predictions_file is not None and pred_records:
        df = pd.DataFrame(pred_records)
        predictions_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(predictions_file, index=False)

    return {
        "loss_masked": avg_loss_masked,
        "brier_masked": brier_masked_value,
        "wass_masked": wass_masked_value,
        "mae_masked": mae_masked_value,
    }


def train_with_batched_sets(
    model,
    loader,
    optimizer,
    criterion,
    fisher_transform=True,
    normalize_training_batches=False,
):
    logger.debug("Training model")
    model.train()
    torch.set_grad_enabled(True)
    total_loss = 0
    steps = 0

    for protein_features, ligand_features, labels, info, metadata in tqdm.tqdm(
        loader, desc="training"
    ):
        set_boundaries = metadata["set_boundaries"].squeeze()
        num_sets = metadata["num_sets"].squeeze()
        set_ids_tensor = metadata["set_ids_tensor"].to(device)

        predictions = model(
            protein_features.squeeze(),
            ligand_features.squeeze(),
            set_ids=set_ids_tensor,
        ).squeeze()

        labels = labels.squeeze()
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
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            total_loss += batch_loss.item()
            steps += 1

    return total_loss / max(1, steps)


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


def train_and_evaluate_pfn_model(
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
            "test_loss",
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

    best_loss = 1000
    epochs_without_improvement = 0
    optimization = []

    for epoch in range(opts["num_epochs"]):
        train_loss = train_with_batched_masked_sets(
            model,
            train_loader,
            optimizer,
            unmasked_weight=opts.get("unmasked_weight", 1.0),
        )
        val_results = evaluate_with_batched_masked_sets(model, val_loader)
        val_loss = val_results["loss_masked"]

        scheduler.step(val_loss)
        lr = scheduler.get_last_lr()
        logger.debug(f"learning rate: {lr}")

        logger.info(f"epoch: {epoch + 1}")
        logger.info(f" train loss: {train_loss:.4e}")
        logger.info(f" validation Brier: {val_results['brier_masked']:.4e}")
        logger.info(f" validation EMD: {val_results['wass_masked']:.4e}")
        logger.info(f" validation MAE: {val_results['mae_masked']:.4e}")

        optimization.append(Epoch(epoch, lr, train_loss, val_loss))
        pd.DataFrame(optimization).to_csv(
            OUTPUT / run_name / "optimization.csv", index=False
        )

        if val_loss < best_loss:
            logger.info("updating test set predictions")
            best_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), OUTPUT / run_name / "model.pt")
            with open(OUTPUT / run_name / "bin_dist", "w") as f_bins:
                f_bins.write(f"{model.bin_dist._construct_edges()}")
            test_results = evaluate_with_batched_masked_sets(
                model,
                test_loader,
                predictions_file=OUTPUT / run_name / "predictions.csv",
            )
            logger.info(f"test epoch: {epoch + 1} ")
            logger.info(f" test loss: {test['loss_masked']:.4e}")
            logger.info(f" test Brier: {test_results['brier_masked']:.4e}")
            logger.info(f" test EMD: {test_results['wass_masked']:.4e}")
            logger.info(f" test MAE: {test_results['mae_masked']:.4e}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= opts["patience_termination"]:
                logger.info(f"early stopping triggered after {epoch + 1} epochs.")
                break
