import copy
from typing import Type, Any, Dict, Callable, List
from joblib import Parallel, delayed
from pathlib import Path
from functools import partial

import tqdm
import pandas as pd
import numpy as np
import torch
from torch.func import stack_module_state, functional_call
from torch import nn, Tensor, vmap
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from scipy.stats import spearmanr

import logging
from functools import namedtuple

from .utils import device
from .constants import ASSAY, OUTPUT, ACT, COMPOUND, PREDICTION, TID
from .hodge_ranking import assay_ranks
from .data import MultiSetActivityDataset

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

    for protein_features, ligand_features, labels, _, weights in (
        pbar := tqdm.tqdm(loader, desc="training")
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

        pbar.set_description(f"loss={loss.item():.2e}")

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
        logger.info(f"Writing predictions to {prediction_file}")
        prediction_data.to_csv(prediction_file)

    mean_rank_corr = -1 if rank_corr_fn is None else rank_corr_fn(prediction_data)
    return total_loss, mean_rank_corr


def train_and_evaluate_model_pointwise(
    model_cls: Type[nn.Module],
    run_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    target_name: str,
    index: int,
    **kwargs: Dict[str, Any],
) -> None:
    logger.info(f"training model (pointwise) for target: {target_name}")
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
            best_corr = val_rank_corr
            epochs_without_improvement = 0
            torch.save(model.state_dict(), OUTPUT / run_name / f"model{index}.pt")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= opts["patience_termination"]:
                logger.info(
                    f"[{run_name}] Early stopping triggered after {epoch + 1} epochs."
                )
                break

    logger.info(f"[{run_name}] Loading best model for final test evaluation")
    model.load_state_dict(torch.load(OUTPUT / run_name / f"model{index}.pt"))
    pred_file = OUTPUT / run_name / "predictions.csv"
    _, test_rank_corr = evaluate_epoch(
        model,
        test_loader,
        rank_corr_fn=opts["rank_corr_fn"],
        prediction_file=pred_file,
    )
    logger.info(f"[{run_name}] Final Test Rank Corr: {test_rank_corr:.4f}")


def train_and_evaluate_model_setbased(
    model_cls: Type[nn.Module],
    run_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    target_name: str,
    index: int,
    **kwargs: Dict[str, Any],
) -> None:
    logger.info(f"training model (set-based) for target: {target_name}")
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
            fisher_transform=True,
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
        train_loss = train_with_batched_sets(
            model,
            train_loader,
            optimizer,
            criterion=opts["training_loss"],
            fisher_transform=opts["fisher_transform"],
            normalize_training_batches=opts["normalize_training_batches"],
        )
        val_loss, val_rank_corr = eval_with_batched_sets(
            model,
            val_loader,
            rank_corr_fn=opts["rank_corr_fn"],
            fisher_transform=opts["fisher_transform"],
            normalize_training_batches=opts["normalize_training_batches"],
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
            best_corr = val_rank_corr
            epochs_without_improvement = 0
            torch.save(model.state_dict(), OUTPUT / run_name / f"model{index}.pt")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= opts["patience_termination"]:
                logger.info(
                    f"[{run_name}] Early stopping triggered after {epoch + 1} epochs."
                )
                break

    logger.info(f"[{run_name}] Loading best model for final test evaluation")
    model.load_state_dict(torch.load(OUTPUT / run_name / f"model{index}.pt"))
    pred_file = OUTPUT / run_name / "predictions.csv"
    _, test_rank_corr = eval_with_batched_sets(
        model,
        test_loader,
        rank_corr_fn=opts["rank_corr_fn"],
        prediction_file=pred_file,
        fisher_transform=opts["fisher_transform"],
        normalize_training_batches=opts["normalize_training_batches"],
    )
    logger.info(f"[{run_name}] Final Test Rank Corr: {test_rank_corr:.4f}")


def train_and_evaluate_model_setbased_ensemble(
    model_cls: Type[nn.Module],
    run_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    target_name: str,
    index: int,
    ensemble_size=5,
    **kwargs: Dict[str, Any],
) -> None:
    logger.info(f"training model (set-based) for target: {target_name}")
    Epoch = namedtuple(
        "Epoch",
        ["epoch", "lr", "train_loss", "val_loss", "val_rank_corr"],
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
            fisher_transform=True,
        )
        | kwargs
    )

    logger.info("training options:")
    for k, v in opts.items():
        logger.info(f" - {k}={v}")

    create_model = partial(
        model_cls,
        ligand_input_size=opts["ligand_dim"],
        embedding_size=opts["embedding_size"],
        protein_input_size=opts["protein_dim"],
        cosine_agg=opts["cosine_agg"],
    )

    models = [create_model().to(device) for _ in range(ensemble_size)]

    params = []
    for model in models:
        params.extend(list(model.parameters()))

    optimizer = torch.optim.Adam(params, lr=opts["lr"])
    scheduler = ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=opts["patience_lr"]
    )

    best_corr = 0.0
    epochs_without_improvement = 0
    optimization = []

    (OUTPUT / run_name).mkdir(parents=True, exist_ok=True)

    for epoch in range(opts["num_epochs"]):
        train_loss = train_with_batched_sets_ensemble(
            models,
            train_loader,
            optimizer,
            criterion=opts["training_loss"],
            fisher_transform=opts["fisher_transform"],
            unlabeled_weight=min(1, epoch / 100),
        )

        val_loss, val_rank_corr = eval_with_batched_sets_ensemble(
            models,
            val_loader,
            rank_corr_fn=opts["rank_corr_fn"],
            fisher_transform=opts["fisher_transform"],
        )

        scheduler.step(val_rank_corr)
        lr = scheduler.get_last_lr()[0]
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
            best_corr = val_rank_corr
            epochs_without_improvement = 0
            for i, model in enumerate(models):
                torch.save(model.state_dict(), OUTPUT / run_name / f"model{i}.pt")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= opts["patience_termination"]:
                logger.info(
                    f"[{run_name}] Early stopping triggered after {epoch + 1} epochs."
                )
                break

    logger.info(f"[{run_name}] Loading best model for final test evaluation")
    for i, model in enumerate(models):
        model.load_state_dict(torch.load(OUTPUT / run_name / f"model{i}.pt"))

    pred_file = OUTPUT / run_name / "predictions.csv"

    _, test_rank_corr = eval_with_batched_sets_ensemble(
        models,
        test_loader,
        rank_corr_fn=opts["rank_corr_fn"],
        prediction_file=pred_file,
        fisher_transform=opts["fisher_transform"],
        normalize_training_batches=opts["normalize_training_batches"],
    )
    logger.info(f"[{run_name}] Final Test Rank Corr: {test_rank_corr:.4f}")


def train_with_batched_sets_ensemble(
    models: List[nn.Module],
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    fisher_transform: bool = True,
    unlabeled_weight: float = 0.1,
):
    """
    Trains an ensemble of models using a standard loop and semi-supervised set-based loss.
    Gradients are accumulated across all models for a single optimizer step.
    """
    logger.debug("Training model (Ensemble - Standard Loop)")
    for model in models:
        model.train()

    torch.set_grad_enabled(True)
    total_epoch_loss = 0
    steps = 0
    ensemble_size = len(models)

    for protein_features, ligand_features, labels, info, metadata in (
        pbar := tqdm.tqdm(loader, desc="training")
    ):
        p_feat = protein_features.squeeze().to(device)
        l_feat = ligand_features.squeeze().to(device)
        labels = labels.squeeze().to(device)

        set_boundaries = metadata["set_boundaries"].squeeze().to(device)
        set_ids_tensor = metadata["set_ids_tensor"].to(device)
        set_labeled = metadata["set_labeled"]  # list of bools
        num_sets = metadata["num_sets"]

        attn_mask = None
        if hasattr(models[0], "num_heads"):
            attn_mask = create_set_attention_mask_from_ids(
                set_ids_tensor, models[0].num_heads
            ).to(device)

        optimizer.zero_grad()

        ensemble_batch_loss = 0.0

        all_predictions = []

        for model in models:
            model_kwargs = {"attn_mask": attn_mask} if attn_mask is not None else {}
            predictions = model(p_feat, l_feat, **model_kwargs).squeeze()
            all_predictions.append(predictions.unsqueeze(0))

        predictions_stack = torch.cat(all_predictions, dim=0)

        ensemble_consensus = predictions_stack.mean(dim=0).detach()

        for model_idx, predictions in enumerate(all_predictions):
            current_model_loss = 0.0
            total_samples_in_batch = 0.0

            for i in range(num_sets):
                start_idx = set_boundaries[i]
                end_idx = set_boundaries[i + 1]
                set_size = end_idx - start_idx

                set_preds = predictions[:, start_idx:end_idx]

                if set_labeled[i]:
                    set_targets = labels[start_idx:end_idx].unsqueeze(0)
                else:
                    set_targets = ensemble_consensus[start_idx:end_idx].unsqueeze(0)

                if set_targets.std() < 1e-8:
                    continue

                loss_val = criterion(set_preds, set_targets)

                if fisher_transform:
                    loss_val = fisher_transform_torch(loss_val)

                weight = 1.0 if set_labeled[i] else unlabeled_weight
                current_model_loss += loss_val * set_size * weight
                total_samples_in_batch += set_size * weight

            if total_samples_in_batch > 0:
                current_model_loss /= total_samples_in_batch

            ensemble_batch_loss += current_model_loss.item()

            if total_samples_in_batch > 0:
                current_model_loss.backward()

        optimizer.step()

        avg_batch_loss = ensemble_batch_loss / ensemble_size
        total_epoch_loss += avg_batch_loss
        steps += 1
        pbar.set_description(f"loss={avg_batch_loss:.2e}")

    return total_epoch_loss / max(1, steps)


def eval_with_batched_sets_ensemble(
    models: List[nn.Module],
    loader: DataLoader,
    criterion=nn.L1Loss(),
    rank_corr_fn=None,
    fisher_transform=True,
    normalize_training_batches=True,
    prediction_file: Path | str | None = None,
):
    logger.debug("Evaluating model (Ensemble)")
    for model in models:
        model.eval()

    # Prepare vmap state for evaluation
    params, buffers = stack_module_state(models)
    meta_model = copy.deepcopy(models[0]).to("meta")

    def fmodel(params, buffers, p_feat, l_feat, mask):
        return functional_call(meta_model, (params, buffers), (p_feat, l_feat, mask))

    torch.set_grad_enabled(False)
    total_loss = 0
    steps = 0

    all_preds, all_labels, all_info = [], [], []

    for protein_features, ligand_features, labels, info, metadata in tqdm.tqdm(
        loader, desc="evaluate"
    ):
        p_feat = protein_features.squeeze().to(device)
        l_feat = ligand_features.squeeze().to(device)
        labels = labels.squeeze().to(device)
        set_boundaries = metadata["set_boundaries"].squeeze().to(device)
        num_sets = metadata["num_sets"].squeeze()
        set_ids_tensor = metadata["set_ids_tensor"].to(device)
        set_labeled = metadata["set_labeled"]  # list of bools

        model_kwargs = dict()
        attn_mask = None
        if hasattr(models[0], "num_heads"):
            attn_mask = create_set_attention_mask_from_ids(
                set_ids_tensor, models[0].num_heads
            ).to(device)

        # 1. Vectorized Inference
        # Output shape: (Ensemble_Size, Batch_Size)
        raw_preds = vmap(
            fmodel, in_dims=(0, 0, None, None, None), randomness="different"
        )(params, buffers, p_feat, l_feat, attn_mask)

        # 2. Aggregate Predictions (Mean voting)
        # Shape: (Batch_Size)
        predictions = raw_preds.mean(dim=0)

        all_preds.extend(predictions.cpu().numpy().flatten())
        all_labels.extend(labels.cpu().numpy().flatten())
        all_info.append(info.squeeze().numpy())

        batch_loss = 0
        total_samples = 0

        for i in range(num_sets):
            if not set_labeled[i]:
                continue
            start_idx = set_boundaries[i]
            end_idx = set_boundaries[i + 1]

            set_preds = predictions[start_idx:end_idx]
            set_labels = labels[start_idx:end_idx]

            if set_labels.std() < 1e-8:
                continue

            if normalize_training_batches:
                set_labels = (set_labels - set_labels.mean()) / (
                    set_labels.std() + 1e-8
                )

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

    # Handle info cols dynamically if possible, or assume indices
    if hasattr(loader.dataset, "info_cols"):
        for i, col in enumerate(loader.dataset.info_cols):
            content[col] = list(all_info[:, i].flatten())

    prediction_data = pd.DataFrame(content)
    if prediction_file is not None:
        logger.info(f"Writing predictions to {prediction_file}")
        prediction_data.to_csv(prediction_file)

    mean_rank_corr = -1 if rank_corr_fn is None else rank_corr_fn(prediction_data)

    return total_loss / max(1, steps), mean_rank_corr
