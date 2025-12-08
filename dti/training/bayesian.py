import copy
import logging
from typing import Type, Any, Dict, Optional, List, Tuple
from collections import defaultdict
import tqdm
import polars as pl
import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from numpy import tanh, arctanh

from ..model.bin_distribution import BinDistribution
from ..utils import device
from ..utils.constants import OUTPUT
from .metrics import MetricTracker, EnsembleMetricTracker

logger = logging.getLogger(__name__)

_defaults = dict(
    protein_dim=1280,
    ligand_dim=2048,
    embedding_size=512,
    hidden_channels=512,
    num_epochs=500,
    lr=5e-5,
    test=True,
    ensemble_size=3,
    unlabeled_ratio=2,
    consistency_weight=1.0,
)


class UnlabeledDatasource:
    def __init__(self, feature_dim: int):
        self.feature_dim = feature_dim

    def get_batch(self, n_samples: int) -> torch.Tensor:
        return torch.randn(n_samples, self.feature_dim, device=device)


def train_ensemble_pfn(
    model_cls: Type[nn.Module],
    run_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    target_name: str,
    **kwargs: Dict[str, Any],
) -> None:
    opts: Dict[str, Any] = _defaults | kwargs
    logger.info(f"Initializing Semi-Supervised Ensemble (N={opts['ensemble_size']})")

    models = []
    for i in range(opts["ensemble_size"]):
        m = model_cls(
            ligand_input_size=opts["ligand_dim"],
            protein_input_size=opts["protein_dim"],
            **opts,
        ).to(device)
        models.append(m)

    logger.info("Fitting BinDistribution on generic training data...")
    models[0].bin_dist.fit(train_loader)
    bin_dist_state = models[0].bin_dist.state_dict()
    for m in models[1:]:
        m.bin_dist.load_state_dict(bin_dist_state)
        m.bin_dist.edges.data.copy_(models[0].bin_dist.edges.data)

    all_params = []
    for m in models:
        all_params.extend(list(m.parameters()))

    optimizer = AdamW(all_params, lr=opts["lr"], weight_decay=1e-2)
    scheduler = CosineAnnealingLR(
        optimizer, T_max=opts["num_epochs"], eta_min=opts.get("min_lr", 1e-6)
    )

    unlabeled_source = UnlabeledDatasource(feature_dim=opts["ligand_dim"])

    best_loss = float("inf")

    for epoch in range(opts["num_epochs"]):
        logger.info(f" == Epoch: {epoch + 1} == ")

        train_loss = train_epoch_semi_supervised(
            models,
            train_loader,
            unlabeled_source,
            optimizer,
            opts["unlabeled_ratio"],
            opts["consistency_weight"],
        )

        val_results = evaluate_ensemble(models, val_loader)

        current_lr = scheduler.get_last_lr()[0]
        results = (
            {"training loss": training_loss}
            | {f"val {k}": v for k, v in val_results.items()}
            | {"lr": scheduler.get_last_lr()[0]}
        for metric, value in results.items():
            logger.info(f" {metric}: {value:.4e}")

        scheduler.step()

        if val_results["NLL"] < best_loss:
            best_loss = val_results["NLL"]
            save_path = OUTPUT / run_name / "ensemble_best.pt"
            (OUTPUT / run_name).mkdir(parents=True, exist_ok=True)

            state = {f"model_{i}": m.state_dict() for i, m in enumerate(models)}
            torch.save(state, save_path)
            logger.info(f"Checkpoint saved: Val NLL {best_loss:.4f}")

    if opts.get("test"):
        load_path = OUTPUT / run_name / "ensemble_best.pt"
        if load_path.exists():
            checkpoint = torch.load(load_path)
            for i, m in enumerate(models):
                m.load_state_dict(checkpoint[f"model_{i}"])

            test_results = evaluate_ensemble(models, test_loader)
            logger.info("=== Final Ensemble Test Results ===")
            for k, v in test_results.items():
                logger.info(f" test {k}: {v:.4f}")


def train_epoch_semi_supervised(
    models: List[nn.Module],
    loader: DataLoader,
    unlabeled_source: UnlabeledDatasource,
    optimizer: torch.optim.Optimizer,
    unlabeled_ratio: int,
    consistency_weight: float,
) -> float:
    [model.train() for model in models]
    total_loss = 0.0
    bd = models[0].bin_dist

    for batch in (pbar := tqdm.tqdm(loader, desc="Semi-Supervised Train")):
        (ligand_feats, labels, _, query_mask, metadata) = _unpack_batch(batch)

        (
            aug_feats,
            aug_labels,
            aug_mask,
            unlabeled_mask,
            aug_boundaries,
            aug_num_sets,
            aug_set_ids,
        ) = augment_batch_with_unlabeled(
            ligand_feats,
            labels,
            query_mask,
            metadata,
            unlabeled_source,
            unlabeled_ratio,
        )

        if bd.bounded_support:
            with torch.no_grad():
                normed_labels = _normalize_sets_minmax(
                    aug_labels,
                    aug_boundaries,
                    aug_num_sets,
                    mask=aug_mask,
                    clip_range=(bd.edges[0], bd.edges[-1]),
                )
        else:
            normed_labels = aug_labels

        context_input = normed_labels.clone()
        context_input[aug_mask] = 0.0

        ensemble_preds = []  # (N_total, n_bins)
        ensemble_means = []  # (N_total,)

        optimizer.zero_grad(set_to_none=True)
        batch_loss = 0.0

        for model in models:
            preds = model(aug_feats, context_input, aug_mask, aug_set_ids)
            ensemble_preds.append(preds)

            labeled_query_mask = aug_mask & (~unlabeled_mask)

            nll = bd.nll(normed_labels[labeled_query_mask], preds[labeled_query_mask])
            batch_loss += nll.mean()  # Add supervised component

            pred_mean = bd.mean(preds)
            ensemble_means.append(pred_mean)

        if consistency_weight > 0.0:
            # Stack means: (n_models, N_total)
            stacked_means = torch.stack(ensemble_means)
            consensus_mean = stacked_means.mean(dim=0).detach()

            consistency_loss_sum = 0.0

            for i, preds in enumerate(ensemble_preds):
                crps = bd.crps(consensus_mean[unlabeled_mask], preds[unlabeled_mask])
                consistency_loss_sum += crps.mean()

            batch_loss += consistency_weight * (consistency_loss_sum / len(models))

        pbar.set_description(f"loss={batch_loss.item():.2e}")
        batch_loss.backward()
        optimizer.step()

        total_loss += batch_loss.item()

    return total_loss / len(loader)


def augment_batch_with_unlabeled(
    ligand_feats,
    labels,
    query_mask,
    metadata,
    unlabeled_source: UnlabeledDatasource,
    ratio: int,
):
    """
    Injects unlabeled data into the flattened batch structure.

    Logic:
    1. Parse existing set boundaries.
    2. For each set, identify # of query samples.
    3. Generate Ratio * QuerySamples of unlabeled data.
    4. Reconstruct the flattened tensors (feats, labels, masks, set_ids).
    """
    boundaries = metadata["set_boundaries"].squeeze().cpu()
    num_sets = metadata["num_sets"]
    if isinstance(num_sets, torch.Tensor):
        num_sets = num_sets.item()

    new_feats_list = []
    new_labels_list = []
    new_query_mask_list = []  # True for Query AND Unlabeled
    unlabeled_mask_list = []  # True ONLY for Unlabeled
    new_set_ids_list = []

    current_idx = 0

    # We process on CPU lists for easy splicing, then move to GPU at end
    # Assuming inputs are on GPU already

    for i in range(num_sets):
        start, end = boundaries[i], boundaries[i + 1]

        # Extract current set data
        set_feats = ligand_feats[start:end]
        set_labels = labels[start:end]
        set_mask = query_mask[start:end]

        # Calculate requirements
        n_query = set_mask.sum().item()
        n_unlabeled = int(n_query * ratio)

        # Fetch unlabeled
        if n_unlabeled > 0:
            unlabeled_feats = unlabeled_source.get_batch(n_unlabeled)

            # Unlabeled Labels (Dummy, will be ignored by mask)
            unlabeled_labels = torch.zeros(n_unlabeled, device=device)

            # Masks
            # Query Mask is True for unlabeled (because they are masked input)
            unlabeled_q_mask = torch.ones(n_unlabeled, dtype=torch.bool, device=device)
            # Specific mask to identify unlabeled section later
            unlabeled_specific_mask = torch.ones(
                n_unlabeled, dtype=torch.bool, device=device
            )

            # Concatenate: [Original Set | Unlabeled]
            combined_feats = torch.cat([set_feats, unlabeled_feats], dim=0)
            combined_labels = torch.cat([set_labels, unlabeled_labels], dim=0)
            combined_q_mask = torch.cat([set_mask, unlabeled_q_mask], dim=0)

            set_unlabeled_mask = torch.cat(
                [
                    torch.zeros(len(set_feats), dtype=torch.bool, device=device),
                    unlabeled_specific_mask,
                ],
                dim=0,
            )

        else:
            combined_feats = set_feats
            combined_labels = set_labels
            combined_q_mask = set_mask
            set_unlabeled_mask = torch.zeros(
                len(set_feats), dtype=torch.bool, device=device
            )

        # Set IDs
        set_len = len(combined_feats)
        combined_ids = torch.full((set_len,), i, device=device)

        new_feats_list.append(combined_feats)
        new_labels_list.append(combined_labels)
        new_query_mask_list.append(combined_q_mask)
        unlabeled_mask_list.append(set_unlabeled_mask)
        new_set_ids_list.append(combined_ids)

    # Reconstruct Tensors
    aug_feats = torch.cat(new_feats_list, dim=0)
    aug_labels = torch.cat(new_labels_list, dim=0)
    aug_mask = torch.cat(new_query_mask_list, dim=0)
    unlabeled_mask = torch.cat(unlabeled_mask_list, dim=0)
    aug_set_ids = torch.cat(new_set_ids_list, dim=0)

    # Reconstruct Boundaries
    # [0, len1, len1+len2, ...]
    lengths = [len(f) for f in new_feats_list]
    aug_boundaries = torch.tensor([0] + list(np.cumsum(lengths)), device=device)

    return (
        aug_feats,
        aug_labels,
        aug_mask,
        unlabeled_mask,
        aug_boundaries,
        num_sets,
        aug_set_ids,
    )


def _unpack_batch(batch):
    """Helper to unpack batch consistently."""
    (ligand_features, labels, info, query_mask, metadata) = batch

    if isinstance(ligand_features, tuple):  # Graph/FP tuple
        # For simplicity in this example, assuming only vector features (Fingerprints)
        # If graph, augment_batch_with_unlabeled needs to handle PyG Batch objects
        _, ligand_features = ligand_features
        ligand_features = ligand_features.squeeze()
    else:
        ligand_features = ligand_features.squeeze()

    ligand_features = ligand_features.to(device)
    labels = labels.squeeze().to(device)
    query_mask = query_mask.squeeze().to(device)

    return ligand_features, labels, info, query_mask, metadata


@torch.no_grad()
def evaluate_ensemble(models, loader):
    [m.eval() for m in models]
    trackers = [
        MetricTracker(models[0].bin_dist, compute_correlations=False) for _ in models
    ]

    for batch in tqdm.tqdm(loader, desc="Ensemble Eval"):
        (ligand_feats, labels, _, query_mask, metadata) = _unpack_batch(batch)

        set_boundaries = metadata["set_boundaries"].squeeze()
        num_sets = metadata["num_sets"]
        if isinstance(num_sets, torch.Tensor):
            num_sets = num_sets.item()

        if models[0].bin_dist.bounded_support:
            normed_labels = _normalize_sets_minmax(
                labels,
                set_boundaries,
                num_sets,
                mask=query_mask,
                clip_range=(models[0].bin_dist.edges[0], models[0].bin_dist.edges[-1]),
            )
        else:
            normed_labels = labels

        context_input = normed_labels.clone()
        context_input[query_mask] = 0.0
        set_ids = metadata["set_ids_tensor"].to(device).squeeze()

        for i, model in enumerate(models):
            preds = model(ligand_feats, context_input, query_mask, set_ids)
            trackers[i].update(
                preds, normed_labels, query_mask, set_boundaries, num_sets
            )

    ensemble_tracker = EnsembleMetricTracker(trackers)
    return ensemble_tracker.compute_averaged()


def train_and_evaluate_pfn_model(
    model_cls: Type[nn.Module],
    run_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    target_name: str,
    index: int,
    metrics_to_monitor: dict = {
        "MAE": "min",
        "Pearson": "max",
        "NLL": "min",
        "CRPS": "min",
    },
    **kwargs: Dict[str, Any],
) -> None:
    logger.info(f"Training model for target: {target_name}")

    opts: Dict[str, Any] = _defaults | kwargs

    logger.info("Training options:")
    for k, v in opts.items():
        logger.info(f" - {k}={v}")

    model = model_cls(
        ligand_input_size=opts["ligand_dim"],
        protein_input_size=opts["protein_dim"],
        **opts,
    ).to(device)

    model.bin_dist.fit(train_loader)
    (OUTPUT / run_name).mkdir(parents=True, exist_ok=True)
    with open(OUTPUT / run_name / "bin_dist", "w") as f_bins:
        f_bins.write(f"{','.join([str(x.item()) for x in model.bin_dist.edges])}")

    optimizer = AdamW(model.parameters(), lr=opts["lr"], weight_decay=1e-2)
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=opts["num_epochs"],
        eta_min=opts.get("min_lr", 1e-6),
    )

    best_metrics = {
        k: (float("inf") if v == "min" else float("-inf"))
        for k, v in metrics_to_monitor.items()
    }

    optimization = []

    for epoch in range(opts["num_epochs"]):
        logger.info(f" == Epoch: {epoch + 1} == ")
        training_loss = train_with_batched_masked_sets(
            model,
            train_loader,
            optimizer,
            unmasked_weight=opts.get("unmasked_weight", 1.0),
            loss_fn=opts["objective"],
        )

        val_results = evaluate_with_batched_masked_sets(
            model, val_loader, loss_fn=opts["objective"]
        )
        results = (
            {"training loss": training_loss}
            | {f"val {k}": v for k, v in val_results.items()}
            | {"lr": scheduler.get_last_lr()[0]}
        )
        optimization.append(results)
        pl.DataFrame(optimization).write_csv(OUTPUT / run_name / "optimization.csv")

        scheduler.step()

        saved_tags = []
        for metric_name, mode in metrics_to_monitor.items():
            if metric_name in val_results:
                current_val = val_results[metric_name]
                best_val = best_metrics[metric_name]

                is_better = (
                    (current_val < best_val)
                    if mode == "min"
                    else (current_val > best_val)
                )

                if is_better:
                    best_metrics[metric_name] = current_val
                    save_path = OUTPUT / run_name / f"model_best_{metric_name}.pt"
                    torch.save(model.state_dict(), save_path)
                    saved_tags.append(f"{metric_name}: {current_val:.4f}")

        if saved_tags:
            logger.info(f"Checkpoints updated: {', '.join(saved_tags)}")

        for metric, value in results.items():
            logger.info(f" {metric}: {value:.4e}")

    if not opts.get("test"):
        return

    logger.info("=== Starting Test Evaluation ===")

    for metric_name in metrics_to_monitor:
        model_path = OUTPUT / run_name / f"model_best_{metric_name}.pt"

        if model_path.exists():
            logger.info(
                f"Loading Best {metric_name} Model (Val {best_metrics[metric_name]:.4f})..."
            )
            model.load_state_dict(torch.load(model_path))

            test_results = evaluate_with_batched_masked_sets(
                model,
                test_loader,
                loss_fn=opts["objective"],
                predictions_file=OUTPUT
                / run_name
                / f"predictions_best_{metric_name}.npz",
            )

            logger.info(f"Test Results for Model Best {metric_name}:")
            for metric, value in test_results.items():
                logger.info(f" test {metric}: {value:.4e}")
        else:
            logger.warning(f"No checkpoint found for Best {metric_name}.")


def train_with_batched_masked_sets(
    model,
    loader,
    optimizer,
    mask_fraction: float = 0.2,
    unmasked_weight: float = 0.0,
    loss_fn: str = "crps",
    **kwargs,
):
    model.train()
    model_device = next(model.parameters()).device
    logger.info(f"Training with unmasked_weight={unmasked_weight}")

    bd = model.bin_dist

    if not hasattr(bd, loss_fn):
        raise ValueError(f"Invalid loss function '{loss_fn}'")
    logger.info(f"Loss function: {loss_fn}")
    calc_loss = getattr(bd, loss_fn)

    tracker = MetricTracker(bd, compute_correlations=False)
    masked_weight = 1.0

    for batch in (pbar := tqdm.tqdm(loader, desc="train")):
        with torch.set_grad_enabled(True):
            (
                preds,
                normed_labels,
                _,
                query_mask,
                _,
                _,
                set_boundaries,
                num_sets,
            ) = _process_batch(
                batch,
                model,
                bd,
                model_device,
                is_train=True,
                mask_fraction=mask_fraction,
            )

            losses = calc_loss(normed_labels, preds)

            if unmasked_weight == 1.0:
                batch_loss = losses.mean()
            elif unmasked_weight == 0.0:
                batch_loss = losses[query_mask].mean()
            else:
                weights = torch.full_like(losses, unmasked_weight)
                weights[query_mask] = masked_weight
                batch_loss = (losses * weights).mean()

            optimizer.zero_grad(set_to_none=True)
            batch_loss.backward()
            optimizer.step()

            loss_value = batch_loss.detach().item()
            pbar.set_description(f"batch loss={loss_value:.4e}")

            tracker.update(
                preds.detach(),
                normed_labels.detach(),
                query_mask,
                set_boundaries,
                num_sets,
                loss_val=loss_value,
            )

    return tracker.compute().get("loss", 0.0)


@torch.no_grad()
def evaluate_with_batched_masked_sets(
    model, loader, predictions_file=None, loss_fn: str = "nll"
):
    model.eval()
    model_device = next(model.parameters()).device
    bd = model.bin_dist

    if not hasattr(bd, loss_fn):
        raise ValueError(f"invalid loss function '{loss_fn}'")
    calc_loss = getattr(bd, loss_fn)

    tracker = MetricTracker(bd, compute_correlations=True)

    all_info, all_labels, all_normed, all_masks, all_probs = [], [], [], [], []
    save_preds = predictions_file is not None

    for batch in tqdm.tqdm(loader, desc="testing" if save_preds else "eval"):
        (
            preds,
            normed_labels,
            labels,
            query_mask,
            _,
            info,
            set_boundaries,
            num_sets,
        ) = _process_batch(batch, model, bd, model_device, is_train=False)

        loss_val = calc_loss(normed_labels, preds)[query_mask].mean()
        tracker.update(
            preds,
            normed_labels,
            query_mask,
            set_boundaries,
            num_sets,
            loss_val=loss_val,
        )

        if save_preds:
            all_info.append(info.cpu())
            all_labels.append(labels.cpu())
            all_normed.append(normed_labels.cpu())
            all_masks.append(query_mask.cpu())
            all_probs.append(torch.softmax(preds, dim=-1).cpu())

    if save_preds and all_info:
        predictions_file = predictions_file.with_suffix(".npz")
        predictions_file.parent.mkdir(parents=True, exist_ok=True)

        data_to_save = {
            "info": torch.cat(all_info).numpy(),
            "true_labels": torch.cat(all_labels).numpy(),
            "normed_labels": torch.cat(all_normed).numpy(),
            "is_masked": torch.cat(all_masks).numpy(),
            "probs": torch.cat(all_probs).numpy(),
        }
        np.savez_compressed(predictions_file, **data_to_save)

    return tracker.compute()


def _process_batch(
    batch: tuple,
    model: nn.Module,
    bd: BinDistribution,
    device: torch.device,
    is_train: bool,
    mask_fraction: float = 0.2,
):
    (
        ligand_features,
        labels,
        info,
        query_mask,
        metadata,
    ) = batch
    set_boundaries = metadata["set_boundaries"].squeeze()
    num_sets = metadata["num_sets"]
    if isinstance(num_sets, torch.Tensor):
        num_sets = num_sets.squeeze().item()
    set_ids_tensor = metadata["set_ids_tensor"].to(device, non_blocking=True)

    if isinstance(ligand_features, torch.Tensor):
        ligand_features = ligand_features.squeeze()
    if isinstance(ligand_features, tuple):
        ligand_graph, ligand_fps = ligand_features
        ligand_fps = ligand_fps.squeeze()
        ligand_features = (
            ligand_graph.to(device, non_blocking=True),
            ligand_fps.to(device, non_blocking=True),
        )
    else:
        ligand_features = ligand_features.to(device, non_blocking=True)
    labels = labels.squeeze().to(device, non_blocking=True)
    info = info.squeeze()
    query_mask = query_mask.squeeze().to(device, non_blocking=True)

    if is_train:
        real_assay = metadata["real_assay"].squeeze().to(device, non_blocking=True)
    else:
        real_assay = None

    if bd.bounded_support:
        with torch.no_grad():
            normed_labels = _normalize_sets_minmax(
                labels,
                set_boundaries,
                num_sets,
                mask=query_mask,
                clip_range=(bd.edges[0], bd.edges[-1]),
            )
    else:
        normed_labels = labels

    context_labels = normed_labels.clone()
    context_labels[query_mask] = 0.0

    with torch.set_grad_enabled(is_train):
        preds = model(
            ligand_features,
            context_labels,
            query_mask,
            set_ids_tensor,
        )

    return (
        preds,
        normed_labels,
        labels,
        query_mask,
        real_assay,
        info,
        set_boundaries,
        num_sets,
    )


def _normalize_sets_minmax(
    labels, set_boundaries, num_sets, mask=None, clip_range=None
):
    normed_labels = torch.zeros_like(labels)

    for i in range(num_sets):
        start_idx = set_boundaries[i]
        end_idx = set_boundaries[i + 1]
        set_labels = labels[start_idx:end_idx]
        set_size = end_idx - start_idx
        if set_size < 2:
            continue

        if mask is not None:
            set_mask = mask[start_idx:end_idx]
            unmasked = set_labels[~set_mask]
        else:
            unmasked = set_labels

        if unmasked.numel() < 2:
            continue

        min_val = unmasked.min()
        max_val = unmasked.max()
        range_val = (max_val - min_val).clamp_min(1e-6)
        normed = (set_labels - min_val) / range_val

        if clip_range is not None:
            normed = normed.clamp(*clip_range)

        normed_labels[start_idx:end_idx] = normed

    return normed_labels
