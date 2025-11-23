from typing import Type, Any, Dict, Optional
from collections import defaultdict

import tqdm
import polars as pl
import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import logging
from scipy.stats import spearmanr, pearsonr
from numpy import tanh, arctanh

from ..model.bin_distribution import BinDistribution
from ..utils import device
from ..utils.constants import OUTPUT

logger = logging.getLogger(__name__)

_defaults = dict(
    protein_dim=1280,
    ligand_dim=2048,
    embedding_size=512,
    hidden_channels=512,
    num_epochs=500,
    lr=5e-5,
    test=True,
)


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
        loss_val: Optional[float] = None,
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

    def compute(self) -> Dict[str, float]:
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

    optimizer = torch.optim.AdamW(model.parameters(), lr=opts["lr"], weight_decay=1e-2)
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
            {f"training loss": training_loss}
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

    logger.info("\n=== Starting Comprehensive Test Evaluation ===")

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
    logger.info(f"loss function: {loss_fn}")
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
