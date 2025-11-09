from typing import Type, Any, Dict

import tqdm
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
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
    patience_termination=30,
    patience_lr=10,
    lr=5e-5,
)


def _step_mae(
    bd: BinDistribution,
    masked_preds: torch.Tensor,
    masked_normed: torch.Tensor,
) -> float:
    pred_mean = bd.mean(masked_preds)
    assert pred_mean.size() == masked_normed.size()
    return (pred_mean - masked_normed).abs().sum().item()


def _step_wass(
    bd: BinDistribution,
    masked_preds: torch.Tensor,
    masked_normed: torch.Tensor,
) -> float:
    bin_edges = bd.edges.to(device)
    bin_widths = torch.diff(bin_edges)
    total_width = (bin_edges[-1] - bin_edges[0]).clamp_min(1e-6)

    probs = torch.softmax(masked_preds, dim=-1)
    true_bins = bd.labels(masked_normed)
    one_hot = bd.dist(true_bins)

    cdf_pred = torch.cumsum(probs, dim=-1)
    cdf_true = torch.cumsum(one_hot, dim=-1)
    wass_dists = (torch.abs(cdf_pred - cdf_true) * bin_widths).sum(dim=-1) / total_width

    return wass_dists.sum().item()


def _step_var_explained(
    bd: BinDistribution,
    preds: torch.Tensor,
    normed_labels: torch.Tensor,
    sample_mask: torch.Tensor,
    set_boundaries: torch.Tensor,
    num_sets: int,
) -> (float, float):
    total_set_sse = 0.0
    total_set_sst = 0.0
    pred_means = bd.mean(preds)

    for i in range(num_sets):
        start_idx = set_boundaries[i]
        end_idx = set_boundaries[i + 1]

        set_mask = sample_mask[start_idx:end_idx]
        n_samples = set_mask.float().sum()
        set_normed_labels = normed_labels[start_idx:end_idx]
        set_pred_means = pred_means[start_idx:end_idx]

        masked_true_in_set = set_normed_labels[set_mask]
        masked_preds_in_set = set_pred_means[set_mask]

        if masked_true_in_set.numel() == 0:
            continue

        unmasked_true_in_set = set_normed_labels[~set_mask]

        if unmasked_true_in_set.numel() == 0:
            continue

        baseline_mean = unmasked_true_in_set.mean()

        set_sse = ((masked_true_in_set - masked_preds_in_set) ** 2).sum()
        set_sst = ((masked_true_in_set - baseline_mean) ** 2).sum()

        total_set_sse += set_sse.item()
        total_set_sst += set_sst.item()

    return total_set_sse, total_set_sst


def _step_corr_per_set(
    corr_fn: spearmanr,
    bd: BinDistribution,
    preds: torch.Tensor,
    normed_labels: torch.Tensor,
    sample_mask: torch.Tensor,
    set_boundaries: torch.Tensor,
    num_sets: int,
) -> (float, int):
    total_z_transformed_rho = 0.0
    num_valid_sets = 0
    pred_means = bd.mean(preds).cpu().numpy()
    normed_labels_np = normed_labels.cpu().numpy()
    sample_mask_np = sample_mask.cpu().numpy()

    for i in range(num_sets):
        start_idx = set_boundaries[i]
        end_idx = set_boundaries[i + 1]

        set_mask = sample_mask_np[start_idx:end_idx]

        masked_preds_in_set = pred_means[start_idx:end_idx][set_mask]
        masked_labels_in_set = normed_labels_np[start_idx:end_idx][set_mask]

        n_samples = masked_preds_in_set.size

        if n_samples < 4:
            continue

        try:
            rho, _ = corr_fn(masked_preds_in_set, masked_labels_in_set)

            if not np.isfinite(rho):
                rho = 0.0
            rho = np.clip(rho, -1 + 1e-6, 1 - 1e-6)

            z_transformed_rho = arctanh(rho)

            total_z_transformed_rho += z_transformed_rho * (n_samples - 3)
            num_valid_sets += n_samples - 3

        except ValueError:
            continue

    return total_z_transformed_rho, num_valid_sets


def _step_brier(
    bd: BinDistribution,
    masked_preds: torch.Tensor,
    masked_normed: torch.Tensor,
) -> float:
    probs = torch.softmax(masked_preds, dim=-1)
    true_bins = bd.labels(masked_normed)
    one_hot = bd.dist(true_bins)

    brier_scores = (probs - one_hot).pow(2).sum(dim=-1)

    return brier_scores.sum().item()


def _compute_masked_metrics(
    bd: BinDistribution,
    preds: torch.Tensor,
    normed_labels: torch.Tensor,
    mask: torch.Tensor,
    calculate_var_explained: bool = False,
    set_boundaries: torch.Tensor = None,
    num_sets: int = 0,
) -> dict:
    """
    Computes all metrics for a given set of masked predictions and labels.
    """
    masked_preds = preds[mask]
    masked_normed = normed_labels[mask]

    n_masked = float(masked_preds.size(0))

    mae = _step_mae(bd, masked_preds, masked_normed)
    wass = _step_wass(bd, masked_preds, masked_normed)
    brier = _step_brier(bd, masked_preds, masked_normed)

    metrics = {
        "mae": mae,
        "wass": wass,
        "brier": brier,
        "n_masked": n_masked,
    }

    if calculate_var_explained:
        assert set_boundaries is not None, "set_boundaries must be provided for R2"
        assert num_sets > 0, "num_sets must be > 0 for R2"

        sse, sst = _step_var_explained(
            bd, preds, normed_labels, mask, set_boundaries, num_sets
        )
        metrics["sse"] = sse
        metrics["sst"] = sst

    return metrics


def _process_batch(
    batch: tuple,
    model: nn.Module,
    bd: BinDistribution,
    device: torch.device,
    is_train: bool,
    mask_fraction: float = 0.2,
):
    (
        protein_features,
        ligand_features,
        labels,
        info,
        metadata,
    ) = batch

    set_boundaries = metadata["set_boundaries"].squeeze()
    num_sets = metadata["num_sets"].squeeze().item()
    set_ids_tensor = metadata["set_ids_tensor"].to(device, non_blocking=True)

    ligand_features = ligand_features.squeeze().to(device, non_blocking=True)
    protein_features = protein_features.squeeze().to(device, non_blocking=True)
    labels = labels.squeeze().to(device, non_blocking=True)
    info = info.squeeze().to(device, non_blocking=True)
    batch_size = labels.size(0)

    if is_train:
        sample_mask = _generate_mask_for_sets(
            set_boundaries, num_sets, batch_size, mask_fraction
        )
        real_assay = metadata["real_assay"].squeeze().to(device, non_blocking=True)
    else:
        assert len(torch.unique(info[:, 0])) == 2
        sample_mask = info[:, 0].bool()
        real_assay = None

    with torch.no_grad():
        clip_range = (bd.edges[0], bd.edges[-1])
        normed_labels = _normalize_sets_minmax(
            labels,
            set_boundaries,
            num_sets,
            mask=sample_mask,
            clip_range=clip_range,
        )

    context_labels = normed_labels.clone()
    context_labels[sample_mask] = 0.0

    with torch.set_grad_enabled(is_train):
        preds = model(
            ligand_features,
            protein_features,
            context_labels,
            sample_mask,
            set_ids_tensor,
        )

    return (
        preds,
        normed_labels,
        labels,
        sample_mask,
        real_assay,
        info,
        set_boundaries,
        num_sets,
    )


def train_with_batched_masked_sets(
    model,
    loader,
    optimizer,
    mask_fraction: float = 0.2,
    unmasked_weight: float = 0.0,
    **kwargs,
):
    model.train()
    model_device = next(model.parameters()).device
    logger.info(f"training with unmasked_weight={unmasked_weight}")

    bd = model.bin_dist
    logger.debug(f"clipping labels to [{bd.edges[0].item()}, {bd.edges[-1].item()}]")

    total_loss = 0.0
    steps = 0
    masked_weight = 1.0

    n_masked = 0
    total_mae = 0
    total_brier = 0
    total_wass = 0

    for batch in (pbar := tqdm.tqdm(loader, desc="train")):
        with torch.set_grad_enabled(True):
            (
                preds,
                normed_labels,
                _,
                sample_mask,
                real_assay,
                _,
                _,
                _,
            ) = _process_batch(
                batch,
                model,
                bd,
                model_device,
                is_train=True,
                mask_fraction=mask_fraction,
            )

            nll_losses = -bd.log_prob(normed_labels, preds)

            if unmasked_weight == 1.0:
                batch_loss = nll_losses.mean()
            elif unmasked_weight == 0.0:
                batch_loss = nll_losses[sample_mask].mean()
            else:
                weights = torch.full_like(nll_losses, unmasked_weight)
                weights[sample_mask] = masked_weight
                batch_loss = (nll_losses * weights).mean()

        optimizer.zero_grad(set_to_none=True)
        batch_loss.backward()
        optimizer.step()

        real_samples_mask = sample_mask & real_assay
        batch_metrics = _compute_masked_metrics(
            bd,
            preds.detach(),
            normed_labels,
            real_samples_mask,
            calculate_var_explained=False,
        )

        n_masked += batch_metrics["n_masked"]
        total_wass += batch_metrics["wass"]
        total_mae += batch_metrics["mae"]
        total_brier += batch_metrics["brier"]

        loss_value = batch_loss.detach().item()
        total_loss += loss_value
        steps += 1
        pbar.set_description(f"batch loss={loss_value:.4e}")

    return {
        "NLL": total_loss / max(1, steps),
        "EMD": total_wass / max(1, n_masked),
        "Brier": total_brier / max(1, n_masked),
        "MAE": total_mae / max(1, n_masked),
    }


@torch.no_grad()
def evaluate_with_batched_masked_sets(
    model,
    loader,
    predictions_file=None,
):
    model.eval()
    model_device = next(model.parameters()).device

    total_loss = 0.0
    total_wass = 0.0
    total_mae = 0.0
    total_brier = 0.0
    total_var_sse = 0.0
    total_var_sst = 0.0
    total_z_rho = 0.0
    total_pearson = 0.0
    num_valid_sets = 0
    n_masked = 0

    all_info = []
    all_labels = []
    all_normed = []
    all_masks = []
    all_probs = []
    save_preds = predictions_file is not None

    bd = model.bin_dist

    for batch in tqdm.tqdm(loader, desc="testing" if save_preds else "eval"):
        (
            preds,
            normed_labels,
            labels,
            sample_mask,
            _,
            info,
            set_boundaries,
            num_sets,
        ) = _process_batch(batch, model, bd, model_device, is_train=False)

        masked_preds_for_nll = preds[sample_mask]
        masked_normed_for_nll = normed_labels[sample_mask]

        nll_losses = -bd.log_prob(masked_normed_for_nll, masked_preds_for_nll)
        total_loss += nll_losses.sum().item()

        batch_metrics = _compute_masked_metrics(
            bd,
            preds,
            normed_labels,
            sample_mask,
            calculate_var_explained=True,
            set_boundaries=set_boundaries,
            num_sets=num_sets,
        )

        n_masked += batch_metrics["n_masked"]
        total_wass += batch_metrics["wass"]
        total_brier += batch_metrics["brier"]
        total_mae += batch_metrics["mae"]
        total_var_sse += batch_metrics["sse"]
        total_var_sst += batch_metrics["sst"]

        batch_z_rho, batch_num_valid_sets = _step_corr_per_set(
            spearmanr, bd, preds, normed_labels, sample_mask, set_boundaries, num_sets
        )
        batch_pearson, valid_sets = _step_corr_per_set(
            pearsonr, bd, preds, normed_labels, sample_mask, set_boundaries, num_sets
        )
        assert valid_sets == batch_num_valid_sets
        total_z_rho += batch_z_rho
        total_pearson += batch_pearson
        num_valid_sets += batch_num_valid_sets
        if save_preds:
            all_info.append(info.cpu())
            all_labels.append(labels.cpu())
            all_normed.append(normed_labels.cpu())
            all_masks.append(sample_mask.cpu())
            all_probs.append(torch.softmax(preds, dim=-1).cpu())

    avg_loss = total_loss / max(1, n_masked)
    avg_wass = total_wass / max(1, n_masked)
    avg_mae = total_mae / max(1, n_masked)
    avg_brier = total_brier / max(1, n_masked)
    avg_var_explained = 1.0 - (total_var_sse / (total_var_sst + 1e-6))
    if num_valid_sets > 0:
        avg_z_rho = total_z_rho / num_valid_sets
        avg_pearson_r = total_pearson / num_valid_sets
        avg_spearman_rho = tanh(avg_z_rho)
        avg_pearson_r = tanh(avg_pearson_r)
    else:
        logger.warning("no valid sets for correlation metrics")
        avg_spearman_rho = 0.0
        avg_pearson_r = 0.0

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

    return {
        "NLL": avg_loss,
        "EMD": avg_wass,
        "MAE": avg_mae,
        "Brier": avg_brier,
        "R2": avg_var_explained,
        "Spearman": avg_spearman_rho,
        "Pearson": avg_pearson_r,
    }


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
    logger.info(f"training model for target: {target_name}")

    opts: Dict[str, Any] = _defaults | kwargs

    logger.info("training options:")
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

    optimizer = torch.optim.Adam(model.parameters(), lr=opts["lr"])
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=opts["patience_lr"], cooldown=opts["patience_lr"],
    )

    best_loss = float("inf")
    epochs_without_improvement = 0
    optimization = []

    for epoch in range(opts["num_epochs"]):
        train_results = train_with_batched_masked_sets(
            model,
            train_loader,
            optimizer,
            unmasked_weight=opts.get("unmasked_weight", 1.0),
        )
        val_results = evaluate_with_batched_masked_sets(model, val_loader)

        results = (
            {f"train {k}": v for k, v in train_results.items()}
            | {f"val {k}": v for k, v in val_results.items()}
            | {"lr": scheduler.get_last_lr()[0]}
        )

        val_loss = val_results["NLL"]
        scheduler.step(val_loss)

        logger.info(f"epoch: {epoch + 1}")
        for metric, value in results.items():
            logger.info(f" {metric}: {value:.4e}")

        optimization.append(results)
        pd.DataFrame(optimization).to_csv(
            OUTPUT / run_name / "optimization.csv", index=False
        )

        if val_loss < best_loss:
            logger.info("validation improved, saving model.")
            best_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), OUTPUT / run_name / "model.pt")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= opts["patience_termination"]:
                logger.info(f"early stopping triggered after {epoch + 1} epochs.")
                break

    logger.info("loading best model for final test evaluation...")
    model.load_state_dict(torch.load(OUTPUT / run_name / "model.pt"))

    test_results = evaluate_with_batched_masked_sets(
        model,
        test_loader,
        predictions_file=OUTPUT / run_name / "predictions.npz",
    )
    logger.info("final test set performance:")
    for metric, value in test_results.items():
        logger.info(f" test {metric}: {value:.4e}")


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


def _generate_mask_for_sets(set_boundaries, num_sets, total_size, mask_fraction):
    sample_mask = torch.zeros(total_size, dtype=torch.bool, device=device)

    for i in range(num_sets):
        start_idx = set_boundaries[i]
        end_idx = set_boundaries[i + 1]
        set_size = end_idx - start_idx
        if set_size < 2:
            continue

        n_masked = max(1, int(set_size * mask_fraction))
        mask_idx = torch.randperm(set_size, device=device)[:n_masked]
        sample_mask[start_idx:end_idx][mask_idx] = True

    return sample_mask
