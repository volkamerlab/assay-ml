from typing import Type, Any, Dict

import tqdm
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import logging
from functools import namedtuple

from .utils import device
from .constants import OUTPUT

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


def _normalize_sets_minmax(labels, sample_mask, padding_mask, clip_range=None):
    num_sets, K = sample_mask.shape
    normed_labels = torch.zeros_like(labels, device=device)

    for i in range(num_sets):
        set_labels = labels[i]
        unmasked = set_labels[~sample_mask[i] & ~padding_mask[i]]

        if unmasked.numel() < 2:
            continue

        min_val = unmasked.min()
        max_val = unmasked.max()
        range_val = (max_val - min_val).clamp_min(1e-6)
        normed = (set_labels[~padding_mask[i]] - min_val) / range_val

        if clip_range is not None:
            normed = normed.clamp(*clip_range)

        normed_labels[i][~padding_mask[i]] = normed

    return normed_labels


def _generate_mask_for_sets(padding_mask, mask_fraction):
    num_sets, k = padding_mask.shape
    set_sizes = k - padding_mask.sum(1)
    sample_mask = torch.zeros_like(padding_mask, device=device).bool()

    for i, set_size in enumerate(set_sizes):
        if set_size < 2:
            continue

        n_masked = max(1, int(set_size * mask_fraction))
        mask_idx = torch.randperm(set_size, device=device)[:n_masked]
        sample_mask[i][mask_idx] = True

    return sample_mask


def train_with_batched_masked_sets(
    model,
    loader,
    optimizer,
    mask_fraction: float = 0.2,
    unmasked_weight: float = 0.0,
    **kwargs,
):
    model.train()
    device = next(model.parameters()).device
    logger.info(f"training with unmasked_weight={unmasked_weight}")

    bd = model.bin_dist
    clip_range = (bd.edges[0], bd.edges[-1])
    logger.debug(f"clipping labels to [{clip_range[0].item()}, {clip_range[1].item()}]")

    total_loss = 0.0
    steps = 0
    masked_weight = 1.0

    for batch_idx, (
        protein_features,
        ligand_features,
        labels,
        info,
        metadata,
    ) in enumerate(pbar := tqdm.tqdm(loader, desc="training")):
        padding_mask = metadata["attention_mask"].to(device)

        ligand_features = ligand_features.to(device, non_blocking=True)
        protein_features = protein_features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        batch_size = labels.size(0)

        sample_mask = _generate_mask_for_sets(padding_mask, mask_fraction)
        assert (sample_mask.sum(1) > 0).all()

        with torch.no_grad():
            assert not torch.isnan(sample_mask).any()
            assert not torch.isnan(ligand_features).any()
            assert not torch.isnan(labels).any()
            normed_labels = _normalize_sets_minmax(
                labels,
                sample_mask,
                padding_mask,
                clip_range=clip_range,
            )
            assert not torch.isnan(normed_labels).any()

        preds = model(
            ligand_features,
            protein_features,
            normed_labels,
            sample_mask,
            padding_mask,
        )
        assert not torch.isnan(preds).any(), preds

        nll_losses = -bd.log_prob(normed_labels, preds)

        if unmasked_weight == 1.0:
            batch_loss = nll_losses.mean()
        else:
            weights = torch.full_like(nll_losses, unmasked_weight)
            weights[sample_mask] = masked_weight
            batch_loss = (nll_losses * weights).mean()

        # logger.debug(f"feats: {ligand_features.shape[1]} loss: {batch_loss.item():.3f}")

        optimizer.zero_grad(set_to_none=True)
        batch_loss.backward()
        optimizer.step()

        loss_value = batch_loss.detach().item()
        total_loss += loss_value
        steps += 1
        pbar.set_description(f"loss={loss_value:.3e}")

    return total_loss / max(1, steps)


@torch.no_grad()
def evaluate_with_batched_masked_sets(
    model,
    loader,
    mask_fraction: float = 0.2,
    predictions_file=None,
):
    model.eval()
    device = next(model.parameters()).device

    total_loss = 0.0
    total_wass = 0.0
    total_mae = 0.0
    n_masked = 0

    all_info = []
    all_labels = []
    all_normed = []
    all_masks = []
    all_probs = []
    save_preds = predictions_file is not None

    bd = model.bin_dist
    clip_range = (bd.edges[0], bd.edges[-1])
    bin_edges = bd.edges.to(device)
    bin_widths = torch.diff(bin_edges)
    total_width = (bin_edges[-1] - bin_edges[0]).clamp_min(1e-6)

    for batch_idx, (
        protein_features,
        ligand_features,
        labels,
        info,
        metadata,
    ) in enumerate(
        pbar := tqdm.tqdm(loader, desc="testing" if save_preds else "evaluating")
    ):
        padding_mask = metadata["attention_mask"].to(device)
        info = info.to(device, non_blocking=True)
        ligand_features = ligand_features.to(device, non_blocking=True)
        protein_features = protein_features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        batch_size = labels.size(0)

        sample_mask = info[:, :, 0].bool()

        normed_labels = _normalize_sets_minmax(
            labels,
            sample_mask,
            padding_mask,
            clip_range=clip_range,
        )

        preds = model(
            ligand_features,
            protein_features,
            normed_labels,
            sample_mask,
            padding_mask,
        )

        if sample_mask.any():
            masked_preds = preds[sample_mask]
            masked_normed = normed_labels[sample_mask]
            n_masked_batch = masked_preds.size(0)

            batch_loss = -bd.log_prob(masked_normed, masked_preds).sum().item()
            pbar.set_description(f"loss={batch_loss:.3e}")
            total_loss += batch_loss
            n_masked += n_masked_batch

            probs = torch.softmax(masked_preds, dim=-1)
            true_bins = bd.labels(masked_normed)
            one_hot = bd.dist(true_bins)

            cdf_pred = torch.cumsum(probs, dim=-1)
            cdf_true = torch.cumsum(one_hot, dim=-1)
            wass_dists = (torch.abs(cdf_pred - cdf_true) * bin_widths).sum(
                dim=-1
            ) / total_width
            total_wass += wass_dists.sum().item()

            pred_mean = bd.mean(masked_preds)
            true_centers = bd.bucket_centers()[true_bins]
            total_mae += (pred_mean - true_centers).abs().sum().item()

        if save_preds:
            all_info.append(info.reshape(-1, 4).cpu())
            all_labels.append(labels.flatten().cpu())
            all_normed.append(normed_labels.flatten().cpu())
            all_masks.append(sample_mask.flatten().cpu())
            all_probs.append(
                torch.softmax(preds, dim=-1).reshape(-1, preds.size(-1)).cpu()
            )

    avg_loss = total_loss / max(1, n_masked)
    avg_wass = total_wass / max(1, n_masked)
    avg_mae = total_mae / max(1, n_masked)

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

    return {"nll": avg_loss, "wasserstein": avg_wass, "mae": avg_mae}


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
    Testing is performed only once after termination (not during training).
    """
    logger.info(f"training model for target: {target_name}")
    Epoch = namedtuple(
        "Epoch",
        ["epoch", "lr", "train_loss", "val_loss"],
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

    model.bin_dist.fit(train_loader)
    with open(OUTPUT / run_name / "bin_dist", "w") as f_bins:
        f_bins.write(f"{','.join([str(x.item()) for x in model.bin_dist.edges])}")

    optimizer = torch.optim.Adam(model.parameters(), lr=opts["lr"])
    scheduler = ReduceLROnPlateau(
        optimizer, mode="max", factor=0.75, patience=opts["patience_lr"]
    )

    best_loss = float("inf")
    epochs_without_improvement = 0
    optimization = []

    for epoch in range(opts["num_epochs"]):
        train_loader.dataset._make_batches()
        train_loss = train_with_batched_masked_sets(
            model,
            train_loader,
            optimizer,
            unmasked_weight=opts.get("unmasked_weight", 1.0),
        )

        val_results = evaluate_with_batched_masked_sets(model, val_loader)
        val_loss = val_results["nll"]

        scheduler.step(val_loss)
        lr = scheduler.get_last_lr()

        logger.info(f"epoch: {epoch + 1}")
        logger.info(f" train loss: {train_loss:.4e}")
        logger.info(f" val masked NLL: {val_results['nll']:.4e}")
        logger.info(f" val masked EMD: {val_results['wasserstein']:.4e}")
        logger.info(f" val masked MAE: {val_results['mae']:.4e}")
        logger.info(f" learning rate: {lr:.2e}")

        optimization.append(Epoch(epoch, lr, train_loss, val_loss))
        pd.DataFrame(optimization).to_csv(
            OUTPUT / run_name / "optimization.csv", index=False
        )

        # Early stopping check
        if val_loss < best_loss:
            best_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), OUTPUT / run_name / "model.pt")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= opts["patience_termination"]:
                logger.info(f"early stopping triggered after {epoch + 1} epochs.")
                break

    logger.info("loading best model for final testing...")
    model.load_state_dict(torch.load(OUTPUT / run_name / "model.pt"))
    test_results = evaluate_with_batched_masked_sets(
        model,
        test_loader,
        predictions_file=OUTPUT / run_name / "predictions.npz",
    )
    logger.info("final test results:")
    logger.info(f" test masked NLL: {test_results['nll']:.4e}")
    logger.info(f" test masked EMD: {test_results['wasserstein']:.4e}")
    logger.info(f" test masked MAE: {test_results['mae']:.4e}")
