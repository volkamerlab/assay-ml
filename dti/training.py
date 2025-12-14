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
from scipy.stats import spearmanr, pearsonr

import logging
from functools import namedtuple

from .utils import device
from .constants import ASSAY, OUTPUT, ACT, COMPOUND, PREDICTION, TID
from .hodge_ranking import assay_ranks
from .data import MultiSetActivityDataset

logger = logging.getLogger(__name__)


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


class BatchPairwiseRankingLoss(nn.Module):
    def __init__(self, margin=0.1):
        super().__init__()
        self.margin = margin

    def forward(self, preds, labels, set_ids):
        pred_diff = preds.unsqueeze(0) - preds.unsqueeze(1)
        label_diff = labels.unsqueeze(0) - labels.unsqueeze(1)

        set_mask = set_ids.unsqueeze(0) == set_ids.unsqueeze(1)

        valid_pair_mask = (label_diff > 0) & set_mask

        loss_matrix = torch.relu(self.margin - pred_diff)

        masked_loss = loss_matrix * valid_pair_mask.float()

        num_pairs = valid_pair_mask.sum()
        return masked_loss.sum() / (num_pairs + 1e-8)


def train_with_batched_sets(
    model,
    loader,
    optimizer,
):
    criterion = BatchPairwiseRankingLoss(margin=0.1)

    model.train()
    torch.set_grad_enabled(True)
    total_loss = 0
    steps = 0

    pbar = tqdm.tqdm(loader, desc="training")

    for protein_features, ligand_features, labels, info, metadata in pbar:
        protein_features = protein_features.squeeze().to(device)
        ligand_features = ligand_features.squeeze().to(device)
        labels = labels.squeeze().to(device)
        set_ids = metadata["set_ids_tensor"].squeeze().to(device)

        model_kwargs = {}
        if hasattr(model, "num_heads"):
            model_kwargs["attn_mask"] = create_set_attention_mask_from_ids(
                set_ids, model.num_heads
            )

        predictions = model(protein_features, ligand_features, **model_kwargs).squeeze()

        loss = criterion(predictions, labels, set_ids)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        steps += 1
        pbar.set_description(f"loss={loss.item():.4f}")

    return total_loss / max(1, steps)


def eval_with_batched_sets(
    model,
    loader,
    criterion=None,
    prediction_file: Path | str | None = None,
):
    model.eval()
    torch.set_grad_enabled(False)

    total_loss = 0.0
    total_samples = 0

    intra_assay_spearmans = []

    all_preds, all_labels, all_info = [], [], []

    for protein_features, ligand_features, labels, info, metadata in tqdm.tqdm(
        loader, desc="Evaluating"
    ):
        protein_features = protein_features.squeeze().to(device)
        ligand_features = ligand_features.squeeze().to(device)
        labels = labels.squeeze().to(device)

        set_boundaries = metadata["set_boundaries"].squeeze()
        num_sets = metadata["num_sets"].squeeze()
        set_ids = metadata["set_ids_tensor"].squeeze().to(device)

        model_kwargs = dict()
        if hasattr(model, "num_heads"):
            model_kwargs["attn_mask"] = create_set_attention_mask_from_ids(
                set_ids, model.num_heads
            )

        predictions = model(protein_features, ligand_features, **model_kwargs).squeeze()

        if criterion is not None:
            batch_loss = criterion(predictions, labels, set_ids)
            total_loss += batch_loss.item()

        preds_np = predictions.cpu().numpy()
        labels_np = labels.cpu().numpy()

        all_preds.extend(preds_np)
        all_labels.extend(labels_np)
        if info is not None:
            all_info.append(info.squeeze().numpy())

        for i in range(num_sets):
            start_idx = set_boundaries[i].item()
            end_idx = set_boundaries[i + 1].item()

            set_p = preds_np[start_idx:end_idx]
            set_l = labels_np[start_idx:end_idx]

            if len(set_l) < 2 or np.std(set_l) < 1e-9:
                continue

            rho, _ = spearmanr(set_p, set_l)

            if not np.isnan(rho):
                intra_assay_spearmans.append(rho)

    mean_rank_corr = np.mean(intra_assay_spearmans) if intra_assay_spearmans else 0.0
    avg_loss = total_loss / len(loader)

    if prediction_file is not None:
        if len(all_info) > 0:
            all_info = np.concatenate(all_info)

        content = {"prediction": all_preds, "label": all_labels}

        if hasattr(loader.dataset, "info_cols") and len(all_info) > 0:
            for i, col in enumerate(loader.dataset.info_cols):
                if i < all_info.shape[1]:
                    content[col] = all_info[:, i]

        pd.DataFrame(content).to_csv(prediction_file, index=False)

    return avg_loss, mean_rank_corr


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

    for protein_features, ligand_features, labels, _, _ in (
        pbar := tqdm.tqdm(loader, desc="training")
    ):
        labels = labels.squeeze()

        optimizer.zero_grad()
        predictions = model(protein_features, ligand_features).squeeze()
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()

        pbar.set_description(f"loss={loss.item():.4f}")

        total_loss += loss.item()

    total_loss /= len(loader)
    return total_loss


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

    train_fn, eval_fn, criterion = opts["train_fn"], opts["eval_fn"], opts["criterion"]

    optimizer = torch.optim.AdamW(model.parameters(), lr=opts["lr"])

    scheduler = ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=opts["patience_lr"]
    )

    best_corr = -1.0
    epochs_without_improvement = 0

    for epoch in range(opts["num_epochs"]):
        train_loss = train_fn(model, train_loader, optimizer)

        val_loss, val_rank_corr = eval_fn(
            model, val_loader, criterion=criterion if opts["eval_criterion"] else None
        )

        scheduler.step(val_rank_corr)

        current_lr = scheduler.get_last_lr()[0]

        logger.info(f"Run: {run_name}")
        logger.info(f" Epoch: {epoch + 1}")
        logger.info(f" Train Loss: {train_loss:.4f}")
        logger.info(f" Val Rank Corr (Spearman): {val_rank_corr:.4f}")
        logger.info(f" LR: {current_lr:.2e}")

        if val_rank_corr > best_corr:
            logger.info(
                f"Checkpointing (corr={val_rank_corr:.4f}) in epoch {epoch + 1}."
            )
            best_corr = val_rank_corr
            epochs_without_improvement = 0
            torch.save(model.state_dict(), OUTPUT / run_name / f"model{index}.pt")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= opts["patience_termination"]:
                logger.info("Early stopping triggered.")
                break

    model.load_state_dict(torch.load(OUTPUT / run_name / f"model{index}.pt"))
    _, test_rank_corr = eval_fn(
        model,
        test_loader,
        prediction_file=OUTPUT / run_name / "predictions.csv",
    )
    logger.info(f"Final Test Intra-Assay Spearman: {test_rank_corr:.4f}")
