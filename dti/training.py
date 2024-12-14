import tqdm
import pandas as pd
import numpy as np
import torch
from torch import nn
from scipy.stats import spearmanr

from .utils import DEVICE

import logging

logger = logging.getLogger(__name__)


def val_model(model, loader, criterion=nn.L1Loss()):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        all_preds = list()
        all_labels = list()
        all_info = list()
        for protein_features, ligand_features, labels, info in tqdm.tqdm(loader):
            protein_features, ligand_features, labels = (
                protein_features.to(DEVICE),
                ligand_features.to(DEVICE),
                labels.to(DEVICE),
            )
            predictions = model(protein_features, ligand_features).squeeze()
            loss = criterion(predictions, labels)
            val_loss += loss.item()

            all_preds.extend(list(predictions.cpu().numpy().flatten()))
            all_labels.extend(list(labels.cpu().numpy().flatten()))
            all_info.extend(list(info.cpu().numpy().flatten()))

    val_data = pd.DataFrame(
        {
            "prediction": all_preds,
            "label": all_labels,
            "assay": all_info,
        }
    )
    mean_rank_corr = 0
    for assay_id, group in val_data.groupby("assay"):
        if group["label"].nunique() == 1 or group["prediction"].nunique() == 1:
            continue
        rank_corr, _ = spearmanr(group["prediction"], group["label"])
        if np.isnan(rank_corr):
            continue
        mean_rank_corr += len(group) * rank_corr / len(val_data)

    val_loss /= len(loader)

    return val_loss, mean_rank_corr


def train_model(model, loader, optimizer, criterion=nn.MSELoss()):
    model.train()
    train_loss = 0
    for protein_features, ligand_features, labels, _ in tqdm.tqdm(loader):
        protein_features, ligand_features, labels = (
            protein_features.to(DEVICE),
            ligand_features.to(DEVICE),
            labels.to(DEVICE),
        )

        optimizer.zero_grad()
        predictions = model(protein_features, ligand_features).squeeze()
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
    train_loss /= len(loader)
    return train_loss
