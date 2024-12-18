import tqdm
import pandas as pd
import numpy as np
import torch
from torch import nn
from scipy.stats import spearmanr, kendalltau
from scipy.special import binom

import logging

logger = logging.getLogger(__name__)

device = lambda: "cuda" if torch.cuda.is_available() else "cpu"

def model_epoch(model, loader, optimizer=None, criterion=None, prediction_file=None):
    if optimizer is not None:
        logger.debug('train model')
        criterion = nn.MSELoss() if criterion is None else criterion
        model.train()
    else:
        logger.debug('evaluate model')
        criterion = nn.L1Loss() if criterion is None else criterion
        model.eval()
    torch.set_grad_enabled(optimizer is not None)

    total_loss = 0
    all_preds = list()
    all_labels = list()
    all_info = list()
    for protein_features, ligand_features, labels, info in tqdm.tqdm(loader):
        protein_features, ligand_features, labels = (
            protein_features.to(device()),
            ligand_features.to(device()),
            labels.to(device()),
        )
        if optimizer is not None:
            optimizer.zero_grad()
        predictions = model(protein_features, ligand_features).squeeze()
        loss = criterion(predictions, labels)
        if optimizer is not None:
            loss.backward()
            optimizer.step()
        total_loss += loss.item()

        all_preds.extend(list(predictions.detach().cpu().numpy().flatten()))
        all_labels.extend(list(labels.detach().cpu().numpy().flatten()))
        all_info.append(info.detach().cpu().numpy())

    torch.set_grad_enabled(True)

    content = {
        "prediction": all_preds,
        "target": all_labels,
    }
    all_info = np.concat(all_info)
    for i, col in enumerate(loader.dataset.info_cols):
        content[col] = list(all_info[:, i].flatten())
    prediction_data = pd.DataFrame(content)
    if prediction_file is not None:
        logger.info(f'writing predictions to {prediction_file}')
        prediction_data.to_csv(prediction_file)
    if "assay_id" in prediction_data.columns:
        mean_rank_corr = rank_corr(prediction_data)

    total_loss /= len(loader)

    return total_loss, mean_rank_corr


def rank_corr(prediction_data: pd.DataFrame) -> float:
    overall_tau = 0
    total_weight = 0
    for assay_id, group in prediction_data.groupby("assay_id"):
        if group["target"].nunique() == 1 or group["prediction"].nunique() == 1:
            continue
        assay_weight = binom(len(group), 2)
        tau = kendalltau(
            group["prediction"], group["target"], nan_policy="raise", variant="c"
        ).statistic
        if np.isnan(tau):
            continue
        total_weight += assay_weight
        overall_tau += assay_weight * tau
    overall_tau /= total_weight

    return overall_tau
