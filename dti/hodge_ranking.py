import tempfile
from typing import Tuple
from concurrent.futures import ProcessPoolExecutor
from collections import namedtuple
import itertools as itt

import tqdm
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .constants import ACT, SMILES, TID, ASSAY, COMPOUND, HODGE, PREDICTION

import logging

logger = logging.getLogger(__name__)

HodgeRank = namedtuple("HodgeRank", [TID, SMILES, HODGE])


def _build_matrices(
    cmpd_indices: np.ndarray,
    assay_ids: np.ndarray,
    activity: np.ndarray,
    dim: int,
    inter_assay_weight: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Optimized construction of y_bar and weights matrices.

    Args:
        cmpd_indices  (np.ndarray): Array of compound indices
        assay_ids  (np.ndarray): Array of assay IDs
        activity  (np.ndarray): Array of activity values
        dim  (int): Dimension of matrices (number of unique compounds)
        inter_assay_weight  (float): Weight for inter-assay comparisons

    Returns:
        Tuple[np.ndarray, np.ndarray]: y_bar and weights matrices
    """
    y_bar = np.zeros((dim, dim))
    weights = np.zeros((dim, dim))

    for i, j in itt.combinations(range(dim), 2):
        c_i, c_j = cmpd_indices[i], cmpd_indices[j]
        weight = inter_assay_weight if assay_ids[i] != assay_ids[j] else 1.0
        pref = activity[i] - activity[j]

        y_bar[c_i, c_j] += weight * pref
        y_bar[c_j, c_i] -= weight * pref
        weights[c_i, c_j] += weight

    np.fill_diagonal(weights, 0)
    weights = weights + weights.T

    return y_bar, weights


def _process_target_group_from_file(args):
    """
    Process target group data from file to compute Hodge rank scores.

    Args:
        args (tuple): Tuple containing (input_file, inter_assay_weight, scale_scores)

    Returns:
        List[HodgeRank]: List of HodgeRank tuples with compounds and their scores
    """
    input_file, inter_assay_weight, scale_scores = args

    group_data = pd.read_csv(input_file, engine="c", low_memory=True)
    usecols = [SMILES, ASSAY, ACT]
    if TID in group_data.columns:
        usecols.append(TID)
    target = group_data[TID].iloc[0] if TID in group_data.columns else None

    if inter_assay_weight == 0:
        group_data = group_data[group_data.groupby(ASSAY)[ASSAY].transform("count") > 1]

    if len(group_data) <= 1:
        return []

    cmpds = group_data[SMILES].unique()
    dim = len(cmpds)

    if dim <= 1:
        return []

    cmpd_to_idx = {c: i for i, c in enumerate(cmpds)}
    cmpd_indices = group_data[SMILES].map(cmpd_to_idx).values
    assay_indices, _ = pd.factorize(group_data[ASSAY])
    activity = group_data[ACT].values

    y_bar, weights = _build_matrices(
        cmpd_indices, assay_indices, activity, dim, inter_assay_weight
    )

    scores = hodge_rank(y_bar, weights, scale_scores=scale_scores)

    return [
        HodgeRank(target, cmpd, float(score))
        for cmpd, score in zip(cmpds, scores)
        if not np.isnan(score)
    ]


def parallel_hodge_rank(
    data: pd.DataFrame,
    inter_assay_weight: float = 0,
    scale_scores: bool = False,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """
    Compute Hodge ranking for targets as specified by the `TID` column in parallel.

    Args:
        data (pd.DataFrame): Affinity data
        inter_assay_weight (float, optional): Inter-assay preference weights in [0,1]
        scale_scores (bool, optional): Apply Z-transform to Hodge potentials
        n_jobs (int, optional): Number of jobs

    Returns:
        pd.DataFrame: data with hodge potential in `HODGE` column
    """
    logger.info(f"compute Hodge ranking (inter_assay_weight={inter_assay_weight})")
    all_scores = list()

    groups = [g for _, g in data.groupby(TID)] if TID in data.columns else [data]

    args = []
    for tgt_data in groups:
        with tempfile.NamedTemporaryFile(delete=False) as fp:
            tgt_data.to_csv(fp, index=False)
        args.append((fp.name, inter_assay_weight, scale_scores))

    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        results = list(
            tqdm.tqdm(
                executor.map(_process_target_group_from_file, args),
                total=len(args),
            )
        )

    for result in results:
        all_scores.extend(result)

    hodge_df = pd.DataFrame(all_scores)
    hodge_df = hodge_df.rename(columns={"smiles": SMILES})

    return hodge_df


def assay_ranks(preds: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Hodge ranking for pairwise predictions over the whole dataframe.

    Args:
        data (pd.DataFrame): Prediction data

    Returns:
        pd.DataFrame: dataframe with compounds and Hodge potentials
    """
    cmpd_a = COMPOUND + "_a"
    cmpd_b = COMPOUND + "_b"
    unique_cmpds = np.unique(
        np.concat(
            [
                preds[cmpd_a].values,
                preds[cmpd_b].values,
            ]
        )
    )
    cmpd_to_idx = {cmpd: idx for idx, cmpd in enumerate(unique_cmpds)}
    dim = len(unique_cmpds)

    if dim <= 1:
        return 0

    y_bar = np.zeros((dim, dim))
    weights = np.zeros((dim, dim))

    for _, row in preds.iterrows():
        c_i = cmpd_to_idx[row[cmpd_a]]
        c_j = cmpd_to_idx[row[cmpd_b]]
        pref = row[PREDICTION]
        y_bar[c_i, c_j] += pref
        y_bar[c_j, c_i] -= pref
        weights[c_i, c_j] += 1

    weights[np.diag_indices_from(weights)] = 0
    weights += weights.T

    scores = hodge_rank(y_bar, weights)

    return pd.DataFrame({COMPOUND: unique_cmpds, PREDICTION: scores})


def hodge_rank(
    y_bar: np.ndarray, w: np.ndarray, diag_stab: float = 0.0, scale_scores: bool = False
):
    laplacian = -w.copy()
    laplacian[np.diag_indices_from(w)] = w.sum(0) + diag_stab
    y_bar = np.nan_to_num(y_bar)
    divergence = (w * y_bar).sum(0)
    try:
        scores = -np.linalg.pinv(laplacian) @ divergence
        if scale_scores:
            scores = StandardScaler().fit_transform(scores.reshape(-1, 1)).flatten()
        return scores
    except np.linalg.LinAlgError:
        new_diag_stab = max(1e-6, diag_stab) * 10
        logger.warning(
            f"unstable SVD ({str(e)}); retrying with diag_stab={new_diag_stab}"
        )
        return hodge_rank(y_bar, w, diag_stab=new_diag_stab, scale_scores=scale_scores)
