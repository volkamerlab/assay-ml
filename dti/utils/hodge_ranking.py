import os
from typing import Tuple, List
from concurrent.futures import ProcessPoolExecutor
from collections import namedtuple
import itertools as itt

import tqdm
import numpy as np
import polars as pl  # Changed from pandas
from sklearn.preprocessing import StandardScaler

# Assuming constants are accessible in the Polars context
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
    y_bar = np.zeros((dim, dim))
    weights = np.zeros((dim, dim))

    data_dim = len(cmpd_indices)
    for i, j in itt.combinations(range(data_dim), 2):
        c_i, c_j = cmpd_indices[i], cmpd_indices[j]
        weight = inter_assay_weight if assay_ids[i] != assay_ids[j] else 1.0
        pref = activity[i] - activity[j]

        y_bar[c_i, c_j] += weight * pref
        y_bar[c_j, c_i] -= weight * pref
        weights[c_i, c_j] += weight

    np.fill_diagonal(weights, 0)
    weights = weights + weights.T

    return y_bar, weights


def _rank_target(
    group_data: pl.DataFrame, inter_assay_weight: float, scale_scores: bool
) -> List[HodgeRank]:
    target = group_data[TID][0] if TID in group_data.columns else None

    if inter_assay_weight == 0:
        group_counts = group_data.group_by(ASSAY).agg(pl.count().alias("count"))
        group_data = (
            group_data.join(group_counts, on=ASSAY, how="left")
            .filter(pl.col("count") > 1)
            .drop("count")
        )

    if len(group_data) <= 1:
        return []

    cmpds = group_data[SMILES].unique().to_list()
    dim = len(cmpds)

    if dim <= 1:
        return []

    cmpd_to_idx = {c: i for i, c in enumerate(cmpds)}

    cmpd_indices = group_data.with_columns(
        pl.col(SMILES).map_dict(cmpd_to_idx).alias(COMPOUND)
    )[COMPOUND].to_numpy()

    assay_series = group_data[ASSAY]
    assay_indices = assay_series.to_physical().to_numpy()

    activity = group_data[ACT].to_numpy()

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
    data: pl.DataFrame,
    inter_assay_weight: float = 0,
    scale_scores: bool = False,
    n_jobs: int = min(os.cpu_count() - 2, 32),
) -> pl.DataFrame:
    logger.info(
        f"compute Hodge ranking (inter_assay_weight={inter_assay_weight}, scale_scores={scale_scores})"
    )

    groups = [g for g in data.group_by(TID).groups] if TID in data.columns else [data]

    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = [
            executor.submit(
                _rank_target,
                tgt_data,
                inter_assay_weight,
                scale_scores,
            )
            for tgt_data in groups
        ]

        futures = tqdm.tqdm(futures, total=len(futures), desc="Ranking")
        results = [future.result() for future in futures]

    flat_results = sum(results, start=[])
    result_df = pl.DataFrame(flat_results)

    if "smiles" in result_df.columns:
        result_df = result_df.rename({"smiles": SMILES})

    return result_df


def assay_ranks(
    preds: pl.DataFrame, suffixes: tuple[str, str] = ("_a", "_b")
) -> pl.DataFrame:
    cmpd_a = COMPOUND + suffixes[0]
    cmpd_b = COMPOUND + suffixes[1]

    cmpds = (
        pl.concat(
            [
                preds.select(cmpd_a).unique().rename({cmpd_a: COMPOUND}),
                preds.select(cmpd_b).unique().rename({cmpd_b: COMPOUND}),
            ]
        )[COMPOUND]
        .unique()
        .to_list()
    )

    cmpd_to_idx = {cmpd: idx for idx, cmpd in enumerate(cmpds)}
    dim = len(cmpds)

    if dim <= 1:
        return pl.DataFrame({COMPOUND: cmpds, PREDICTION: np.zeros_like(cmpds)})

    y_bar = np.zeros((dim, dim))
    weights = np.zeros((dim, dim))

    for row in preds.iter_rows(named=True):
        c_i = cmpd_to_idx[row[cmpd_a]]
        c_j = cmpd_to_idx[row[cmpd_b]]
        y_bar[c_i, c_j] += row[PREDICTION]
        weights[c_i, c_j] += 1

    weights[np.diag_indices_from(weights)] = 0
    weights += weights.T
    y_bar -= y_bar.T

    scores = hodge_rank(y_bar, weights)

    return pl.DataFrame({COMPOUND: cmpds, PREDICTION: scores})


def hodge_rank(
    y_bar: np.ndarray, w: np.ndarray, diag_stab: float = 0.0, scale_scores: bool = True
) -> np.ndarray:
    laplacian = -w.copy()
    laplacian[np.diag_indices_from(w)] = w.sum(0) + diag_stab
    y_bar = np.nan_to_num(y_bar)
    divergence = (w * y_bar).sum(0)
    try:
        scores = -np.linalg.pinv(laplacian) @ divergence
        if scale_scores:
            scores = StandardScaler().fit_transform(scores.reshape(-1, 1)).flatten()
        return scores
    except np.linalg.LinAlgError as e:
        new_diag_stab = max(1e-6, diag_stab) * 10
        logger.warning(
            f"unstable SVD ({str(e)}); retrying with diag_stab={new_diag_stab}"
        )
        return hodge_rank(y_bar, w, diag_stab=new_diag_stab, scale_scores=scale_scores)
