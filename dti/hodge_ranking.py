import tempfile
from concurrent.futures import ProcessPoolExecutor
from collections import namedtuple

import tqdm
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import scipy.sparse
import scipy.sparse.linalg

from .constants import ACT, SMILES, TID, ASSAY, COMPOUND

import logging

logger = logging.getLogger(__name__)

HodgeRank = namedtuple("HodgeRank", [TID, SMILES, "hodge_screen"])


def _process_target_group_from_file(args):
    input_file, inter_assay_weight, scale_scores = args
    group_data: pd.DataFrame = pd.read_csv(input_file)
    target = group_data[TID].iloc[0] if TID in group_data.columns else None

    if inter_assay_weight == 0:
        group_data = group_data[group_data.groupby(ASSAY)[ASSAY].transform("count") > 1]

    cmpds = group_data[SMILES].unique()
    cmpd_to_idx = {cmpd: idx for idx, cmpd in enumerate(cmpds)}
    dim = len(cmpds)

    if dim <= 1:
        return []

    y_bar = np.zeros((dim, dim))
    weights = np.zeros((dim, dim))

    cmpd_indices = group_data[SMILES].map(cmpd_to_idx).values
    assay_ids = group_data[ASSAY].values
    activity = group_data[ACT].values

    for i in range(len(group_data)):
        c_i = cmpd_indices[i]
        for j in range(i):
            c_j = cmpd_indices[j]
            weight = inter_assay_weight if assay_ids[i] != assay_ids[j] else 1.0
            pref = activity[i] - activity[j]
            y_bar[c_i, c_j] += weight * pref
            y_bar[c_j, c_i] -= weight * pref
            weights[c_i, c_j] += weight

    weights[np.diag_indices_from(weights)] = 0
    weights += weights.T

    scores = hodge_rank(y_bar, weights, scale_scores=scale_scores)

    return [HodgeRank(target, cmpd, float(score)) for cmpd, score in zip(cmpds, scores)]


def parallel_hodge_rank(
    data: pd.DataFrame, inter_assay_weight: float = 0, scale_scores: bool = False
) -> pd.DataFrame:
    logger.info(f"compute Hodge ranking (inter_assay_weight={inter_assay_weight})")
    all_scores = list()

    groups = [g for _, g in data.groupby(TID)] if TID in data.columns else [data]

    args = []
    for tgt_data in groups:
        with tempfile.NamedTemporaryFile(delete=False) as fp:
            tgt_data.to_csv(fp, index=False)
        args.append((fp.name, inter_assay_weight, scale_scores))

    with ProcessPoolExecutor(max_workers=32) as executor:
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


def assay_ranks(preds: pd.DataFrame):
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
        pref = row["prediction"]
        y_bar[c_i, c_j] += pref
        y_bar[c_j, c_i] -= pref
        weights[c_i, c_j] += 1

    weights[np.diag_indices_from(weights)] = 0
    weights += weights.T

    scores = hodge_rank(y_bar, weights)

    return pd.DataFrame({COMPOUND: unique_cmpds, "prediction": scores})


#
#
# def hodge_rank(
#     y_bar: np.ndarray, w: np.ndarray, diag_stab: float = 0.0, scale_scores: bool = False
# ):
#     laplacian = -w
#     laplacian[np.diag_indices_from(w)] = w.sum(0) + diag_stab
#     y_bar = np.nan_to_num(y_bar)  # nan to zero
#     divergence = (w * y_bar).sum(0)
#     try:
#         scores = -np.linalg.pinv(laplacian) @ divergence
#         if scale_scores:
#             scores = StandardScaler().fit_transform(scores.reshape(-1, 1)).flatten()
#         return scores
#     except np.linalg.LinAlgError:
#         logger.warning(f"unstable SVD (diag_stab={diag_stab})")
#         return hodge_rank(y_bar, w, diag_stab=diag_stab + 1e-5)
#


def hodge_rank(
    y_bar: np.ndarray,
    w: np.ndarray,
    diag_stab: float = 1e-5,
    scale_scores: bool = False,
):
    w = np.asarray(w, dtype=np.float64)
    y_bar = np.nan_to_num(y_bar, copy=False)

    laplacian = -w.copy()
    np.fill_diagonal(laplacian, w.sum(axis=0) + diag_stab)

    divergence = np.einsum("ij,ij->j", w, y_bar)

    try:
        scores, *_ = scipy.sparse.linalg.lsmr(
            laplacian, -divergence, atol=1e-10, btol=1e-10, maxiter=1000
        )

        if scale_scores:
            scores = StandardScaler().fit_transform(scores.reshape(-1, 1)).flatten()

        return scores

    except np.linalg.LinAlgError:
        logger.warning(f"unstable SVD (diag_stab={diag_stab})")
        return hodge_rank(y_bar, w, diag_stab=diag_stab * 10, scale_scores=scale_scores)
