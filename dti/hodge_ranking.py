from concurrent.futures import ProcessPoolExecutor
from collections import namedtuple
import uuid
from pathlib import Path
import os

import tqdm
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .utils import DATA, ACT, SMILES

import logging

logger = logging.getLogger(__name__)

__all_scores = list()

HodgeRank = namedtuple("HodgeRank", "UniprotID smiles hodge_score".split())


def _process_target_group_from_file(args):
    input_file, inter_assay_weight = args
    group_data = pd.read_csv(input_file)
    target = group_data["UniprotID"].iloc[0]

    if inter_assay_weight == 0:
        group_data = group_data[
            group_data.groupby("assay_id")["assay_id"].transform("count") > 1
        ]

    cmpds = group_data[SMILES].unique()
    cmpd_to_idx = {cmpd: idx for idx, cmpd in enumerate(cmpds)}
    dim = len(cmpds)

    y_bar = np.zeros((dim, dim))
    weights = np.zeros((dim, dim))

    cmpd_indices = group_data[SMILES].map(cmpd_to_idx).values
    assay_ids = group_data["assay_id"].values
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

    scores = hodge_rank(y_bar, weights)

    return [HodgeRank(target, cmpd, float(score)) for cmpd, score in zip(cmpds, scores)]


def parallel_hodge_rank(kinodata: pd.DataFrame, inter_assay_weight: float = 0):
    logger.info(f"compute Hodge ranking (inter_assay_weight={inter_assay_weight})")
    global __all_scores
    __all_scores = list()

    groups = list(kinodata.groupby("UniprotID"))

    temp_dir = DATA / "hodge_temp_data" / uuid.uuid4().hex
    temp_dir.mkdir(exist_ok=True, parents=True)

    args = []
    for i, (target, tgt_data) in enumerate(groups):
        file_path = temp_dir / f"group_{i}.csv"
        tgt_data.to_csv(file_path, index=False)
        args.append((file_path, inter_assay_weight))

    with ProcessPoolExecutor(max_workers=32) as executor:
        results = list(
            tqdm.tqdm(
                executor.map(_process_target_group_from_file, args),
                total=len(args),
            )
        )

    for result in results:
        __all_scores.extend(result)

    hodge_df = pd.DataFrame(__all_scores)
    hodge_df = hodge_df.rename(columns={"smiles": SMILES})

    return hodge_df


def hodge_rank(y_bar, w, diag_stab=0.0):
    laplacian = -w
    laplacian[np.diag_indices_from(w)] = w.sum(0) + diag_stab
    y_bar = np.nan_to_num(y_bar)  # nan to zero
    divergence = (w * y_bar).sum(0)
    try:
        scores = -np.linalg.pinv(laplacian) @ divergence
        scores = StandardScaler().fit_transform(scores.reshape(-1, 1)).flatten()
        return scores
    except np.linalg.LinAlgError:
        logger.warning("unstable SVD")
        return hodge_rank(y_bar, w, diag_stab=diag_stab + 1e-5)
