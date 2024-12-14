from concurrent.futures import ProcessPoolExecutor
from collections import namedtuple
import uuid

import tqdm
import numpy as np
import pandas as pd
import os

from .utils import DATA, ACT, SMILES

import logging

logger = logging.getLogger(__name__)

__all_scores = list()

HodgeRank = namedtuple("HodgeRank", "UniprotID smiles hodge_score".split())


def process_target_group_from_file(input_file):
    group_data = pd.read_csv(input_file)
    target = group_data["UniprotID"].iloc[0]
    cmpds = list(group_data[SMILES].unique())
    dim = len(cmpds)
    y_bar = np.zeros((dim, dim))
    weights = np.zeros((dim, dim))

    for assay_id, subset in group_data.groupby("assay_id"):
        for i in range(len(subset)):
            for j in range(i):
                pref = subset.iloc[i][ACT] - subset.iloc[j][ACT]
                cmpd_i, cmpd_j = subset.iloc[i][SMILES], subset.iloc[j][SMILES]
                c_i, c_j = cmpds.index(cmpd_i), cmpds.index(cmpd_j)
                y_bar[c_i, c_j] += pref
                y_bar[c_j, c_i] -= pref
                weights[c_i, c_j] += 1

    weights[np.diag_indices_from(weights)] = 0
    weights += weights.T

    scores = hodge_rank(y_bar, weights)

    return [HodgeRank(target, cmpd, float(score)) for cmpd, score in zip(cmpds, scores)]


def parallel_hodge_rank(kinodata):
    logger.info("Compute Hodge ranking for all targets")
    global __all_scores
    __all_scores = list()

    groups = list(kinodata.groupby("UniprotID"))

    temp_dir = DATA / "hodge_temp_data" / uuid.uuid4().hex
    temp_dir.mkdir(exist_ok=True, prents=True)

    group_files = []
    for i, (target, tgt_data) in enumerate(groups):
        file_path = temp_dir / f"group_{i}.csv"
        tgt_data.to_csv(file_path, index=False)
        group_files.append(file_path)

    with ProcessPoolExecutor(max_workers=32) as executor:
        results = list(
            tqdm.tqdm(
                executor.map(process_target_group_from_file, group_files),
                total=len(group_files),
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
        return scores
    except np.linalg.LinAlgError:
        logger.warning("Unstable SVD")
        return hodge_rank(y_bar, w, diag_stab=diag_stab + 1e-5)
