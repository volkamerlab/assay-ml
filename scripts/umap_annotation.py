import logging
import numpy as np
from pathlib import Path
from multiprocessing import Pool
from dti.utils import compute_fp

import umap
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph

from dti import data
from dti.utils import init_logging
from dti.constants import SMILES


def main():
    init_logging()
    logger = logging.getLogger(__name__)

    data_path = Path(".") / "data"
    raw_data_path = data_path / "raw"

    for target_file, df in [
        (
            "kinodata_umap.csv",
            data.load_kinodata(raw_data_path / "activities-chembl33_v0.5.csv"),
        ),
        ("landrum_umap.csv", data.load_landrum(raw_data_path / "landrum.csv")),
        ("omnivore_umap.csv", data.load_landrum(raw_data_path / "omnivore.csv")),
    ]:
        logger.info("computing fingerprints")
        with Pool(8) as p:
            fp_list = p.map(compute_fp, df[SMILES].values)

        df = df[[fp is not None for fp in fp_list]]

        logger.info("dimensionality reduction - PCA")
        pca = PCA(n_components=15)
        fp_list = [fp for fp in fp_list if fp is not None]
        pcs = pca.fit_transform(np.stack(fp_list))

        logger.info("dimensionality reduction - UMAP")
        reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1)
        embedding = reducer.fit_transform(pcs)
        df["umap_1"] = embedding[:, 0]
        df["umap_2"] = embedding[:, 1]

        target_file = data_path / target_file
        logger.info(f"writing result to {target_file}")
        df.to_csv(target_file)

        logger.info("KNN graph")
        connectivity = kneighbors_graph(embedding, n_neighbors=10, include_self=False)

        logger.info("clustering")
        ac = AgglomerativeClustering(
            n_clusters=5,
            connectivity=connectivity,
            linkage="ward",
        )
        ac = AgglomerativeClustering(n_clusters=5, compute_full_tree=False)
        ac.fit_predict(embedding)

        df["cluster"] = ac.labels_

        target_file = data_path / target_file
        logger.info(f"writing result to {target_file}")
        df.to_csv(target_file)


if __name__ == "__main__":
    main()
