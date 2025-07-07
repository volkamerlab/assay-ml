import logging
from pathlib import Path
from functools import namedtuple

import pandas as pd
import numpy as np

import tqdm.auto as tqdm

from dti import data
from dti.data import split_data
from dti.utils import butina_clusters
from dti.constants import ACT, ASSAY


def main():
    init_logging()
    logger = logging.getLogger(__name__)

    data_path = Path(".") / "data"
    raw_data_path = data_path / "raw"

    for target_file, df in [
        (
            "kinodata_butina.csv",
            data.load_kinodata(raw_data_path / "activities-chembl33_v0.5.csv"),
        ),
        ("landrum_butina.csv", data.load_landrum(raw_data_path / "landrum.csv")),
        ("omnivore_butina.csv", data.load_landrum(raw_data_path / "omnivore.csv")),
    ]:
        logger.info("computing fingerprints")
        butina_clusters(df)

        target_file = data_path / target_file
        logger.info(f"writing result to {target_file}")
        df.to_csv(target_file)


if __name__ == "__main__":
    main()
