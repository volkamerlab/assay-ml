import logging
from pathlib import Path


from dti import data
from dti.utils import butina_clusters, init_logging


def main():
    init_logging()
    logger = logging.getLogger(__name__)

    data_path = Path(".") / "data"
    raw_data_path = data_path / "raw"

    for target_file, loader, path in [
 #       ("kinodata_butina.csv", data.load_kinodata, "activities-chembl33_v0.5.csv"),
 #       ("landrum_butina.csv", data.load_landrum, "landrum.csv"),
        ("omnivore_butina.csv", data.load_landrum, "omnivore.csv"),
    ]:
        path = raw_data_path / path
        logger.info(f"dataset: {path}")
        df = loader(path)
        logger.info("computing fingerprints")
        butina_clusters(df)

        target_file = data_path / target_file
        logger.info(f"writing result to {target_file}")
        df.to_csv(target_file)


if __name__ == "__main__":
    main()
