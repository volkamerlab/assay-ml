import logging
import uuid
import sys
import random

import numpy as np
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
import torch

from dti.model import PairModel, CombinedModel
from dti.data import (
    ActivityDataset,
    PairDataset,
    prepare_datasets,
    load_landrum,
    load_kinodata,
)
from dti.utils import (
    init_logging,
    set_random_seeds,
    write_header,
    train_and_evaluate_model,
)

from dti.constants import DATA


def main():
    seed = int(sys.argv[1])
    set_random_seeds(seed)

    dataset = sys.argv[2]
    match dataset:
        case "kinodata":
            data = load_kinodata()
            info_cols = ["activities.activity_id", "assay_id"]
        case "landrum":
            data = load_landrum()
            info_cols = ["activity_id", "assay_id"]
        case "landrum_large":
            data = load_landrum(DATA / "raw" / "landrum_large.csv")
            info_cols = ["activity_id", "assay_id"]
        case _:
            print(f"Unknown dataset: {dataset}", file=sys.stderr)
            sys.exit(1)

    method = sys.argv[3]
    match method:
        case "pair":
            model_cls = PairModel
            dataset_cls = PairDataset
            num_epochs = 100
        case "ic50":
            model_cls = CombinedModel
            dataset_cls = ActivityDataset
            num_epochs = 500
        case _:
            print(f"Unknown method: {method}", file=sys.stderr)
            sys.exit(1)

    run_name = f"{dataset}_{method}_" + uuid.uuid4().hex[:4]
    init_logging(run_name)
    logger = logging.getLogger(run_name)

    batch_size = 512

    write_header(run_name)
    data_dir = DATA / "processed" / dataset
    tgt_name = "scaled_ic50"

    for index, train_data, val_data, test_data in prepare_datasets(
        data, data_dir, tgt_name, 5, None, False
    ):
        val_dataset = dataset_cls(val_data, target=tgt_name, info_cols=info_cols)
        test_dataset = dataset_cls(test_data, target=tgt_name, info_cols=info_cols)
        scaler = StandardScaler()
        train_data[tgt_name] = scaler.fit_transform(
            train_data[tgt_name].values.reshape(-1, 1)
        )
        train_dataset = dataset_cls(train_data, target=tgt_name, info_cols=info_cols)

        # https://pytorch.org/docs/stable/notes/randomness.html
        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        g = torch.Generator()
        g.manual_seed(seed + index)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            worker_init_fn=seed_worker,
            generator=g,
        )
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        train_and_evaluate_model(
            model_cls,
            run_name,
            train_loader,
            val_loader,
            test_loader,
            logger,
            method,
            index,
            cosine_agg=True,
        )

    logger.info(f"{run_name} completed")


if __name__ == "__main__":
    main()
