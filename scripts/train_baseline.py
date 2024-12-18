import logging
import uuid
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

from dti.data import (
    ActivityDataset,
    prepare_dataset,
)
from dti.utils import ACT, DEVICE, DATA, OUTPUT, init_logging, write_info, write_header
from dti.model import CombinedModel
from dti.training import model_epoch


def main():
    run_name = f"baseline_" + uuid.uuid4().hex[:5]
    init_logging(run_name)
    logger = logging.getLogger("main")
    batch_size = 256
    
    write_header(run_name)
    data_dir = DATA / "processed"
    tgt_name = "scaled_ic50"

    for index, train_data, hodge_kd, val_data, test_data in prepare_datasets(
        data_dir, tgt_name, 5, logger,None
    ):
        info_cols = ["activities.activity_id", "assay_id"]
        train_dataset = ActivityDataset(
            train_data, target=tgt_name, info_cols=info_cols
        )
        val_dataset = ActivityDataset(val_data, target=tgt_name, info_cols=info_cols)
        test_dataset = ActivityDataset(test_data, target=tgt_name, info_cols=info_cols)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        train_and_evaluate_model(
            run_name, train_loader, val_loader, test_loader, logger, "ic50", index
        )

    logger.info("pipeline completed")


if __name__ == "__main__":
    main()
