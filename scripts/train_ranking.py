import logging
import uuid
import sys

from torch.utils.data import DataLoader

from dti.data import (
    ActivityDataset,
    prepare_datasets,
)
from dti.utils import (
    DATA,
    init_logging,
    write_header,
    train_and_evaluate_model,
)


def main():
    inter_assay_weight = float(sys.argv[1])
    run_name = f"hodge_cos_rand_valset_lam{inter_assay_weight}_" + uuid.uuid4().hex[:3]
    init_logging(run_name)
    logger = logging.getLogger("main")
    batch_size = 512

    write_header(run_name)
    data_dir = DATA / "processed_rand_valset"
    tgt_name = "scaled_ic50"

    for index, train_data, hodge_kd, val_data, test_data in prepare_datasets(
        data_dir, tgt_name, 5, logger, inter_assay_weight, True
    ):
        assert hodge_kd is not None
        info_cols = ["activities.activity_id", "assay_id"]
        val_dataset = ActivityDataset(val_data, target=tgt_name, info_cols=info_cols)
        test_dataset = ActivityDataset(test_data, target=tgt_name, info_cols=info_cols)
        train_dataset = ActivityDataset(
            hodge_kd, target="hodge_score", info_cols=info_cols
        )

        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        train_and_evaluate_model(
            run_name, train_loader, val_loader, test_loader, logger, "rank", index
        )

    logger.info("pipeline completed")


if __name__ == "__main__":
    main()
