import logging
import uuid
import sys

from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

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
    if inter_assay_weight < 0:
        inter_assay_weight = None
    run_name = f"cos_rvs_lam{inter_assay_weight}_" + uuid.uuid4().hex[:3]
    init_logging(run_name)
    logger = logging.getLogger("main")
    batch_size = 512

    write_header(run_name)
    data_dir = DATA / "processed"
    tgt_name = "scaled_ic50"

    for index, train_data, val_data, test_data in prepare_datasets(
        data_dir, tgt_name, 5, inter_assay_weight, True
    ):
        info_cols = ["activities.activity_id", "assay_id"]
        val_dataset = ActivityDataset(val_data, target=tgt_name, info_cols=info_cols)
        test_dataset = ActivityDataset(test_data, target=tgt_name, info_cols=info_cols)
        scaler = StandardScaler()
        train_tgt_name = "hodge_score" if inter_assay_weight is not None else tgt_name
        train_data[train_tgt_name] = scaler.fit_transform(
                train_data[train_tgt_name].values.reshape(-1, 1)
        )
        train_dataset = ActivityDataset(
            train_data, target=train_tgt_name, info_cols=info_cols
        )

        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        train_and_evaluate_model(
            run_name, train_loader, val_loader, test_loader, logger, "rank", index,
            cosine_agg=True,
        )

    logger.info("pipeline completed")


if __name__ == "__main__":
    main()
