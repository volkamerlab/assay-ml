from collections.abc import Iterator, Iterable
import logging
from pathlib import Path

import polars as pl
import numpy as np

from sklearn.preprocessing import StandardScaler

from ..utils.constants import (
    DATA,
    SMILES,
    ACT,
    TID,
    SEQUENCE,
    ASSAY,
    COMPOUND,
    HODGE,
    INTRA_ASSAY_TEST,
    IDENT,
)
from ..utils.hodge_ranking import parallel_hodge_rank

logger = logging.getLogger(__name__)


def aggregate_multi_measurements(
    data: pl.DataFrame, keys: Iterable[str] = [COMPOUND, ASSAY]
) -> pl.DataFrame:
    """Aggregate multiple measurements for the same compound and assay.

    Args:
        data (pl.DataFrame): DataFrame containing activity measurements.

    Returns:
        pl.DataFrame: DataFrame with aggregated measurements.
    """
    group_keys = list(keys)
    if TID in data.columns and TID not in group_keys:
        group_keys.append(TID)

    logger.debug(f"Aggregate multiple measurements per {group_keys}")

    return data.group_by(group_keys).agg(
        pl.col(ACT).mean(), pl.all().exclude(group_keys + [ACT]).first()
    )


def get_overlapping_keys(key_set: set[str], all_keys: np.ndarray) -> set[str]:
    """Find all keys that share components with any key in key_set."""
    components_in_set = set()
    for key in key_set:
        if key != "__DUMMY__":
            components_in_set.update(key.split("§"))

    overlapping = set()
    for key in all_keys:
        if key != "__DUMMY__":
            key_components = set(key.split("§"))
            if key_components & components_in_set:  # If there's any intersection
                overlapping.add(key)

    return overlapping


def split_kfold_by(data: pl.DataFrame, k: int, column: str, seed: int = 1) -> list:
    """Return the k-fold partitioning of `data[column]` as a list of k arrays."""
    values = data[column].unique().to_numpy()
    np.random.seed(seed)
    np.random.shuffle(values)

    fold_size = len(values) // k
    remainder = len(values) % k

    folds = []
    start_idx = 0

    for i in range(k):
        current_fold_size = fold_size + (1 if i < remainder else 0)
        end_idx = start_idx + current_fold_size
        folds.append(values[start_idx:end_idx])
        start_idx = end_idx

    return folds


def split_data(
    data: pl.DataFrame,
    target_dir: Path = DATA / "processed",
    k: int = 5,
    random_valset: bool = False,
    columns: list[str] = [ASSAY],
    random_seed: int = 1,
):
    if len(columns) != 1:
        logger.error("split along multiple columns not implemented")
        raise NotImplementedError("split along multiple columns not implemented")
    col = columns[0]

    logger.info(f"computing split along {col} and saving to {target_dir}")
    if (target_dir / "0").exists():
        return target_dir
    target_dir.mkdir(exist_ok=True, parents=True)
    partition = split_kfold_by(data, column=col, k=k, seed=random_seed)

    for index in range(k):
        split_dir = target_dir / f"{index}"
        split_dir.mkdir()

        test_mask = pl.col(col).is_in(partition[index])
        test_data = data.filter(test_mask)
        test_data.write_csv(split_dir / "test.csv")

        rest = data.filter(~test_mask)

        test_keys = set(test_data[col].unique().to_list())
        rest_keys = set(rest[col].unique().to_list())
        assert test_keys & rest_keys == set(), (
            f"Overlap found between test and rest data in fold {index}"
        )

        if random_valset:
            logger.info(f"random validation set for split {index}")
            rest_shuffled = rest.sample(fraction=1.0, seed=random_seed, shuffle=True)

            split_point = len(rest_shuffled) // 8
            val_data = rest_shuffled.slice(0, split_point)
            train_data = rest_shuffled.slice(split_point, len(rest_shuffled))

            val_data.write_csv(split_dir / "val.csv")
            train_data.write_csv(split_dir / "train.csv")
        else:
            logger.info(f"col-split validation set for split {index}")
            val_fold_idx = (index + 1) % k
            val_assays = partition[val_fold_idx][: len(partition[val_fold_idx]) // 2]

            val_mask = pl.col(col).is_in(val_assays)
            val_data = rest.filter(val_mask)
            train_data = rest.filter(~val_mask)

            val_data.write_csv(split_dir / "val.csv")
            train_data.write_csv(split_dir / "train.csv")

            val_keys = set(val_data[col].unique().to_list())
            train_keys = set(train_data[col].unique().to_list())

            assert val_keys & train_keys == set(), (
                f"Overlap found between train and val data in fold {index}"
            )

        val_keys_check = set(val_data[col].unique().to_list())
        train_keys_check = set(train_data[col].unique().to_list())

        assert test_keys & val_keys_check == set(), (
            f"Overlap found between test and val data in fold {index}"
        )
        assert test_keys & train_keys_check == set(), (
            f"Overlap found between test and train data in fold {index}"
        )

    return target_dir


def load_split(
    index: int,
    data_dir: Path,
    tgt_name: str,
    inter_assay_weight: float | None = None,
    scale_scores: bool = False,
    scale_targets: bool = True,
) -> tuple[pl.DataFrame, pl.DataFrame | None, pl.DataFrame, pl.DataFrame]:
    split_dir = data_dir / f"{index}"
    logger.info(f"reading dataset from {split_dir}")
    val_data = pl.read_csv(split_dir / "val.csv")
    train_data = pl.read_csv(split_dir / "train.csv")
    test_data = pl.read_csv(split_dir / "test.csv")

    for df in [val_data, train_data, test_data]:
        if "Unnamed: 0" in df.columns:
            df.drop_in_place("Unnamed: 0")

    if scale_targets:
        scaler = StandardScaler()
        train_vals = train_data[ACT].to_numpy().reshape(-1, 1)
        scaler.fit(train_vals)

        def transform_and_attach(df, col_name):
            vals = df[ACT].to_numpy().reshape(-1, 1)
            scaled = scaler.transform(vals).flatten()
            return df.with_columns(pl.Series(col_name, scaled))

        train_data = transform_and_attach(train_data, tgt_name)
        test_data = transform_and_attach(test_data, tgt_name)
        val_data = transform_and_attach(val_data, tgt_name)
    else:
        train_data = train_data.with_columns(pl.col(ACT).alias(tgt_name))
        test_data = test_data.with_columns(pl.col(ACT).alias(tgt_name))
        val_data = val_data.with_columns(pl.col(ACT).alias(tgt_name))

    if inter_assay_weight is not None:
        train_data = _load_hodge_ranking(
            split_dir, inter_assay_weight, train_data, scale_scores
        )

    return train_data, val_data, test_data


def _load_hodge_ranking(
    split_dir: Path,
    inter_assay_weight: float,
    train_data: pl.DataFrame,
    scale_scores: bool,
) -> pl.DataFrame:
    logger.info("reading Hodge rank data")

    hodge_file = split_dir / f"train_hodge_lam{inter_assay_weight:.2f}.csv"
    if not hodge_file.exists():
        logger.info("no cached Hodge ranking")

        train_pd = train_data.to_pandas()
        hodge_df_pd = parallel_hodge_rank(
            train_pd, inter_assay_weight, scale_scores=False
        )
        hodge_df = pl.from_pandas(hodge_df_pd)

        merge_keys = [SMILES]
        if TID in train_data.columns:
            merge_keys.append(TID)

        train_data = train_data.join(
            hodge_df,
            on=merge_keys,
            how="inner",
        )
        train_data.write_csv(hodge_file)
    else:
        logger.info(f"cached Hodge ranking data at {hodge_file}")
        cached_hodge = pl.read_csv(hodge_file)
        if "Unnamed: 0" in cached_hodge.columns:
            cached_hodge = cached_hodge.drop("Unnamed: 0")
        train_data = cached_hodge

    if scale_scores:
        scores_np = train_data[HODGE].to_numpy().reshape(-1, 1)
        scores = StandardScaler().fit_transform(scores_np).flatten()
        train_data = train_data.with_columns(pl.Series(HODGE, scores))

    return train_data


def prepare_datasets(
    data: pl.DataFrame,
    data_dir: Path,
    k: int,
    inter_assay_weight: float | None = None,
    random_valset: bool = False,
    aggregate: bool = True,
    columns: list[str] = [ASSAY],
) -> Iterator[
    tuple[int, pl.DataFrame, pl.DataFrame | None, pl.DataFrame, pl.DataFrame]
]:
    """Prepare train, validation, and test datasets."""
    logger.info(f"split along {columns}")
    if aggregate:
        data = aggregate_multi_measurements(data)
    split_data(data, data_dir, columns=columns, k=k, random_valset=random_valset)


def _process(data: pl.DataFrame, col_map: dict) -> pl.DataFrame:
    existing_cols = set(data.columns)
    assert all(k in existing_cols for k in col_map.keys()), (
        f"{data.columns} missing keys from {col_map.keys()}"
    )

    expressions = []
    for k, v in col_map.items():
        if k != v:
            expressions.append(
                pl.col(v).alias(v + "_orig") if v in existing_cols else None
            )

    # Filter out Nones from expressions list
    expressions = [e for e in expressions if e is not None]
    if expressions:
        data = data.with_columns(expressions)

    data = data.rename(col_map)

    data = data.filter(pl.col(SMILES).is_not_null())
    data = data.filter(pl.col(ACT).is_not_null())

    if SEQUENCE in data.columns:
        data = data.filter(pl.col(SEQUENCE).is_not_null())

    return data


def load_kinodata(
    kinodata_path: Path = DATA / "raw" / "activities-chembl33_v0.5.csv",
    activity_types: list[str] = ["pIC50"],
) -> pl.DataFrame:
    logger.info(f"loading kinodata activities from {kinodata_path}")
    data = pl.read_csv(kinodata_path)

    data = data.filter(pl.col("activities.standard_type").is_in(activity_types))
    data = data.filter(pl.col("compound_structures.canonical_smiles").is_not_null())

    data = data.with_columns(
        [
            pl.col("assays.chembl_id").str.slice(6).cast(pl.Int64).alias(ASSAY),
            pl.col("molecule_dictionary.chembl_id")
            .str.slice(6)
            .cast(pl.Int64)
            .alias(COMPOUND),
        ]
    )

    return _process(
        data,
        {
            "activities.activity_id": IDENT,
            "activities.standard_value": ACT,
            "compound_structures.canonical_smiles": SMILES,
            "component_sequences.sequence": SEQUENCE,
            "UniprotID": TID,
        },
    )


def load_landrum(landrum_path: Path = DATA / "raw" / "landrum.csv") -> pl.DataFrame:
    logger.info(f"loading data from {landrum_path}")
    data = pl.read_csv(landrum_path)
    return _process(
        data,
        {
            "molregno": COMPOUND,
            "pchembl_value": ACT,
            "canonical_smiles": SMILES,
            "component_sequence": SEQUENCE,
            "tid": TID,
            "assay_id": ASSAY,
        },
    )


def load_activities(path: Path = DATA / "raw" / "activities.csv") -> pl.DataFrame:
    logger.info(f"loading activities from {path}")
    data = pl.read_csv(path)
    return _process(
        data,
        {
            "molregno": COMPOUND,
            "binding_score": ACT,
            "canonical_smiles": SMILES,
            "protein_sequence": SEQUENCE,
            "uniprot_accession": TID,
            "assay_id": ASSAY,
        },
    )


def load_nci(path: Path = DATA / "raw" / "atcc.csv") -> pl.DataFrame:
    logger.info(f"loading NCI ATCC data from {path}")
    data = pl.read_csv(path)

    unique_exp = data["EXPID"].unique()
    mapping_df = pl.DataFrame({"EXPID": unique_exp, ASSAY: np.arange(len(unique_exp))})

    data = data.join(mapping_df, on="EXPID", how="left")

    return _process(
        data,
        {
            "NSC": COMPOUND,
            "IC50": ACT,
            "SMILES": SMILES,
        },
    )


def load_solubility(path: Path = DATA / "raw" / "solubility.csv") -> pl.DataFrame:
    logger.info("loading ChEMBL solubility data")
    data = pl.read_csv(path)
    return _process(
        data,
        {
            "molregno": COMPOUND,
            "harmonized_nM": ACT,
            "canonical_smiles": SMILES,
            "assay_id": ASSAY,
        },
    )


def load_lipo(path: Path = DATA / "raw" / "lipo.csv") -> pl.DataFrame:
    logger.info(f"loading ChEMBL data from {path}")
    data = pl.read_csv(path)
    return _process(
        data,
        {
            "molregno": COMPOUND,
            "standard_value": ACT,
            "canonical_smiles": SMILES,
            "assay_id": ASSAY,
        },
    )


def load_clearance(path: Path = DATA / "raw" / "clearance.csv") -> pl.DataFrame:
    logger.info("loading ChEMBL solubility data")
    data = pl.read_csv(path)
    return _process(
        data,
        {
            "molregno": COMPOUND,
            "value_mL_per_min_kg": ACT,
            "canonical_smiles": SMILES,
            "assay_id": ASSAY,
        },
    )


def load_chembl_endpoints(
    path: Path = DATA / "raw" / "chembl_endpoints_processed.csv.gz",
) -> pl.DataFrame:
    logger.info("loading general ChEMBL endpoints")
    data = pl.read_csv(path)
    assert data["compound_id"].dtype == pl.Int64
    return _process(
        data,
        {
            "compound_id": COMPOUND,
            "target_transformed": ACT,
            "canonical_smiles": SMILES,
            "group": ASSAY,
            "test": INTRA_ASSAY_TEST,
            "activity_id": IDENT,
        },
    )


def load_chembl_endpoints_protein(
    path: Path = DATA / "raw" / "chembl_endpoints_protein_processed.csv.gz",
) -> pl.DataFrame:
    logger.info("loading general ChEMBL protein endpoints")
    data = pl.read_csv(path)
    assert data["compound_id"].dtype == pl.Int64
    return _process(
        data,
        {
            "compound_id": COMPOUND,
            "target_transformed": ACT,
            "canonical_smiles": SMILES,
            "group": ASSAY,
            "test": INTRA_ASSAY_TEST,
            "activity_id": IDENT,
            "sequence": SEQUENCE,
            "uniprot_id": TID,
        },
    )
