from typing import List, Union, Iterator, Tuple
import functools
import logging
from pathlib import Path

import pandas as pd
import numpy as np


import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

from .constants import DATA, SMILES, ACT, TID, SEQUENCE, ASSAY, COMPOUND, HODGE
from .utils import device
from .featurization import MolFingerprint, esm2_features
from .hodge_ranking import parallel_hodge_rank

logger = logging.getLogger(__name__)


class ActivityDataset(Dataset):
    """Dataset class for molecular activity data with protein and ligand features.

    Args:
        data (pd.DataFrame): DataFrame containing activity data.
        target (str): Column name for target values. Defaults to ACT.
        info_cols (List[str]): Column names to include as information. Defaults to [].
        model_name (str): Name of the protein language model. Defaults to "esm2_t33_650M_UR50D".
        n_jobs (int): Number of parallel jobs for fingerprint computation. Defaults to 16.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        mol_featurizer: MolFingerprint,
        target: str = ACT,
        info_cols: List[str] = [],
        model_name: str = "esm2_t33_650M_UR50D",
    ):
        super().__init__()
        logger.info(f"creating dataset of size {len(data)}")
        logger.info("computing fingerprints")
        fps = mol_featurizer.compute_parallel(data[SMILES].values)
        mask = [fp is not None for fp in fps]
        if len(mask) - sum(mask) > 0:
            logger.info(
                f"dropping {len(mask) - sum(mask)}/{len(mask)} data points w/o FP"
            )
        self.data = data[mask].copy()
        self.data.reset_index(inplace=True)
        self.ligand_features = torch.tensor(
            np.stack([fp for fp in fps if fp is not None]),
            dtype=torch.float32,
            device=device,
        )
        self.protein_features = esm2_features(data, model_name=model_name)
        self.labels = torch.tensor(
            self.data[target].values, dtype=torch.float32, device=device
        )
        self.info_cols = info_cols
        logger.info(f"info cols: {info_cols}")
        self.info = torch.tensor(self.data[info_cols].values)

    @functools.cached_property
    def weights(self):
        """Get sample weights for the dataset.

        Returns:
            torch.Tensor: Uniform weights for all samples.
        """
        return torch.ones(len(self.labels)).to(device)

    def __len__(self):
        """Get the number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return len(self.labels)

    def __getitem__(self, idx):
        """Get a sample from the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: Protein features, ligand features, label, and info for the sample.
        """
        prot_feats = (
            torch.ones(1).to(device)
            if self.protein_features is None
            else self.protein_features[idx]
        )
        return (
            prot_feats,
            self.ligand_features[idx],
            self.labels[idx],
            self.info[idx],
            torch.ones(1).to(device),
        )


class PairDataset(ActivityDataset):
    """Dataset for pairwise comparisons of molecular activities within the same assay.

    Args:
        data (pd.DataFrame): DataFrame containing activity data.
        **kwargs: Additional arguments passed to ActivityDataset.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        **kwargs,
    ):
        super().__init__(data, **kwargs)
        self.pairs = list()
        weights = list()
        for _, group in self.data.groupby(ASSAY):
            for i, ix0 in enumerate(group.index):
                for j, ix1 in enumerate(group.index):
                    if j > i:
                        break
                    self.pairs.append((ix0, ix1))
                    weights.append(2 / (len(group) + 1))
        self.pairs = torch.tensor(self.pairs, device=device, dtype=torch.int)
        assert len(weights) == len(self.pairs), (len(weights), len(self.pairs))
        self._weights = torch.tensor(weights, dtype=torch.double, device=device)
        self.info_cols = [col + "_a" for col in self.info_cols] + [
            col + "_b" for col in self.info_cols
        ]

    @functools.cached_property
    def weights(self):
        """Get sample weights for the dataset based on group size.

        Returns:
            torch.Tensor: Weights for each pair.
        """
        return self._weights

    def __len__(self):
        """Get the number of pairs in the dataset.

        Returns:
            int: Number of pairs.
        """
        return len(self.pairs)

    def __getitem__(self, idx):
        """Get a pair sample from the dataset.

        Args:
            idx (int): Index of the pair.

        Returns:
            tuple: Protein features, stacked ligand features, activity difference, concatenated info, and sample weights
        """
        p = self.pairs[idx]

        if self.protein_features is None:
            if not hasattr(self, "_ones_cache"):
                self._ones_cache = torch.ones(1, device=device)
            prot_feats = self._ones_cache
        else:
            prot_feats = self.protein_features[p[0]]

        if not hasattr(self, "_label_diffs"):
            self._label_diffs = -torch.diff(self.labels[self.pairs], axis=1)
        label_diff = self._label_diffs[idx]

        if not hasattr(self, "_flattened_info"):
            self._flattened_info = self.info[self.pairs.detach().cpu()].reshape(
                len(self.pairs), -1
            )
        flattened_info = self._flattened_info[idx]

        return (
            prot_feats,
            self.ligand_features[p],
            label_diff,
            flattened_info,
            self.weights[idx],
        )


class SetActivityDataset(ActivityDataset):
    """Dataset that groups samples by assay and returns batches of samples.

    Args:
        data (pd.DataFrame): DataFrame containing activity data.
        target (str): Column name for target values.
        info_cols (List[str]): Column names to include as information.
        model_name (str): Name of the protein language model. Defaults to "esm2_t33_650M_UR50D".
        min_batch_size (int): Minimum size of a batch. Defaults to 3.
        max_batch_size (int): Maximum size of a batch. For no limit use 0. Defaults to 0.
        random_seed (int): Random seed for shuffling. Defaults to 0.
    """

    def __init__(
        self,
        data,
        target=...,
        info_cols=...,
        model_name="esm2_t33_650M_UR50D",
        min_batch_size: int = 3,
        max_batch_size: int = 0,
        random_seed: int = 0,
    ):
        super().__init__(data, target, info_cols, model_name)
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.random = np.random.default_rng(random_seed)
        self.groups_index = []
        for _, group in self.data.groupby(ASSAY):
            assert group is not None
            group_idcs = np.array(group.index)
            np.random.shuffle(group_idcs)
            self.groups_index.append(group_idcs)
        self._make_batches()

    def _make_batches(self):
        """Create batches of samples from groups, respecting size constraints."""
        self.batches = []
        num_unused = 0
        for group in self.groups_index:
            if len(group) < self.min_batch_size:
                num_unused += len(group)
                continue
            if self.max_batch_size > 0 and len(group) > self.max_batch_size:
                batches = np.array_split(group, len(group) // self.max_batch_size)
            else:
                batches = [group]
            self.batches.extend(batches[:-1])
            if batches[-1].size >= self.min_batch_size:
                self.batches.append(batches[-1])
                continue
            num_unused += batches[-1].shape[0]
        self.random.shuffle(self.batches)
        self.used = np.full(len(self.batches), False, dtype=bool)
        logger.debug(f"Number of unused examples: {num_unused} / {len(self.data)}")

    def _get_next_batch(self, idx: int):
        """Get the next available batch and mark it as used.

        Args:
            idx (int): Index of the batch.

        Returns:
            np.ndarray: Indices of samples in the batch.
        """
        if np.all(self.used):
            self._make_batches()
        self.used[idx] = True
        return self.batches[idx]

    @property
    def weights(self):
        """Get sample weights for the dataset.

        Returns:
            torch.Tensor: Uniform weights for all batches.
        """
        return torch.ones(len(self), device=device)

    def __len__(self):
        """Get the number of batches in the dataset.

        Returns:
            int: Number of batches.
        """
        return len(self.batches)

    def __getitem__(self, idx):
        """Get a batch of samples from the dataset.

        Args:
            idx (int): Index of the batch.

        Returns:
            tuple: Protein features, ligand features, labels, info for the batch, and sample weight.
        """
        batch_idcs = self._get_next_batch(idx)
        prot_feats = (
            torch.ones(1, device=device)
            if self.protein_features is None
            else self.protein_features[batch_idcs]
        )
        return (
            prot_feats,
            self.ligand_features[batch_idcs],
            self.labels[batch_idcs],
            self.info[batch_idcs],
            torch.ones(1, device=device),
        )


class MultiSetActivityDataset(ActivityDataset):
    """Dataset that processes labeled sets and injects random unlabeled sets for semi-supervised learning.

    Args:
        data (pd.DataFrame): DataFrame containing activity data.
        random_sets_per_batch (int): Number of random unlabeled sets to inject into every batch.
        min_batch_size (int): Minimum size of a set.
        max_set_size (int): Maximum samples per set.
        sets_per_batch (int): Number of *labeled* sets to process in a single batch.
    """

    def __init__(
        self,
        data,
        target=...,
        info_cols=...,
        min_batch_size: int = 3,
        max_set_size: int = 0,
        sets_per_batch: int = 2,
        random_sets_per_batch: int = 16,
        random_seed: int = 0,
        **kwargs,
    ):
        super().__init__(data, target=target, info_cols=info_cols, **kwargs)
        self.min_batch_size = min_batch_size
        self.max_set_size = max_set_size if max_set_size > 0 else 128
        self.sets_per_batch = sets_per_batch
        self.random_sets_per_batch = random_sets_per_batch
        self.random = np.random.default_rng(random_seed)

        # 1. Organize Labeled Sets (The "Actual" Epoch)
        self.valid_sets = []  # List of arrays of indices
        self.set_ids = []  # List of assay/target IDs

        # Group by assay/target to form natural labeled sets
        group_col = ASSAY if ASSAY in self.data.columns else TID

        for group_id, (_, group) in enumerate(self.data.groupby(group_col)):
            group_idcs = np.array(group.index)
            # Shuffle indices within the set
            self.random.shuffle(group_idcs)

            if len(group_idcs) < self.min_batch_size:
                continue

            # Split large assays into smaller sets if necessary
            if self.max_set_size > 0 and len(group_idcs) > self.max_set_size:
                sub_batches = np.array_split(
                    group_idcs, int(np.ceil(len(group_idcs) / self.max_set_size))
                )
                for batch in sub_batches:
                    if len(batch) >= self.min_batch_size:
                        self.valid_sets.append(batch)
                        self.set_ids.append(group_id)
            else:
                self.valid_sets.append(group_idcs)
                self.set_ids.append(group_id)

        # Pre-calculate available indices for random sampling
        self.all_indices = np.arange(len(self.data))

        self._make_batches()
        logger.info(
            f"Dataset initialized: {len(self.batches)} batches. "
            f"Each batch: {self.sets_per_batch} labeled sets + {self.random_sets_per_batch} random sets."
        )

    def _make_batches(self):
        """
        Creates the batch schedule for the labeled data.
        This ensures every labeled set is seen exactly once per epoch.
        Random sets are generated dynamically in __getitem__.
        """
        indices = np.arange(len(self.valid_sets))
        self.random.shuffle(indices)

        # Reorder valid_sets based on shuffle
        self.valid_sets = [self.valid_sets[i] for i in indices]
        self.set_ids = [self.set_ids[i] for i in indices]

        self.batches = []
        self.batch_set_ids = []

        # Create batches of LABELED sets
        for i in range(0, len(self.valid_sets), self.sets_per_batch):
            end_idx = min(i + self.sets_per_batch, len(self.valid_sets))
            batch_sets = self.valid_sets[i:end_idx]
            batch_ids = self.set_ids[i:end_idx]

            self.batches.append(batch_sets)
            self.batch_set_ids.append(batch_ids)

        self.used = np.full(len(self.batches), False, dtype=bool)

    def _generate_random_set(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Constructs a random set:
        1. Picks 1 random row index to select a Protein.
        2. Picks N random row indices to select Ligands.
        Returns:
            prot_indices: Array of shape (N,) containing the SAME protein index.
            ligand_indices: Array of shape (N,) containing random ligand indices.
        """
        # 1. Pick random set size
        set_size = self.random.integers(self.min_batch_size, self.max_set_size // 8)

        # 2. Pick one random protein (by picking a random row)
        prot_source_idx = self.random.choice(self.all_indices)

        # 3. Pick N random ligands
        ligand_indices = self.random.choice(
            self.all_indices, size=set_size, replace=False
        )

        # 4. Create protein index array (broadcasted)
        # We repeat the source index so the feature lookup works in __getitem__
        prot_indices = np.full(set_size, prot_source_idx, dtype=int)

        return prot_indices, ligand_indices

    def __getitem__(self, idx):
        # Reset epoch if needed
        if np.all(self.used):
            self._make_batches()
            self.used = np.full(len(self.batches), False, dtype=bool)

        # 1. Retrieve Labeled Sets for this batch
        labeled_sets_ligands = self.batches[idx]  # List[np.array] of ligand indices
        labeled_ids = self.batch_set_ids[idx]
        self.used[idx] = True

        # Construct tuples of (prot_idx, ligand_idx, is_labeled) for labeled sets
        # For labeled sets, prot_idx == ligand_idx (same row in dataframe implies correct pairing)
        combined_sets = []
        for l_indices, s_id in zip(labeled_sets_ligands, labeled_ids):
            combined_sets.append(
                {
                    "prot_indices": l_indices,  # Use same indices to fetch paired protein
                    "ligand_indices": l_indices,
                    "is_labeled": True,
                    "set_id": s_id,
                }
            )

        # 2. Generate Random Unlabeled Sets
        for _ in range(self.random_sets_per_batch):
            r_prot_indices, r_ligand_indices = self._generate_random_set()
            combined_sets.append(
                {
                    "prot_indices": r_prot_indices,
                    "ligand_indices": r_ligand_indices,
                    "is_labeled": False,
                    "set_id": -1,  # Dummy ID
                }
            )

        # 3. Mix (Shuffle) the sets within the batch
        # This ensures the model treats them equally during batch processing
        self.random.shuffle(combined_sets)

        # 4. Flatten for Tensor Construction
        final_prot_indices = []
        final_ligand_indices = []
        final_labels = []
        final_info_indices = []  # Just to track which row info to grab

        set_boundaries = [0]
        set_labeled_flags = []
        set_ids_list = []
        batch_set_ids_tensor = []

        cumulative_count = 0

        for i, cset in enumerate(combined_sets):
            p_idxs = cset["prot_indices"]
            l_idxs = cset["ligand_indices"]
            is_labeled = cset["is_labeled"]
            count = len(l_idxs)

            final_prot_indices.append(p_idxs)
            final_ligand_indices.append(l_idxs)
            final_info_indices.append(
                l_idxs
            )  # Info usually tracks the ligand meta-data

            # For labels: Real labels if labeled, dummy zeros if unlabeled
            if is_labeled:
                final_labels.append(self.labels[l_idxs])
            else:
                final_labels.append(torch.zeros(count, dtype=self.labels.dtype))

            cumulative_count += count
            set_boundaries.append(cumulative_count)
            set_labeled_flags.append(is_labeled)
            set_ids_list.append(cset["set_id"])
            batch_set_ids_tensor.extend([i] * count)

        # Concatenate everything
        flat_prot_indices = np.concatenate(final_prot_indices)
        flat_ligand_indices = np.concatenate(final_ligand_indices)
        flat_labels = torch.cat(final_labels)
        flat_info_indices = np.concatenate(final_info_indices)

        batch_set_ids_tensor = torch.tensor(batch_set_ids_tensor, dtype=torch.long)

        # 5. Fetch Features
        # Handle Protein Features
        if self.protein_features is None:
            prot_feats = torch.ones(1, device=device)  # Fallback
        else:
            # We use the flat_prot_indices.
            # For labeled sets: index matches the ligand row.
            # For random sets: index is the randomly chosen protein source row.
            prot_feats = self.protein_features[flat_prot_indices]

        # Handle Ligand Features
        lig_feats = self.ligand_features[flat_ligand_indices]

        # Handle Info
        batch_info = self.info[flat_info_indices]

        return (
            prot_feats,
            lig_feats,
            flat_labels,
            batch_info,
            {
                "set_boundaries": torch.tensor(set_boundaries, dtype=torch.long),
                "set_ids": set_ids_list,
                "set_ids_tensor": batch_set_ids_tensor,
                "num_sets": len(combined_sets),
                "set_labeled": set_labeled_flags,  # Used by loss function to gate supervision
            },
        )

    def __len__(self):
        return len(self.batches)


def aggregate_multi_measurements(data: pd.DataFrame) -> pd.DataFrame:
    """Aggregate multiple measurements for the same compound and assay.

    Args:
        data (pd.DataFrame): DataFrame containing activity measurements.

    Returns:
        pd.DataFrame: DataFrame with aggregated measurements.
    """
    keys = [COMPOUND, ASSAY]
    if TID in data.columns:
        keys += [TID]
    non_numeric_cols = data.select_dtypes(exclude=["number"]).columns
    return (
        data.groupby(keys, as_index=False)
        .agg({ACT: "mean", **{col: lambda x: x.iloc[0] for col in non_numeric_cols}})
        .reset_index()
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


def split_kfold_by(data: pd.DataFrame, k: int, column: str, seed: int = 1) -> list:
    """Return the k-fold partitioning of `data[column]` as a list of k arrays."""
    values = data[column].unique()
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
    data: pd.DataFrame,
    target_dir: Path = DATA / "processed",
    k: int = 5,
    random_valset: bool = False,
    columns: str = [ASSAY],
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
        test_data = data[data[col].isin(partition[index])]
        test_data.to_csv(split_dir / "test.csv")
        rest = data[~data[col].isin(partition[index])]

        assert set(test_data[col]) & set(rest[col]) == set(), (
            f"Overlap found between test and rest data in fold {index}"
        )

        if random_valset:
            logger.info(f"random validation set for split {index}")
            idcs = np.arange(len(rest))
            np.random.shuffle(idcs)
            split = len(rest) // 8
            val_data = rest.iloc[idcs[:split]]
            train_data = rest.iloc[idcs[split:]]

            val_data.to_csv(split_dir / "val.csv")
            train_data.to_csv(split_dir / "train.csv")

            assert set(val_data.index) & set(train_data.index) == set(), (
                f"Overlap found between train and val indices in fold {index}"
            )

        else:
            logger.info(f"col-split validation set for split {index}")
            val_fold_idx = (index + 1) % k
            val_assays = partition[val_fold_idx][: len(partition[val_fold_idx]) // 2]

            val_data = rest[rest[col].isin(val_assays)]
            train_data = rest[~rest[col].isin(val_assays)]

            val_data.to_csv(split_dir / "val.csv")
            train_data.to_csv(split_dir / "train.csv")

            assert set(val_data[col]) & set(train_data[col]) == set(), (
                f"Overlap found between train and val data in fold {index}"
            )

        assert set(test_data[col]) & set(val_data[col]) == set(), (
            f"Overlap found between test and val data in fold {index}"
        )
        assert set(test_data[col]) & set(train_data[col]) == set(), (
            f"Overlap found between test and train data in fold {index}"
        )

    return target_dir


def load_split(
    index: int,
    data_dir: Path,
    tgt_name: str,
    inter_assay_weight: Union[float, None] = None,
    scale_scores: bool = False,
) -> Tuple[pd.DataFrame, Union[pd.DataFrame, None], pd.DataFrame, pd.DataFrame]:
    split_dir = data_dir / f"{index}"
    logger.info(f"reading dataset from {split_dir}")

    val_data = pd.read_csv(split_dir / "val.csv", index_col=0)
    train_data = pd.read_csv(split_dir / "train.csv", index_col=0)
    test_data = pd.read_csv(split_dir / "test.csv", index_col=0)

    scaler = StandardScaler()
    train_data[tgt_name] = scaler.fit_transform(train_data[ACT].values.reshape(-1, 1))
    test_data[tgt_name] = scaler.transform(test_data[ACT].values.reshape(-1, 1))
    val_data[tgt_name] = scaler.transform(val_data[ACT].values.reshape(-1, 1))

    if inter_assay_weight is not None:
        train_data = _load_hodge_ranking(
            split_dir, inter_assay_weight, train_data, scale_scores
        )

    return train_data, val_data, test_data


def _load_hodge_ranking(
    split_dir: Path,
    inter_assay_weight: float,
    train_data: pd.DataFrame,
    scale_scores: bool,
) -> pd.DataFrame:
    logger.info("reading Hodge rank data")
    hodge_file = split_dir / f"train_hodge_lam{inter_assay_weight:.2f}.csv"
    if not hodge_file.exists():
        logger.info("no cached Hodge ranking")
        hodge_df = parallel_hodge_rank(
            train_data, inter_assay_weight, scale_scores=False
        )
        merge_keys = [SMILES]
        if TID in train_data:
            merge_keys.append(TID)
        train_data = train_data.merge(
            hodge_df,
            on=merge_keys,
            how="inner",
        )
        train_data.to_csv(hodge_file)
    else:
        logger.info(f"cached Hodge ranking data at {hodge_file}")
        train_data = pd.read_csv(hodge_file, index_col=0)

    if scale_scores:
        scores = StandardScaler().fit_transform(train_data[HODGE].values.reshape(-1, 1))
        train_data[HODGE] = scores.flatten()

    return train_data


def prepare_datasets(
    data: pd.DataFrame,
    data_dir: Path,
    k: int,
    inter_assay_weight: Union[float, None] = None,
    random_valset: bool = False,
    aggregate: bool = True,
    columns: list[str] = [ASSAY],
) -> Iterator[
    Tuple[int, pd.DataFrame, Union[pd.DataFrame, None], pd.DataFrame, pd.DataFrame]
]:
    """Prepare train, validation, and test datasets."""
    logger.info(f"split along {columns}")
    if aggregate:
        data = aggregate_multi_measurements(data)
    split_data(data, data_dir, columns=columns, k=k, random_valset=random_valset)


def _process(data, col_map):
    assert all(k in data.columns for k in col_map.keys()), data.columns
    data = data.rename(columns=col_map)
    data = data[~data[SMILES].isna()]
    data = data[~data[ACT].isna()]
    if SEQUENCE in data.columns:
        data = data[~data[SEQUENCE].isna()]
    return data


def load_kinodata(
    kinodata_path: Path = DATA / "raw" / "activities-chembl33_v0.5.csv",
    activity_types: List[str] = ["pIC50"],
) -> pd.DataFrame:
    logger.info(f"loading kinodata activities from {kinodata_path}")
    data = pd.read_csv(kinodata_path, index_col=0)
    data = data[data["activities.standard_type"].isin(activity_types)]
    data = data[~data["compound_structures.canonical_smiles"].isna()]

    # strip CHEMBL prefixes
    data[ASSAY] = data["assays.chembl_id"].str[6:].astype(int)
    data[COMPOUND] = data["molecule_dictionary.chembl_id"].str[6:].astype(int)
    return _process(
        data,
        {
            "activities.standard_value": ACT,
            "compound_structures.canonical_smiles": SMILES,
            "component_sequences.sequence": SEQUENCE,
            "UniprotID": TID,
        },
    )


def load_landrum(landrum_path: Path = DATA / "raw" / "landrum.csv") -> pd.DataFrame:
    logger.info(f"loading data from {landrum_path}")
    data = pd.read_csv(landrum_path, index_col=0)
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


def load_activities(path: Path = DATA / "raw" / "activities.csv") -> pd.DataFrame:
    logger.info(f"loading activities from {path}")
    data = pd.read_csv(path)
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


def load_nci(path: Path = DATA / "raw" / "atcc.csv") -> pd.DataFrame:
    logger.info(f"loading NCI ATCC data from {path}")
    data = pd.read_csv(path, index_col=0)
    assay_ids = {exp: i for i, exp in enumerate(data["EXPID"].unique())}
    data[ASSAY] = data["EXPID"].map(assay_ids.get)
    return _process(
        data,
        {
            "NSC": COMPOUND,
            "IC50": ACT,
            "SMILES": SMILES,
        },
    )


def load_solubility(path: Path = DATA / "raw" / "solubility.csv") -> pd.DataFrame:
    logger.info("loading ChEMBL solubility data")
    data = pd.read_csv(path)
    return _process(
        data,
        {
            "molregno": COMPOUND,
            "harmonized_nM": ACT,
            "canonical_smiles": SMILES,
            "assay_id": ASSAY,
        },
    )


def load_lipo(path: Path = DATA / "raw" / "lipo.csv") -> pd.DataFrame:
    logger.info(f"loading ChEMBL data from {path}")
    data = pd.read_csv(path)
    return _process(
        data,
        {
            "molregno": COMPOUND,
            "standard_value": ACT,
            "canonical_smiles": SMILES,
            "assay_id": ASSAY,
        },
    )


def load_clearance(path: Path = DATA / "raw" / "clearance.csv") -> pd.DataFrame:
    logger.info("loading ChEMBL solubility data")
    data = pd.read_csv(path)
    return _process(
        data,
        {
            "molregno": COMPOUND,
            "value_mL_per_min_kg": ACT,
            "canonical_smiles": SMILES,
            "assay_id": ASSAY,
        },
    )
