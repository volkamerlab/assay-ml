from collections.abc import Iterator, Iterable
import functools
import logging
from pathlib import Path

import pandas as pd
import numpy as np


import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

from .constants import (
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
        info_cols: list[str] = [],
        model_name: str = "esm2_t33_650M_UR50D",
        **kwargs,
    ):
        super().__init__()
        logger.info(f"creating dataset of size {len(data)}")
        # for line in str(data.dtypes).split("\n"):
        #     logger.debug(line)
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
            device="cpu",
        )
        self.protein_features = esm2_features(data, model_name=model_name)
        self.labels = torch.tensor(
            self.data[target].values, dtype=torch.float32, device="cpu"
        )
        missing_info_cols = [c for c in info_cols if c not in data.columns]
        if len(missing_info_cols) > 0:
            logger.warn(f"missing info cols: {missing_info_cols}")
            info_cols = [c for c in info_cols if c not in missing_info_cols]
        logger.info(f"info cols: {info_cols}")
        self.info_cols = info_cols
        self.info = torch.tensor(
            self.data[info_cols].values.astype(np.int64), device="cpu"
        )

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
        **kwargs,
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
    """Dataset that processes multiple sets in a single batch while preserving set identity.

    Args:
        data (pd.DataFrame): DataFrame containing activity data.
        target (str): Column name for target values.
        info_cols (List[str]): Column names to include as information.
        inter_assay (bool): Compute inter-assay sets (same target).
        shuffle_within_target (bool): Retain target for inter-assay sets.
        model_name (str): Name of the protein language model.
        min_batch_size (int): Minimum size of a set to be included.
        max_set_size (int): Maximum samples per set (0 for no limit).
        sets_per_batch (int): Number of sets to process in a single batch.
        random_seed (int): Random seed for shuffling.
    """

    def __init__(
        self,
        data,
        target=...,
        info_cols=...,
        inter_assay: bool = False,
        shuffle_within_target: bool = True,
        min_batch_size: int = 3,
        max_set_size: int = 0,
        sets_per_batch: int = 20,
        random_seed: int = 0,
        **kwargs,
    ):
        super().__init__(data, target=target, info_cols=info_cols, **kwargs)
        self.min_batch_size = min_batch_size
        self.max_set_size = max_set_size
        self.sets_per_batch = sets_per_batch
        self.inter_assay = inter_assay
        self.shuffle_within_target = shuffle_within_target
        self.random = np.random.default_rng(random_seed)

        num_groups = self.data[ASSAY].nunique()
        self.valid_sets = []
        self.set_ids = []
        num_unused = 0

        grouped = self.data.groupby(ASSAY, sort=False)

        for assay_id, (_, group) in enumerate(grouped):
            group_idcs = group.index.values  # Direct numpy array conversion
            self.random.shuffle(group_idcs)

            group_size = len(group_idcs)
            if group_size < self.min_batch_size:
                num_unused += group_size
                continue

            if self.max_set_size > 0 and group_size > self.max_set_size:
                num_splits = (group_size + self.max_set_size - 1) // self.max_set_size
                sub_batches = np.array_split(group_idcs, num_splits)

                for batch in sub_batches:
                    batch_len = len(batch)
                    if batch_len >= self.min_batch_size:
                        self.valid_sets.append(batch)
                        self.set_ids.append(assay_id)
                    else:
                        num_unused += batch_len
            else:
                self.valid_sets.append(group_idcs)
                self.set_ids.append(assay_id)

        self.set_ids = np.array(self.set_ids, dtype=np.int32)

        logger.debug(f"Number of unused examples: {num_unused} / {len(self.data)}")

    def _shuffle_data(self):
        """Shuffle data only within groups of identical protein features."""
        prot_array = self.data[TID].values.astype(str)
        _, group_ids = np.unique(prot_array, return_inverse=True)

        num_samples = self.ligand_features.shape[0]
        all_indices = np.arange(num_samples)
        new_order = np.empty(num_samples, dtype=np.int64)

        if self.shuffle_within_target:
            unique_gids = np.unique(group_ids)
            for gid in unique_gids:
                mask = group_ids == gid
                idxs = all_indices[mask]
                new_order[mask] = self.random.permutation(idxs)
        else:
            new_order = self.random.permutation(num_samples)

        self.ligand_features = self.ligand_features[new_order]
        self.labels = self.labels[new_order]
        self.info = self.info[new_order]

    def _make_batches(self):
        """Create batches of multiple sets for processing."""
        if self.inter_assay:
            self._shuffle_data()

        num_sets = len(self.valid_sets)
        indices = self.random.permutation(num_sets)
        self.valid_sets = [self.valid_sets[i] for i in indices]
        self.set_ids = self.set_ids[indices]

        num_batches = (num_sets + self.sets_per_batch - 1) // self.sets_per_batch
        self.batches = []
        self.batch_set_ids = []

        self.batches = [None] * num_batches
        self.batch_set_ids = [None] * num_batches

        batch_idx = 0
        for i in range(0, num_sets, self.sets_per_batch):
            end_idx = min(i + self.sets_per_batch, num_sets)
            self.batches[batch_idx] = self.valid_sets[i:end_idx]
            self.batch_set_ids[batch_idx] = self.set_ids[i:end_idx]
            batch_idx += 1

        self.used = np.zeros(len(self.batches), dtype=bool)
        logger.debug(
            f"created {len(self.batches)} batches with up to {self.sets_per_batch} sets each"
        )

    def _get_next_batch(self, idx: int):
        """Get the next available batch and mark it as used."""
        if self.used.all():
            self._make_batches()
        self.used[idx] = True
        return self.batches[idx], self.batch_set_ids[idx]

    def __len__(self):
        """Get the number of batches in the dataset."""
        if not hasattr(self, "batches"):
            self._make_batches()
        return len(self.batches)

    def __getitem__(self, idx):
        """Get a batch of multiple sets from the dataset.

        Returns:
            tuple: Protein features, ligand features, labels, info for each set in the batch,
                  and metadata to track set boundaries for the loss function.
        """
        batch_sets, batch_ids = self._get_next_batch(idx)

        set_sizes = np.array([len(s) for s in batch_sets], dtype=np.int32)
        cumulative_sizes = np.concatenate([[0], np.cumsum(set_sizes)])

        all_indices = np.concatenate(batch_sets)

        set_ids_tensor = torch.from_numpy(
            np.repeat(np.arange(len(set_sizes), dtype=np.int64), set_sizes)
        )

        prot_feats = (
            torch.ones(1, device=device)
            if self.protein_features is None
            else self.protein_features[all_indices]
        )

        return (
            prot_feats,
            self.ligand_features[all_indices],
            self.labels[all_indices],
            self.info[all_indices],
            {
                "set_boundaries": cumulative_sizes,
                "set_ids": batch_ids,
                "set_ids_tensor": set_ids_tensor,
                "num_sets": len(batch_sets),
            },
        )


class MultiSetWithPropertiesDataset(MultiSetActivityDataset):
    """Extended dataset that dynamically creates property sets alongside assay sets.

    Property sets are created on-the-fly during batch creation using random linear
    combinations of physicochemical properties with Gaussian noise, allowing for
    more diverse sampling across epochs.

    Args:
        property_set_ratio (float): Ratio of property sets within the fixed batch size
            (0.0 = all assay sets, 0.5 = half property/half assay, 1.0 = all property sets).
        property_columns (List[str]): List of property column names from data to use as targets.
            These should be columns already present in the data DataFrame (e.g., 'mw_freebase', 'alogp').
        noise_std (float): Standard deviation of Gaussian noise to add to linear combinations.
            Default is 0.1 (relative to normalized property values).
        batch_size (int): Fixed total number of sets (assay + property) in each batch.
            Default is 32.
        fixed_set_size (int): Fixed number of items per set in the output tensor.
            Sets larger than this will be subsampled; smaller sets will be padded.
    """

    def __init__(
        self,
        data,
        target=...,
        info_cols=...,
        property_set_ratio: float = 0.5,
        property_columns: list[str] = None,
        noise_std: float = 0.1,
        batch_size: int = 32,
        fixed_set_size: int = 512,
        **kwargs,
    ):
        super().__init__(data, target=target, info_cols=info_cols, **kwargs)
        logger.info(f"ratio of physchem property set: {property_set_ratio}")

        self._property_set_ratio = property_set_ratio
        self.base_target = target
        self.property_columns = property_columns or []
        self.noise_std = noise_std
        self.batch_size = batch_size
        self.fixed_set_size = fixed_set_size

        logger.info(f"Fixed batch size (total sets per batch): {self.batch_size}")
        logger.info(f"Fixed set size (padding/subsampling): {self.fixed_set_size}")

        if self.property_columns:
            missing_cols = [
                col for col in self.property_columns if col not in self.data.columns
            ]
            if missing_cols:
                logger.warning(f"Missing property columns in data: {missing_cols}")
                self.property_columns = [
                    col for col in self.property_columns if col not in missing_cols
                ]

            if len(self.property_columns) == 0:
                logger.warning("No valid property columns found")
                self.property_values = None
            else:
                self._prepare_property_data()
                logger.info(f"Property columns available: {self.property_columns}")
                logger.info(f"Noise std for linear combinations: {self.noise_std}")
        else:
            self.property_values = None

    def _make_batches(self):
        """Create batches with a fixed total size split between assay and property sets."""
        if self.inter_assay:
            self._shuffle_data()

        indices = np.arange(len(self.valid_sets))
        self.random.shuffle(indices)
        shuffled_assay_sets = [self.valid_sets[i] for i in indices]
        shuffled_assay_ids = [self.set_ids[i] for i in indices]

        self.batches = []
        self.batch_set_ids = []
        self.batch_set_types = []
        self.batch_set_targets = []
        self.batch_set_coefficients = []

        num_assay_sets = len(shuffled_assay_sets)

        # Calculate split based on ratio
        num_property_per_batch = int(self.batch_size * self.property_set_ratio)
        num_assay_per_batch = self.batch_size - num_property_per_batch

        logger.info(
            f"Each batch will contain {num_assay_per_batch} assay sets and {num_property_per_batch} property sets"
        )

        # Handle case where all sets are property sets
        if num_assay_per_batch == 0:
            # Create batches with only property sets
            num_batches = max(1, num_assay_sets // self.batch_size)
            for batch_idx in range(num_batches):
                if num_property_per_batch > 0:
                    (
                        prop_sets,
                        prop_ids,
                        prop_types,
                        prop_targets,
                        prop_coeffs,
                    ) = self._create_property_sets_for_batch(num_property_per_batch)

                    if prop_sets:
                        self._finalize_batch(
                            prop_sets,
                            prop_ids,
                            prop_types,
                            prop_targets,
                            prop_coeffs,
                        )
        else:
            # Normal case with both assay and property sets
            for i in range(0, num_assay_sets, num_assay_per_batch):
                start_idx = i
                end_idx = min(i + num_assay_per_batch, num_assay_sets)

                if start_idx == end_idx:
                    continue

                current_batch_sets = [
                    shuffled_assay_sets[j] for j in range(start_idx, end_idx)
                ]
                current_batch_ids = [
                    shuffled_assay_ids[j] for j in range(start_idx, end_idx)
                ]
                num_assay_in_batch = len(current_batch_sets)

                current_batch_types = ["assay"] * num_assay_in_batch
                current_batch_targets = [self.base_target] * num_assay_in_batch
                current_batch_coeffs = [None] * num_assay_in_batch

                # Add property sets to reach fixed batch size
                if num_property_per_batch > 0:
                    (
                        prop_sets,
                        prop_ids,
                        prop_types,
                        prop_targets,
                        prop_coeffs,
                    ) = self._create_property_sets_for_batch(num_property_per_batch)

                    if prop_sets:
                        current_batch_sets.extend(prop_sets)
                        current_batch_ids.extend(prop_ids)
                        current_batch_types.extend(prop_types)
                        current_batch_targets.extend(prop_targets)
                        current_batch_coeffs.extend(prop_coeffs)

                if current_batch_sets:
                    self._finalize_batch(
                        current_batch_sets,
                        current_batch_ids,
                        current_batch_types,
                        current_batch_targets,
                        current_batch_coeffs,
                    )

        self.used = np.full(len(self.batches), False, dtype=bool)

        total_assay_sets = sum(
            sum(1 for t in types if t == "assay") for types in self.batch_set_types
        )
        total_property_sets = sum(
            sum(1 for t in types if t == "property") for types in self.batch_set_types
        )

        logger.info(
            f"Created {len(self.batches)} batches with {total_assay_sets} assay sets and {total_property_sets} property sets total"
        )

    def _finalize_batch(
        self, batch_sets, batch_ids, batch_types, batch_targets, batch_coeffs
    ):
        """Shuffle and add a batch to the batch lists."""
        batch_indices = np.arange(len(batch_sets))
        self.random.shuffle(batch_indices)

        self.batches.append([batch_sets[j] for j in batch_indices])
        self.batch_set_ids.append([batch_ids[j] for j in batch_indices])
        self.batch_set_types.append([batch_types[j] for j in batch_indices])
        self.batch_set_targets.append([batch_targets[j] for j in batch_indices])
        self.batch_set_coefficients.append([batch_coeffs[j] for j in batch_indices])

    def _get_next_batch(self, idx: int):
        """Get the next available batch and mark it as used."""
        if np.all(self.used):
            self._make_batches()
        self.used[idx] = True
        return (
            self.batches[idx],
            self.batch_set_ids[idx],
            self.batch_set_types[idx],
            self.batch_set_targets[idx],
            self.batch_set_coefficients[idx],
        )

    @property
    def property_set_ratio(self):
        """The property_set_ratio property."""
        return self._property_set_ratio

    @property_set_ratio.setter
    def property_set_ratio(self, value):
        self._property_set_ratio = value
        self._make_batches()

    def _prepare_property_data(self):
        """Prepare compound property data aligned with dataset indices."""
        self.property_values = {}
        self.property_stats = {}

        for prop_col in self.property_columns:
            valid_mask = self.data[prop_col].notna()

            if valid_mask.sum() < self.min_batch_size:
                logger.warning(
                    f"Property {prop_col} has too few valid values ({valid_mask.sum()}), skipping"
                )
                continue

            valid_values = self.data.loc[valid_mask, prop_col].values
            mean_val = np.mean(valid_values)
            std_val = np.std(valid_values)

            if std_val < 1e-8:
                std_val = 1.0

            prop_tensor = torch.full(
                (len(self.data),), float("nan"), dtype=torch.float32, device="cpu"
            )

            valid_indices = self.data.index[valid_mask].values
            normalized_values = (valid_values - mean_val) / std_val
            prop_tensor[valid_indices] = torch.tensor(
                normalized_values, dtype=torch.float32
            )

            self.property_values[prop_col] = {
                "tensor": prop_tensor,
                "valid_indices": valid_indices,
            }

            self.property_stats[prop_col] = {"mean": mean_val, "std": std_val}

        if self.property_values:
            prepared_properties = list(self.property_values.keys())
            if not prepared_properties:
                self.common_valid_indices = np.array([])
            else:
                common_indices = set(
                    self.property_values[prepared_properties[0]]["valid_indices"]
                )
                for prop_col in prepared_properties[1:]:
                    common_indices &= set(
                        self.property_values[prop_col]["valid_indices"]
                    )
                self.common_valid_indices = np.array(sorted(common_indices))

            logger.info(
                f"Common valid indices across all properties: {len(self.common_valid_indices)}"
            )
        else:
            self.common_valid_indices = np.array([])

        logger.info(
            f"Prepared properties for molecules: {list(self.property_values.keys())}"
        )

    def _create_property_sets_for_batch(self, num_sets: int):
        if (
            not self.property_values
            or num_sets <= 0
            or len(self.common_valid_indices) == 0
        ):
            return [], [], [], [], []

        property_sets = []
        property_ids = []
        property_types = []
        property_targets = []
        property_coefficients = []

        available_properties = list(self.property_values.keys())
        n_properties = len(available_properties)

        for set_idx in range(num_sets):
            coeffs = self.random.standard_normal(n_properties)
            coeffs = coeffs / np.linalg.norm(coeffs)

            max_available = len(self.common_valid_indices)
            if self.max_set_size > 0:
                max_size = min(self.max_set_size, max_available)
            else:
                max_size = max_available

            if max_size < self.min_batch_size:
                continue

            set_size = self.random.integers(self.min_batch_size, max_size + 1)

            sample_idx = self.random.choice(
                len(self.common_valid_indices), size=set_size, replace=False
            )

            selected_indices = self.common_valid_indices[sample_idx]

            property_sets.append(selected_indices)
            property_ids.append(f"property_combo_{set_idx}")
            property_types.append("property")
            coeff_dict = {
                prop: float(c) for prop, c in zip(available_properties, coeffs)
            }
            property_targets.append(coeff_dict)
            property_coefficients.append(coeffs)

        return (
            property_sets,
            property_ids,
            property_types,
            property_targets,
            property_coefficients,
        )

    def _compute_linear_combination(self, indices, coefficients):
        available_properties = list(self.property_values.keys())

        result = torch.zeros(len(indices), dtype=torch.float32)

        for prop, coeff in zip(available_properties, coefficients):
            prop_values = self.property_values[prop]["tensor"][indices]
            result += coeff * prop_values

        noise = torch.randn_like(result) * self.noise_std
        result += noise

        return result

    def __getitem__(self, idx):
        batch_sets, batch_ids, batch_types, batch_targets, batch_coeffs = (
            self._get_next_batch(idx)
        )

        num_sets = len(batch_sets)
        K = self.fixed_set_size

        # Use 0 for padding indices.
        all_indices_padded = torch.zeros((num_sets, K), dtype=torch.long)
        all_labels_padded = torch.zeros((num_sets, K), dtype=torch.float32)
        attention_mask = torch.ones((num_sets, K), dtype=torch.bool)  # True = pad

        for i, set_idcs in enumerate(batch_sets):
            set_type = batch_types[i]
            coeffs = batch_coeffs[i]
            set_size = len(set_idcs)

            if set_size == 0:
                continue

            if set_size > K:
                sample_idx = self.random.choice(set_size, size=K, replace=False)
                final_indices = set_idcs[sample_idx]

                all_indices_padded[i, :] = torch.from_numpy(final_indices)
                attention_mask[i, :] = False

                if set_type == "assay":
                    all_labels_padded[i, :] = self.labels[final_indices]
                else:
                    all_labels_padded[i, :] = self._compute_linear_combination(
                        final_indices, coeffs
                    )

            else:
                final_indices = set_idcs

                all_indices_padded[i, :set_size] = torch.from_numpy(final_indices)
                attention_mask[i, :set_size] = False

                if set_type == "assay":
                    all_labels_padded[i, :set_size] = self.labels[final_indices]
                else:
                    all_labels_padded[i, :set_size] = self._compute_linear_combination(
                        final_indices, coeffs
                    )

        prot_feats = (
            torch.ones(1)
            if self.protein_features is None
            else self.protein_features[all_indices_padded]
        )

        return (
            prot_feats,
            self.ligand_features[all_indices_padded],
            all_labels_padded,
            self.info[all_indices_padded],
            {
                "set_ids": batch_ids,
                "set_types": batch_types,
                "set_targets": batch_targets,
                "attention_mask": attention_mask,
            },
        )


def aggregate_multi_measurements(
    data: pd.DataFrame, keys: Iterable[str] = [COMPOUND, ASSAY]
) -> pd.DataFrame:
    """Aggregate multiple measurements for the same compound and assay.

    Args:
        data (pd.DataFrame): DataFrame containing activity measurements.

    Returns:
        pd.DataFrame: DataFrame with aggregated measurements.
    """
    if TID in data.columns:
        keys += [TID]
    logger.debug(f"aggregate multiple measurements per {keys}")
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
    inter_assay_weight: float | None = None,
    scale_scores: bool = False,
    scale_targets: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame, pd.DataFrame]:
    split_dir = data_dir / f"{index}"
    logger.info(f"reading dataset from {split_dir}")

    val_data = pd.read_csv(split_dir / "val.csv", index_col=0)
    train_data = pd.read_csv(split_dir / "train.csv", index_col=0)
    test_data = pd.read_csv(split_dir / "test.csv", index_col=0)

    if scale_targets:
        scaler = StandardScaler()
        train_data[tgt_name] = scaler.fit_transform(
            train_data[ACT].values.reshape(-1, 1)
        )
        test_data[tgt_name] = scaler.transform(test_data[ACT].values.reshape(-1, 1))
        val_data[tgt_name] = scaler.transform(val_data[ACT].values.reshape(-1, 1))
    else:
        train_data[tgt_name] = train_data[ACT]
        test_data[tgt_name] = test_data[ACT]
        val_data[tgt_name] = val_data[ACT]

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
    inter_assay_weight: float | None = None,
    random_valset: bool = False,
    aggregate: bool = True,
    columns: list[str] = [ASSAY],
) -> Iterator[
    tuple[int, pd.DataFrame, pd.DataFrame | None, pd.DataFrame, pd.DataFrame]
]:
    """Prepare train, validation, and test datasets."""
    logger.info(f"split along {columns}")
    if aggregate:
        data = aggregate_multi_measurements(data)
    split_data(data, data_dir, columns=columns, k=k, random_valset=random_valset)


def _process(data, col_map):
    assert all(k in data.columns for k in col_map.keys()), data.columns
    backup_cols = {v: v + "_orig" for k, v in col_map.items() if k != v}
    col_map.update(backup_cols)
    data = data.rename(columns=col_map)
    data = data[~data[SMILES].isna()]
    data = data[~data[ACT].isna()]
    if SEQUENCE in data.columns:
        data = data[~data[SEQUENCE].isna()]
    return data


def load_kinodata(
    kinodata_path: Path = DATA / "raw" / "activities-chembl33_v0.5.csv",
    activity_types: list[str] = ["pIC50"],
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
            "activities.activity_id": IDENT,
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


def load_chembl_endpoints(
    path: Path = DATA / "raw" / "chembl_endpoints_processed.csv.gz",
) -> pd.DataFrame:
    logger.info("loading general ChEMBL endpoints")
    data = pd.read_csv(path, index_col=False)
    assert data["compound_id"].dtype == int
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
) -> pd.DataFrame:
    logger.info("loading general ChEMBL protein endpoints")
    data = pd.read_csv(path, index_col=False)
    assert data["compound_id"].dtype == int
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
