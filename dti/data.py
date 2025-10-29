from collections.abc import Iterator, Iterable
import functools
import logging
from pathlib import Path

import pandas as pd
import numpy as np


import torch
from torch.utils.data import Dataset, Sampler
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


class MultiSetMapDataset(MultiSetActivityDataset):
    """
    Refactored Map-Style Dataset for parallel loading.

    This dataset creates a static pool of all assay and property sets
    at initialization. A custom BatchSampler and collate_fn are
    required to use this class with a DataLoader.
    """

    def __init__(
        self,
        data,
        target: str = ...,
        info_cols: list[str] = ...,
        query_column: str | None = None,
        query_ratio: float = 0.2,
        property_set_ratio: float = 0.5,
        property_columns: list[str] = None,
        noise_std: float = 0.1,
        # These max_batch... params are no longer used here
        max_batch_cost: int = None,
        max_batch_datapoints: int = None,
        **kwargs,
    ):
        super().__init__(data, target=target, info_cols=info_cols, **kwargs)
        self._property_set_ratio = property_set_ratio
        self.property_columns = property_columns or []
        self.noise_std = noise_std

        if max_batch_cost is not None or max_batch_datapoints is not None:
            logger.warning(
                "'max_batch_cost' and 'max_batch_datapoints' are no longer "
                "arguments for the Dataset. Pass 'max_batch_cost' to "
                "the 'CostBasedBatchSampler' instead."
            )

        self.property_tensor = None
        self.common_valid_indices = np.array([])
        self.n_properties = 0
        if query_column is not None and query_column not in data.columns:
            raise ValueError(f"invalid query column: {query_column}")
        self.query_column = query_column
        self.query_ratio = query_ratio

        self.is_query_tensor = None
        if self.query_column is not None:
            self.is_query_tensor = torch.from_numpy(data[query_column].values).bool()

        logger.info(f"Property set ratio: {self._property_set_ratio}")

        if self.property_columns:
            self._prepare_property_data()
        else:
            logger.warning(
                "No property_columns provided. No property sets will be generated."
            )

        # --- NEW: Create the static pool of all sets ---
        self._create_set_pool()

    def _prepare_property_data(self):
        # (Unchanged from your code, but logger.info is used)
        logger.debug(f"Preparing property data for columns: {self.property_columns}")
        valid_mask = self.data[self.property_columns].notna().all(axis=1)
        self.common_valid_indices = self.data.index[valid_mask].values
        if len(self.common_valid_indices) < self.min_batch_size:
            logger.warning(f"Too few samples... Disabling property sets.")
            self.property_columns = []
            return
        logger.info(
            f"Found {len(self.common_valid_indices)} samples with valid data "
            "for all property columns."
        )
        valid_prop_data = self.data.loc[valid_mask, self.property_columns].values
        self.n_properties = valid_prop_data.shape[1]
        mean = valid_prop_data.mean(axis=0)
        std = valid_prop_data.std(axis=0)
        std[std < 1e-8] = 1.0
        self.property_stats = {"mean": mean, "std": std}
        normalized_props = (valid_prop_data - mean) / std
        self.property_tensor = torch.zeros(
            (len(self.data), self.n_properties), dtype=torch.float32
        )
        self.property_tensor[self.common_valid_indices] = torch.tensor(
            normalized_props, dtype=torch.float32
        )

    def _create_propsets(self, num_sets: int) -> tuple:
        # (Unchanged from your code)
        if not self.property_columns or num_sets <= 0 or self.property_tensor is None:
            return [], [], []
        max_available = len(self.common_valid_indices)
        if self.max_set_size > 0:
            max_size = min(self.max_set_size, max_available)
        else:
            max_size = max_available
        if max_size < self.min_batch_size:
            logger.error("Max available property samples < min_batch_size...")
            return [], [], []
        all_coeffs = self.random.standard_normal((num_sets, self.n_properties))
        norms = np.linalg.norm(all_coeffs, axis=1, keepdims=True)
        norms[norms < 1e-8] = 1.0
        all_coeffs /= norms
        all_coeffs = all_coeffs.astype(np.float32)
        set_sizes = self.random.integers(
            self.min_batch_size, max_size + 1, size=num_sets
        )
        property_sets, property_ids, property_coeffs = [], [], []
        for i in range(num_sets):
            size = set_sizes[i]
            sample_idx = self.random.choice(
                len(self.common_valid_indices), size=size, replace=False
            )
            selected_indices = self.common_valid_indices[sample_idx]
            property_sets.append(selected_indices)
            property_ids.append(f"prop_{i}")
            property_coeffs.append(all_coeffs[i])
        return property_sets, property_ids, property_coeffs

    def _create_set_pool(self):
        """Creates the master list of all sets (assay and property)."""
        if self.inter_assay:
            self._shuffle_data()  # From MultiSetActivityDataset

        num_a_sets = len(self.valid_sets)

        # (indices, set_id, type, coefficients)
        a_pool = [
            (self.valid_sets[i], self.set_ids[i], "assay", None)
            for i in range(num_a_sets)
        ]

        num_property_sets = int(num_a_sets * self._property_set_ratio)
        p_indices, p_ids, p_coeffs = self._create_propsets(num_property_sets)
        p_pool = [
            (p_indices[i], p_ids[i], "property", p_coeffs[i])
            for i in range(len(p_indices))
        ]

        self.all_sets_data = a_pool + p_pool
        # Store set sizes separately for the sampler
        self.set_sizes = [len(s[0]) for s in self.all_sets_data]

        logger.info(
            f"Created pool of {len(a_pool)} assay sets and "
            f"{len(p_pool)} property sets. Total: {len(self.all_sets_data)}"
        )

    def _compute_linear_combination(
        self, indices: np.ndarray, coefficients: np.ndarray
    ) -> torch.Tensor:
        """
        Compute linear combination of normalized properties with noise.
        MODIFIED: Returns a CPU tensor.
        """
        # self.property_tensor is already a CPU tensor
        prop_values = self.property_tensor[indices]

        # Create coefficients on CPU
        coeffs_tensor = torch.tensor(coefficients, dtype=torch.float32).unsqueeze(1)

        result = torch.matmul(prop_values, coeffs_tensor).squeeze(1)
        noise = torch.randn_like(result) * self.noise_std
        return result + noise

    @property
    def determistic_queries(self):
        return self.query_column is not None

    def __len__(self) -> int:
        """Returns the total number of sets."""
        return len(self.all_sets_data)

    def __getitem__(self, idx: int) -> dict:
        """
        Fetches data for a SINGLE set (unbatched, unpadded).
        This method is run by the parallel workers.
        """
        indices, set_id, set_type, coeffs = self.all_sets_data[idx]
        size = len(indices)

        # 1. Get features (always on CPU)
        if self.protein_features is None:
            # Use a placeholder float. Collate_fn will handle dims.
            prot_feat = torch.tensor([1.0], dtype=torch.float32)
            prot_dim = 1
        else:
            prot_feat = self.protein_features[indices]
            prot_dim = prot_feat.shape[1]

        lig_feat = self.ligand_features[indices]
        lig_dim = lig_feat.shape[1]

        # 2. Get labels and query info
        is_query = None
        if set_type == "assay":
            label = self.labels[indices]
            if self.determistic_queries:
                is_query = self.is_query_tensor[indices]
        else:  # property
            label = self._compute_linear_combination(indices, coeffs)

        return {
            "indices": indices,  # For info_list
            "prot_feat": prot_feat,
            "lig_feat": lig_feat,
            "label": label,
            "set_type": set_type,
            "is_query": is_query,  # (B, L) or None
            "size": size,
            "prot_dim": prot_dim,  # Pass dims for collator
            "lig_dim": lig_dim,
        }


class CostBasedBatchSampler(Sampler):
    """
    Groups sets into batches based on a quadratic cost limit (B * L_max^2).

    Sorts sets by size to pack efficiently, then shuffles the
    resulting batches to ensure stochasticity during training.
    """

    def __init__(
        self,
        set_sizes: list[int],
        max_batch_cost: int,
        shuffle: bool = True,
        seed: int = 42,
    ):
        self.set_sizes = set_sizes
        self.max_batch_cost = max_batch_cost
        self.shuffle = shuffle
        self.generator = torch.Generator().manual_seed(seed)

        # Pre-compute batches
        self.batches = self._create_batches()

    def _create_batches(self) -> list[list[int]]:
        if self.shuffle:
            indices = torch.randperm(
                len(self.set_sizes), generator=self.generator
            ).tolist()
        else:
            indices = list(range(len(self.set_sizes)))

        # Sort indices by set size in descending order for efficient packing
        sorted_indices = sorted(indices, key=lambda i: self.set_sizes[i], reverse=True)

        batches = []
        current_batch = []
        current_max_len = 0

        for idx in sorted_indices:
            set_size = self.set_sizes[idx]

            set_cost = set_size**2
            if set_cost > self.max_batch_cost:
                logger.warning(
                    f"Set at index {idx} (size {set_size}) has cost ({set_cost}) "
                    f"larger than max_batch_cost ({self.max_batch_cost}). "
                    "Creating a single oversized batch. THIS MAY CAUSE OOM."
                )
                if current_batch:
                    batches.append(current_batch)
                batches.append([idx])
                current_batch = []
                current_max_len = 0
                continue

            new_batch_size = len(current_batch) + 1
            new_max_len = max(current_max_len, set_size)
            new_cost = new_batch_size * (new_max_len**2)

            if new_cost > self.max_batch_cost and current_batch:
                batches.append(current_batch)
                current_batch = [idx]
                current_max_len = set_size
            else:
                current_batch.append(idx)
                current_max_len = new_max_len

        if current_batch:
            batches.append(current_batch)

        return batches

    def __iter__(self):
        if self.shuffle:
            # Shuffle the order of the batches themselves
            indices = torch.randperm(
                len(self.batches), generator=self.generator
            ).tolist()
            for i in indices:
                yield self.batches[i]
        else:
            for batch in self.batches:
                yield batch

    def __len__(self) -> int:
        return len(self.batches)


class SetCollator:
    """
    Collates a list of single-set dictionaries into a padded batch.

    This is where all padding, query mask generation, and attention
    mask generation logic is performed.
    """

    def __init__(
        self,
        query_ratio: float,
        determistic_queries: bool,
        info_accessor=None,  # Pass dataset.info if needed
        device: torch.device = torch.device("cpu"),
    ):
        self.query_ratio = query_ratio
        self.determistic_queries = determistic_queries
        self.info_accessor = info_accessor  # e.g., dataset.info
        self.device = device

        # We will create a random generator inside __call__
        # to ensure workers have different seeds
        self.random = None

    def _get_random_generator(self):
        """Creates a worker-specific random generator."""
        worker_info = torch.utils.data.get_worker_info()
        if worker_info:
            seed = worker_info.seed
        else:
            seed = torch.initial_seed()

        return np.random.RandomState(seed=seed % (2**32 - 1))

    def _random_mask(self, size: int) -> np.ndarray:
        ratio = int(size * self.query_ratio)
        if ratio <= 0 and size > 1:
            ratio = 1  # Ensure at least one query if possible
        if ratio >= size:
            ratio = size - 1  # Ensure at least one support

        mask = self.random.choice(np.arange(size), ratio, replace=False)
        return mask

    def __call__(self, batch_items: list[dict]) -> tuple:
        """
        Processes the list of dictionaries from the Dataset workers.
        """
        # Ensure each worker/epoch has a different random state
        self.random = self._get_random_generator()

        num_sets = len(batch_items)
        max_len = max(item["size"] for item in batch_items)

        # Get feature dimensions from the first item
        prot_dim = batch_items[0]["prot_dim"]
        lig_dim = batch_items[0]["lig_dim"]

        # Initialize tensors (on CPU first)
        prot_feats = torch.zeros(num_sets, max_len, prot_dim)
        lig_feats = torch.zeros(num_sets, max_len, lig_dim)
        labels = torch.zeros(num_sets, max_len)
        padding_mask = torch.ones(num_sets, max_len, dtype=torch.bool)
        query_mask = torch.zeros(num_sets, max_len, dtype=torch.bool)
        info_list = []

        for i, item in enumerate(batch_items):
            size = item["size"]
            padding_mask[i, :size] = False

            if item["prot_dim"] == 1:
                # Handle the prot_feat=1.0 placeholder
                prot_feats[i, :size, :] = 1.0
            else:
                prot_feats[i, :size] = item["prot_feat"]

            lig_feats[i, :size] = item["lig_feat"]
            labels[i, :size] = item["label"]

            # Query mask logic
            set_type = item["set_type"]
            is_query = item["is_query"]

            if set_type == "assay":
                if self.determistic_queries:
                    assert is_query is not None, "Deterministic queries missing"
                    assert not is_query.all(), "Set cannot be 100% queries"
                    query_mask[i, :size] = is_query
                else:
                    query_mask[i, self._random_mask(size)] = True
            else:  # property set
                query_mask[i, self._random_mask(size)] = True

            if self.info_accessor is not None:
                info_list.append(self.info_accessor[item["indices"]])

        # --- Attention Mask Logic (copied from your old __getitem__) ---
        B, L = padding_mask.shape

        # Move masks to device *before* mask logic
        padding_mask = padding_mask.to(self.device)
        query_mask = query_mask.to(self.device)

        base = (~padding_mask).unsqueeze(1) & (~padding_mask).unsqueeze(2)
        disallow_nonq_to_q = (~query_mask).unsqueeze(2) & query_mask.unsqueeze(1)

        q_to_q = query_mask.unsqueeze(1) & query_mask.unsqueeze(2)
        self_mask = torch.eye(L, dtype=torch.bool, device=self.device).unsqueeze(0)
        disallow_q_to_q = q_to_q & ~self_mask

        disallow = disallow_nonq_to_q | disallow_q_to_q
        attn_mask = ~base | disallow
        attn_mask = attn_mask & ~(query_mask.unsqueeze(1) & self_mask)
        # fully_masked_rows = attn_mask.all(dim=-1) # (B,L)
        attn_mask = attn_mask | self_mask

        # --- Final move to device ---
        prot_feats = prot_feats.to(self.device)
        lig_feats = lig_feats.to(self.device)
        labels = labels.to(self.device)

        return (
            prot_feats,
            lig_feats,
            labels,
            info_list,
            query_mask,
            padding_mask,
            attn_mask,
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
