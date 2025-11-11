from collections.abc import Iterator, Iterable
import os
import hashlib
import functools
import logging
from pathlib import Path
from multiprocessing import Pool
from threading import Lock

import pandas as pd
import numpy as np
from filelock import FileLock
import logging


import torch
from torch.utils.data import Dataset, Sampler
from torch_geometric.data import Data, Batch
from sklearn.preprocessing import StandardScaler

from .featurization import MolFingerprint, esm2_features
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
from ..utils import device, check_smi_valid
from ..utils.hodge_ranking import parallel_hodge_rank

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


class PropertySetDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        mol_featurizer: MolFingerprint,
        target: str = ACT,
        property_columns: list[str] = None,
        info_cols: list[str] = [],
        model_name: str = "esm2_t33_650M_UR50D",
        **kwargs,
    ):
        super().__init__()
        logger.info(f"Creating PropertySetDataset of size {len(data)}")
        self._prepare_features_and_data(
            data, mol_featurizer, info_cols, target, **kwargs
        )
        self._setup_sets_and_properties(property_columns=property_columns, **kwargs)
        self.lock = Lock()

    def _prepare_features_and_data(
        self, data, mol_featurizer, info_cols, target, **kwargs
    ):
        logger.info("Computing molecular fingerprints...")
        n_jobs = kwargs.get("n_jobs", 16)
        fps = mol_featurizer.compute_parallel(data[SMILES].values, n_jobs=n_jobs)
        mask = [fp is not None for fp in fps]

        valid_count = sum(mask)
        if len(mask) - valid_count > 0:
            logger.info(
                f"Dropping {len(mask) - valid_count}/{len(mask)} data points "
                f"with invalid fingerprints."
            )

        self.data = data[mask].copy().reset_index(drop=True)

        self.ligand_features = torch.tensor(
            np.stack([fp for fp in fps if fp is not None]),
            dtype=torch.float32,
        )

        self.assay_labels = torch.tensor(self.data[target].values, dtype=torch.float32)
        self.info = torch.tensor(self.data[info_cols].values.astype(np.int64))

    def _setup_sets_and_properties(self, property_columns: list[str] = None, **kwargs):
        self.min_batch_size = kwargs.get("min_batch_size", 3)
        self.max_set_size = kwargs.get("max_set_size", 0)
        self.random = np.random.default_rng(kwargs.get("random_seed", 0))

        self.assay_sets = []
        grouped = self.data.groupby(ASSAY, sort=False)
        num_unused = 0

        for _, group in grouped:
            group_idcs = group.index.values
            group_size = len(group_idcs)
            if group_size < self.min_batch_size:
                num_unused += group_size
                continue
            self.random.shuffle(group_idcs)
            if self.max_set_size > 0 and group_size > self.max_set_size:
                num_splits = (group_size + self.max_set_size - 1) // self.max_set_size
                sub_batches = np.array_split(group_idcs, num_splits)
                for batch in sub_batches:
                    if len(batch) >= self.min_batch_size:
                        self.assay_sets.append(batch)
                    else:
                        num_unused += len(batch)
            else:
                self.assay_sets.append(group_idcs)
        logger.info(
            f"Created {len(self.assay_sets)} assay sets. "
            f"({num_unused}/{len(self.data)} datapoints unused)"
        )

        self.property_set_ratio = kwargs.get("property_set_ratio", 0.5)
        self.noise_std = kwargs.get("noise_std", 0.1)
        self.max_batch_datapoints = kwargs.get("max_batch_datapoints", 2048)
        self.property_columns = property_columns or []

        self._prepare_property_data()

        self.property_coeff_pool_size = 32
        self.coeff_drift_alpha = 0.9
        self._init_property_coeff_pool()

        self.batches_plan = None
        self.used_batches = None

    def _get_ligand_features(self, indices: np.ndarray):
        return self.ligand_features[indices]

    def __getitem__(self, idx):
        batch_plan = self._get_next_batch(idx)

        num_sets = len(batch_plan)
        all_indices_list = []
        all_labels_list = []
        set_sizes = []
        real_assay = []

        for item in batch_plan:
            set_type = item[0]
            indices = item[1]
            set_sizes.append(len(indices))
            all_indices_list.append(indices)
            real_assay.extend([set_type == "assay"] * len(indices))

            if set_type == "assay":
                all_labels_list.append(self.assay_labels[indices])
            else:
                coeffs_tensor = item[2]
                labels = self._compute_property_labels(indices, coeffs_tensor)
                all_labels_list.append(labels)

        all_indices = np.concatenate(all_indices_list)
        all_labels = torch.cat(all_labels_list)

        info = self.info[all_indices]

        lig_feats = self._get_ligand_features(all_indices)

        set_sizes_tensor = torch.tensor(set_sizes, dtype=torch.long)
        set_boundaries = torch.cat(
            [torch.tensor([0]), torch.cumsum(set_sizes_tensor, 0)]
        ).numpy()
        set_ids_tensor = torch.repeat_interleave(
            torch.arange(num_sets), set_sizes_tensor
        )

        metadata = {
            "num_sets": num_sets,
            "set_ids_tensor": set_ids_tensor,
            "set_boundaries": set_boundaries,
            "real_assay": torch.tensor(real_assay),
        }

        return lig_feats, all_labels, info, metadata

    def _init_property_coeff_pool(self):
        if not hasattr(self, "n_properties") or self.n_properties == 0:
            self.property_coeff_pool = []
            return
        self.property_coeff_pool = [
            self._random_coeff_vector() for _ in range(self.property_coeff_pool_size)
        ]

    def _drift_property_coeff_pool(self):
        if not self.property_coeff_pool:
            return
        new_coeffs = [
            self._random_coeff_vector() for _ in range(self.property_coeff_pool_size)
        ]
        self.property_coeff_pool = [
            self.coeff_drift_alpha * old + (1 - self.coeff_drift_alpha) * new
            for old, new in zip(self.property_coeff_pool, new_coeffs)
        ]
        self.property_coeff_pool = [
            c / np.linalg.norm(c) for c in self.property_coeff_pool
        ]

    def _random_coeff_vector(self):
        coeffs = self.random.standard_normal(self.n_properties)
        coeffs /= np.linalg.norm(coeffs)
        return coeffs

    def _prepare_property_data(self):
        if not self.property_columns:
            logger.info("No property columns specified.")
            self.available_properties = []
            self.property_matrix = torch.empty(len(self.data), 0)
            self.common_valid_indices = np.array([], dtype=np.int64)
            self.n_properties = 0
            return
        prop_tensors = []
        self.available_properties = []
        for prop_col in self.property_columns:
            if prop_col not in self.data.columns:
                logger.warning(f"Property column '{prop_col}' not in data. Skipping.")
                continue
            values = self.data[prop_col].values.astype(np.float32)
            valid_mask = ~np.isnan(values)
            if valid_mask.sum() < self.min_batch_size:
                logger.warning(
                    f"Property {prop_col} has too few valid values. Skipping."
                )
                continue
            scaler = StandardScaler()
            valid_values = values[valid_mask].reshape(-1, 1)
            scaler.fit(valid_values)
            normalized_values = scaler.transform(valid_values).flatten()
            prop_tensor = torch.full(
                (len(self.data),), float("nan"), dtype=torch.float32
            )
            prop_tensor[valid_mask] = torch.from_numpy(normalized_values)
            prop_tensors.append(prop_tensor)
            self.available_properties.append(prop_col)

        if not prop_tensors:
            self.property_matrix = torch.empty(len(self.data), 0)
            self.common_valid_indices = np.array([], dtype=np.int64)
            self.n_properties = 0
            return

        self.property_matrix = torch.stack(prop_tensors, dim=1)
        self.n_properties = self.property_matrix.shape[1]
        valid_mask_all = ~torch.isnan(self.property_matrix).any(dim=1)
        self.common_valid_indices = torch.where(valid_mask_all)[0].numpy()
        self.property_matrix.nan_to_num_(0.0)
        logger.info(
            f"Prepared {self.n_properties} property columns. "
            f"{len(self.common_valid_indices)} molecules have all properties."
        )

    def prepare_epoch(self):
        self._drift_property_coeff_pool()
        self._make_batches()
        self.used_batches = np.zeros(len(self.batches_plan), dtype=bool)

    def _make_batches(self):
        logger.debug("Remaking batches for new epoch...")
        self.batches_plan = []
        assay_set_indices = self.random.permutation(len(self.assay_sets))
        shuffled_assay_sets = [self.assay_sets[i] for i in assay_set_indices]
        while shuffled_assay_sets:
            current_batch_sets = []
            current_total_datapoints = 0
            num_assay_sets_in_batch = 0
            while shuffled_assay_sets:
                assay_set = shuffled_assay_sets[-1]
                set_size = len(assay_set)
                max_datapoints = self.max_batch_datapoints * (
                    1 - self.property_set_ratio
                )
                if current_total_datapoints + set_size <= max_datapoints:
                    current_total_datapoints += set_size
                    num_assay_sets_in_batch += 1
                    current_batch_sets.append(("assay", shuffled_assay_sets.pop()))
                else:
                    break
            if not current_batch_sets and shuffled_assay_sets:
                logger.warning(f"Assay set too large, creating oversized batch.")
                assay_set = shuffled_assay_sets.pop()
                current_batch_sets.append(("assay", assay_set))
                current_total_datapoints += len(assay_set)
                num_assay_sets_in_batch = 1

            target_property_sets = int(
                num_assay_sets_in_batch * self.property_set_ratio
            )
            if (
                self.property_set_ratio > 0
                and self.n_properties > 0
                and len(self.common_valid_indices) > 0
            ):
                for _ in range(target_property_sets):
                    max_available = len(self.common_valid_indices)
                    max_size = (
                        min(self.max_set_size, max_available)
                        if self.max_set_size > 0
                        else max_available
                    )
                    if max_size < self.min_batch_size:
                        continue
                    set_size = self.random.integers(self.min_batch_size, max_size + 1)
                    if current_total_datapoints + set_size > self.max_batch_datapoints:
                        break
                    current_total_datapoints += set_size
                    coeffs = self.random.choice(self.property_coeff_pool)
                    sample_idx = self.random.choice(
                        len(self.common_valid_indices), size=set_size, replace=False
                    )
                    selected_indices = self.common_valid_indices[sample_idx]
                    current_batch_sets.append(
                        ("property", selected_indices, torch.from_numpy(coeffs).float())
                    )

            if current_batch_sets:
                self.random.shuffle(current_batch_sets)
                self.batches_plan.append(current_batch_sets)

        self.used_batches = np.zeros(len(self.batches_plan), dtype=bool)

    def _get_next_batch(self, idx: int):
        # with self.lock:
        if self.batches_plan is None:
            self._make_batches()
        if idx >= len(self.batches_plan):
            logger.error(f"Index {idx} out of bounds. Wrapping around")
            idx = idx % len(self.batches_plan)
        self.used_batches[idx] = True
        return self.batches_plan[idx]

    def _compute_property_labels(self, indices, coeffs_tensor):
        props = self.property_matrix[indices]
        labels = props @ coeffs_tensor
        labels = self._corruption(labels)
        labels = torch.nan_to_num(labels)
        return labels

    def _corruption(self, labels):
        choice = self.random.random()
        if choice < 0.1:
            return labels ** self.random.standard_normal()
        elif choice < 0.2:
            return torch.exp(labels)
        elif choice < 0.3:
            return torch.maximum(
                torch.ones_like(labels) * self.random.standard_normal(), labels
            )
        elif choice < 0.9:
            self._add_noise(labels)
            return labels
        else:
            return labels

    def _add_noise(self, labels):
        if self.random.random() < 0.9:
            labels += torch.randn_like(labels) * self.noise_std
        else:
            labels *= 1 + torch.randn_like(labels) * self.noise_std

    def __len__(self):
        if self.batches_plan is None:
            self._make_batches()
        return len(self.batches_plan)


class ResettingBatchSampler(Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset

    def __iter__(self):
        self.dataset.prepare_epoch()
        for idx in range(len(self.dataset.batches_plan)):
            yield [idx]

    def __len__(self):
        if self.dataset.batches_plan is None:
            self.dataset._make_batches()
        return len(self.dataset.batches_plan)


class GraphPropertySetDataset(PropertySetDataset):
    def __init__(
        self,
        data: pd.DataFrame,
        mol_featurizer: MolFingerprint,
        target: str = ACT,
        property_columns: list[str] = None,
        info_cols: list[str] = [],
        **kwargs,
    ):
        super().__init__(
            data=data,
            mol_featurizer=mol_featurizer,
            target=target,
            property_columns=property_columns,
            info_cols=info_cols,
            **kwargs,
        )

    def _prepare_features_and_data(
        self, data, mol_featurizer, info_cols, target, **kwargs
    ):
        logger.info("Initializing GraphPropertySetDataset (on-the-fly).")

        cache_dir_str = os.environ.get("GRAPH_CACHE_DIR", "/tmp/graph_cache")
        self.scratch_dir = Path(cache_dir_str)
        self.scratch_dir.mkdir(exist_ok=True, parents=True)
        logger.info(f"Using sharded graph cache at: {self.scratch_dir}")

        self.mol_featurizer = mol_featurizer
        self.data = data.reset_index(drop=True)

        self.smiles_list = self.data[SMILES].values

        self.assay_labels = torch.tensor(self.data[target].values, dtype=torch.float32)
        self.info = torch.tensor(self.data[info_cols].values.astype(np.int64))

    def _get_ligand_features(self, indices: np.ndarray):
        smiles_to_fetch = self.smiles_list[indices]
        graphs_in_batch = [
            self.mol_featurizer.compute(smi, cache_dir=self.scratch_dir)
            for smi in smiles_to_fetch
        ]
        return Batch.from_data_list(graphs_in_batch)


class GraphAndFingerprintDataset(PropertySetDataset):
    def __init__(
        self,
        data: pd.DataFrame,
        mol_featurizer: MolFingerprint,
        graph_featurizer: MolFingerprint,
        target: str = ACT,
        property_columns: list[str] = None,
        info_cols: list[str] = [],
        **kwargs,
    ):
        logger.info("Initializing GraphAndFingerprintDataset.")

        self.graph_featurizer = graph_featurizer

        super().__init__(
            data=data,
            mol_featurizer=mol_featurizer,
            target=target,
            property_columns=property_columns,
            info_cols=info_cols,
            **kwargs,
        )

        self.smiles_list = self.data[SMILES].values

        cache_dir_str = os.environ.get("GRAPH_CACHE_DIR", "/tmp/graph_cache")
        self.scratch_dir = Path(cache_dir_str)
        self.scratch_dir.mkdir(exist_ok=True, parents=True)
        logger.info(f"Using sharded graph cache at: {self.scratch_dir}")

    def _get_ligand_features(self, indices: np.ndarray):
        fingerprints = self.ligand_features[indices]

        smiles_to_fetch = self.smiles_list[indices]
        graphs_in_batch = [
            self.graph_featurizer.compute(smi, cache_dir=self.scratch_dir)
            for smi in smiles_to_fetch
        ]

        graphs = Batch.from_data_list(graphs_in_batch)

        return graphs, fingerprints

    def __getitem__(self, idx):
        batch_plan = self._get_next_batch(idx)

        num_sets = len(batch_plan)
        all_indices_list = []
        all_labels_list = []
        set_sizes = []
        real_assay = []

        for item in batch_plan:
            set_type = item[0]
            indices = item[1]
            set_sizes.append(len(indices))
            all_indices_list.append(indices)
            real_assay.extend([set_type == "assay"] * len(indices))

            if set_type == "assay":
                all_labels_list.append(self.assay_labels[indices])
            else:
                coeffs_tensor = item[2]
                labels = self._compute_property_labels(indices, coeffs_tensor)
                all_labels_list.append(labels)

        all_indices = np.concatenate(all_indices_list)
        all_labels = torch.cat(all_labels_list)

        info = self.info[all_indices]

        graphs, fingerprints = self._get_ligand_features(all_indices)

        set_sizes_tensor = torch.tensor(set_sizes, dtype=torch.long)
        set_boundaries = torch.cat(
            [torch.tensor([0]), torch.cumsum(set_sizes_tensor, 0)]
        ).numpy()
        set_ids_tensor = torch.repeat_interleave(
            torch.arange(num_sets), set_sizes_tensor
        )

        metadata = {
            "num_sets": num_sets,
            "set_ids_tensor": set_ids_tensor,
            "set_boundaries": set_boundaries,
            "real_assay": torch.tensor(real_assay),
        }

        return (graphs, fingerprints), all_labels, info, metadata


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
