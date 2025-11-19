import os
import functools
import logging
from pathlib import Path
from threading import Lock

import pandas as pd
import numpy as np
from rdkit import Chem
from tqdm.auto import tqdm


import torch
from torch.utils.data import Dataset, Sampler
from torch_geometric.data import Batch
from sklearn.preprocessing import StandardScaler

from .featurization import MolFingerprint, esm2_features
from ..utils.constants import (
    SMILES,
    ACT,
    TID,
    ASSAY,
    INTRA_ASSAY_TEST,
)
from ..utils import device

logger = logging.getLogger(__name__)


class BaseDataModule(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        mol_featurizer: MolFingerprint,
        target: str = ACT,
        info_cols: list[str] = None,
        cache_dir: Path | str | None = None,
        n_jobs: int = 16,
        **kwargs,
    ):
        Dataset.__init__(self)
        logger.info(f"Initializing BaseDataModule with {len(data)} data points.")
        self.mol_featurizer = mol_featurizer
        self.info_cols = info_cols or []
        self.target = target

        self._load_and_featurize(data, cache_dir, n_jobs)

    def _load_and_featurize(self, data, cache_dir=None, n_jobs=16):
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            paths = {
                "ligand_features": os.path.join(
                    cache_dir, f"ligand_features_{self.mol_featurizer.dim}.pt"
                ),
                "labels": os.path.join(cache_dir, f"{self.target}_labels.pt"),
                "info": os.path.join(cache_dir, "info.pt"),
                "data": os.path.join(cache_dir, "data.csv"),
            }

            if all(os.path.exists(p) for p in paths.values()):
                logger.info(f"Loading cached features from {cache_dir}")
                self.ligand_features = torch.load(paths["ligand_features"])
                self.labels = torch.load(paths["labels"])
                self.info = torch.load(paths["info"])
                self.data = pd.read_csv(paths["data"])
                return

        logger.info("Computing molecular fingerprints...")
        fps = self.mol_featurizer.compute_parallel(data[SMILES].values, n_jobs=n_jobs)
        mask = [fp is not None for fp in fps]
        if (n_invalid := len(mask) - sum(mask)) > 0:
            logger.info(f"Dropping {n_invalid}/{len(mask)} invalid fingerprints.")

        self.data = data[mask].copy().reset_index(drop=True)
        self.ligand_features = torch.tensor(
            np.stack([fp for fp in fps if fp is not None]), dtype=torch.float32
        )
        assert self.ligand_features.shape[1] == self.mol_featurizer.dim, (
            self.ligand_features.shape,
            self.mol_featurizer.dim,
            self.mol_featurizer,
        )
        self.labels = torch.tensor(self.data[self.target].values, dtype=torch.float32)

        missing_info_cols = [c for c in self.info_cols if c not in self.data.columns]
        if len(missing_info_cols) > 0:
            logger.warn(f"Missing info cols, will be ignored: {missing_info_cols}")
            self.info_cols = [c for c in self.info_cols if c not in missing_info_cols]

        self.info = (
            torch.tensor(self.data[self.info_cols].values.astype(np.int64))
            if self.info_cols
            else torch.empty((len(self.data), 0), dtype=torch.int64)
        )

        if cache_dir:
            logger.info(f"Caching features to {cache_dir}")
            torch.save(self.ligand_features, paths["ligand_features"])
            torch.save(self.labels, paths["labels"])
            torch.save(self.info, paths["info"])
            self.data.to_csv(paths["data"], index=False)

    def _shuffle(self):
        indices = torch.randperm(self.labels.shape[0])
        self.ligand_features = self.ligand_features[indices]
        self.labels = self.labels[indices]
        self.info = self.info[indices]
        self.data = self.data.iloc[indices]
        return indices

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.ligand_features[idx], self.labels[idx], self.info[idx]


class ProteinFeaturesMixin:
    def __init__(self, model_name: str = "esm2_t33_650M_UR50D", **kwargs):
        logger.info("Initializing ProteinFeaturesMixin...")
        if not hasattr(self, "data"):
            raise TypeError("'self.data' needs to be set by a base class.")

        self.protein_features = esm2_features(self.data, model_name=model_name)

    def _reorder(self, indices):
        if self.protein_features is not None:
            assert len(indices) == len(self.protein_features)
            self.protein_features = self.protein_features[indices]

    def _get_protein_features(self, indices):
        """Helper to get protein features, with a fallback for None."""
        if self.protein_features is None:
            return torch.ones(len(indices))
        return self.protein_features[indices]


class AssaySetGroupingMixin:
    def __init__(
        self,
        min_batch_size: int = 3,
        max_set_size: int = 0,
        random_seed: int = 0,
        **kwargs,
    ):
        logger.info("Initializing AssaySetGroupingMixin...")
        if not hasattr(self, "data"):
            raise TypeError("'self.data' needs to be set by a base class.")

        self.min_batch_size = min_batch_size
        self.max_set_size = max_set_size
        self.random = np.random.default_rng(random_seed)

        self._group_by_assay()

    def _group_by_assay(self):
        self.assay_sets = []
        self.assay_set_ids = []
        grouped = self.data.groupby(ASSAY, sort=False)
        num_unused = 0

        for assay_id, (_, group) in enumerate(grouped):
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
                        self.assay_set_ids.append(assay_id)
                    else:
                        num_unused += len(batch)
            else:
                self.assay_sets.append(group_idcs)
                self.assay_set_ids.append(assay_id)

        self.assay_set_ids = np.array(self.assay_set_ids, dtype=np.int32)
        logger.info(
            f"Created {len(self.assay_sets)} assay sets. "
            f"({num_unused}/{len(self.data)} datapoints unused)"
        )


class ActivityDataset(BaseDataModule, ProteinFeaturesMixin):
    def __init__(
        self,
        data: pd.DataFrame,
        mol_featurizer: MolFingerprint,
        target: str = ACT,
        info_cols: list[str] = [],
        model_name: str = "esm2_t33_650M_UR50D",
        **kwargs,
    ):
        BaseDataModule.__init__(
            self,
            data=data,
            mol_featurizer=mol_featurizer,
            target=target,
            info_cols=info_cols,
            **kwargs,
        )
        ProteinFeaturesMixin.__init__(self, model_name=model_name, **kwargs)

    @functools.cached_property
    def weights(self):
        return torch.ones(len(self.labels))

    def __getitem__(self, idx):
        lig_feat, label, info = super().__getitem__(idx)

        prot_feat = self._get_protein_features([idx]).squeeze(0)

        return (
            prot_feat,
            lig_feat,
            label,
            info,
        )


class MultiSetActivityDataset(
    BaseDataModule, ProteinFeaturesMixin, AssaySetGroupingMixin
):
    def __init__(
        self,
        data: pd.DataFrame,
        mol_featurizer: MolFingerprint,
        target: str = ACT,
        info_cols: list[str] = [],
        model_name: str = "esm2_t33_650M_UR50D",
        min_batch_size: int = 3,
        max_set_size: int = 0,
        max_batch_datapoints: int = 2048,
        random_seed: int = 0,
        inter_assay: bool = False,
        **kwargs,
    ):
        BaseDataModule.__init__(
            self,
            data=data,
            mol_featurizer=mol_featurizer,
            target=target,
            info_cols=info_cols,
            **kwargs,
        )
        ProteinFeaturesMixin.__init__(self, model_name=model_name, **kwargs)

        AssaySetGroupingMixin.__init__(
            self,
            min_batch_size=min_batch_size,
            max_set_size=max_set_size,
            random_seed=random_seed,
            **kwargs,
        )

        self.max_batch_datapoints = max_batch_datapoints
        self.batches = []
        self.batch_set_ids = []

        self.inter_assay = inter_assay

        self._make_batches()

    def _make_batches(self):
        num_sets = len(self.assay_sets)
        indices = self.random.permutation(num_sets)

        shuffled_valid_sets = [self.assay_sets[i] for i in indices]
        shuffled_set_ids = [self.assay_set_ids[i] for i in indices]

        self.batches = []
        self.batch_set_ids = []

        while shuffled_valid_sets:
            current_batch_sets_local = []
            current_batch_ids_local = []
            current_total_datapoints = 0

            while shuffled_valid_sets:
                assay_set = shuffled_valid_sets[-1]
                set_size = len(assay_set)

                if (
                    current_total_datapoints + set_size <= self.max_batch_datapoints
                ) or (current_total_datapoints == 0):
                    current_total_datapoints += set_size
                    current_batch_sets_local.append(shuffled_valid_sets.pop())
                    current_batch_ids_local.append(shuffled_set_ids.pop())

                    if (
                        set_size > self.max_batch_datapoints
                        and len(current_batch_sets_local) == 1
                    ):
                        logger.warning(
                            f"Creating oversized batch. Single set of size {set_size} "
                            f"> {self.max_batch_datapoints}"
                        )
                        break
                else:
                    break

            if current_batch_sets_local:
                self.batches.append(current_batch_sets_local)
                self.batch_set_ids.append(np.array(current_batch_ids_local))

        self.used = np.zeros(len(self.batches), dtype=bool)
        logger.debug(
            f"created {len(self.batches)} batches with up to"
            f"{self.max_batch_datapoints} datapoints each"
        )
        if self.inter_assay:
            logger.info("Shuffling data for inter-assay sets")
            indices = self._shuffle()
            self._reorder(indices)

    def _get_next_batch(self, idx: int):
        if self.used.all():
            logger.debug("Epoch complete, remaking batches.")
            self._make_batches()

        if idx >= len(self.batches):
            logger.warning(f"Index {idx} out of bounds. Wrapping around.")
            idx = idx % len(self.batches)

        self.used[idx] = True
        return self.batches[idx], self.batch_set_ids[idx]

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        """Returns a BATCH of data, including protein features."""
        batch_sets, batch_ids = self._get_next_batch(idx)

        set_sizes = np.array([len(s) for s in batch_sets], dtype=np.int32)
        cumulative_sizes = np.concatenate([[0], np.cumsum(set_sizes)])

        all_indices = np.concatenate(batch_sets)

        set_ids_tensor = torch.from_numpy(
            np.repeat(np.arange(len(set_sizes), dtype=np.int64), set_sizes)
        )

        prot_feats = self._get_protein_features(all_indices)
        lig_feats = self.ligand_features[all_indices]
        labels = self.labels[all_indices]
        info = self.info[all_indices]

        return (
            prot_feats,
            lig_feats,
            labels,
            info,
            {
                "set_boundaries": cumulative_sizes,
                "set_ids": batch_ids,
                "set_ids_tensor": set_ids_tensor,
                "num_sets": len(batch_sets),
            },
        )


class PropertySetDataset(BaseDataModule, AssaySetGroupingMixin):
    def __init__(
        self,
        data: pd.DataFrame,
        mol_featurizer: MolFingerprint,
        target: str = ACT,
        property_columns: list[str] = None,
        info_cols: list[str] = [],
        query_col: str | None = INTRA_ASSAY_TEST,
        mask_fraction: float = 0.2,
        cache_dir: Path | None = None,
        label_normalization: str = "none",
        min_batch_size: int = 3,
        max_set_size: int = 0,
        random_seed: int = 0,
        property_set_ratio: float = 0.5,
        noise_std: float = 0.1,
        max_batch_datapoints: int = 2048,
        **kwargs,
    ):
        BaseDataModule.__init__(
            self,
            data=data,
            mol_featurizer=mol_featurizer,
            target=target,
            info_cols=info_cols,
            cache_dir=cache_dir,
            **kwargs,
        )

        AssaySetGroupingMixin.__init__(
            self,
            min_batch_size=min_batch_size,
            max_set_size=max_set_size,
            random_seed=random_seed,
            **kwargs,
        )

        if label_normalization not in ["none", "zscore", "minmax"]:
            raise ValueError(
                f"Invalid label_normalization mode: {label_normalization}. "
                "Must be 'none', 'zscore', or 'minmax'."
            )
        self.label_normalization = label_normalization
        self.query_col = query_col
        self.mask_fraction = mask_fraction
        self.lock = Lock()

        self.query_mask = (
            torch.from_numpy(self.data[self.query_col].values).bool().flatten()
            if self.query_col is not None and self.query_col in self.data.columns
            else torch.zeros(len(self.data), dtype=torch.bool)
        )

        self.property_set_ratio = property_set_ratio
        self.noise_std = noise_std
        self.max_batch_datapoints = max_batch_datapoints
        self.property_columns = property_columns or []

        self._prepare_property_data()

        self.property_coeff_pool_size = 32
        self.coeff_drift_alpha = 0.9
        self._init_property_coeff_pool()

        self.batches_plan = None
        self.used_batches = None

    def __getitem__(self, idx):
        batch_plan = self._get_next_batch(idx)

        num_sets = len(batch_plan)
        all_indices_list = []
        all_labels_list = []
        set_sizes = []
        real_assay = []
        query_mask_list = []

        for item in batch_plan:
            set_type = item[0]
            indices = item[1]
            set_size = len(indices)
            set_sizes.append(set_size)
            all_indices_list.append(indices)
            real_assay.extend([set_type == "assay"] * set_size)

            if set_type == "assay":
                labels = self.labels[indices].clone()
            else:
                coeffs_tensor = item[2]
                labels = self._compute_property_labels(indices, coeffs_tensor)

            if self.query_col is not None:
                set_query_mask = self.query_mask[indices]
            else:
                set_query_mask = torch.zeros(set_size, dtype=torch.bool)
                n_masked = max(1, int(set_size * self.mask_fraction))
                mask_idx = torch.from_numpy(
                    self.random.permutation(set_size)[:n_masked]
                )
                set_query_mask[mask_idx] = True

            query_mask_list.append(set_query_mask)
            labels = self._normalize_labels(labels, set_query_mask)
            all_labels_list.append(labels)

        all_indices = np.concatenate(all_indices_list)
        all_labels = torch.cat(all_labels_list)

        info = self.info[all_indices]

        query_mask = torch.concat(query_mask_list).bool()

        lig_feats = self.ligand_features[all_indices]
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

        assert len(query_mask) == len(lig_feats), (len(query_mask), len(all_labels))
        return lig_feats, all_labels, info, query_mask, metadata

    def _normalize_labels(self, labels, set_query_mask):
        if self.label_normalization != "none":
            non_query_mask = ~set_query_mask
            non_query_labels = labels[non_query_mask]

            if non_query_labels.numel() > 0:
                if self.label_normalization == "zscore":
                    if non_query_labels.numel() > 1:
                        mean = non_query_labels.mean()
                        std = non_query_labels.std()
                        if std < 1e-6:
                            labels = labels - mean
                        else:
                            labels = (labels - mean) / std
                    else:
                        logger.warning("set too small for zscore normalization")
                        labels = labels - non_query_labels.mean()

                elif self.label_normalization == "minmax":
                    min_val = non_query_labels.min()
                    max_val = non_query_labels.max()
                    data_range = max_val - min_val

                    if data_range > 1e-6:
                        labels = (labels - min_val) / data_range
                    else:
                        labels = labels - min_val
        return labels

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
                if (current_total_datapoints + set_size <= max_datapoints) or (
                    current_total_datapoints == 0
                ):
                    current_total_datapoints += set_size
                    num_assay_sets_in_batch += 1
                    current_batch_sets.append(("assay", shuffled_assay_sets.pop()))

                    if set_size > max_datapoints and len(current_batch_sets) == 1:
                        logger.warning(
                            "Assay set too large, creating oversized assay-only batch."
                        )
                        break
                else:
                    break

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
        with self.lock:
            if self.batches_plan is None:
                self.prepare_epoch()
            if self.used_batches.all():
                logger.debug("Epoch complete, preparing new epoch.")
                self.prepare_epoch()

            if idx >= len(self.batches_plan):
                logger.warning(f"Index {idx} out of bounds. Wrapping around")
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

    def _load_and_featurize(self, data, cache_dir, n_jobs, **kwargs):
        logger.info("Initializing GraphPropertySetDataset (on-the-fly featurization).")

        cache_dir_str = os.environ.get("GRAPH_CACHE_DIR", "/tmp/graph_cache")
        self.scratch_dir = Path(cache_dir_str)
        self.scratch_dir.mkdir(exist_ok=True, parents=True)
        logger.info(f"Using sharded graph cache at: {self.scratch_dir}")

        self.data = data.reset_index(drop=True)

        self.smiles_list = self.data[SMILES].values

        self.labels = torch.tensor(self.data[self.target].values, dtype=torch.float32)

        missing_info_cols = [c for c in self.info_cols if c not in self.data.columns]
        if len(missing_info_cols) > 0:
            logger.warn(f"Missing info cols, will be ignored: {missing_info_cols}")
            self.info_cols = [c for c in self.info_cols if c not in missing_info_cols]

        self.info = (
            torch.tensor(self.data[self.info_cols].values.astype(np.int64))
            if self.info_cols
            else torch.empty((len(self.data), 0), dtype=torch.int64)
        )

        self.deg_histogram = None
        if self.__dict__.get("estimate_deg", False):
            self.deg_histogram = _estimate_degree_histogram(
                self.smiles_list,
            )

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
        self.deg_histogram = None
        if kwargs.get("estimate_deg", False):
            self.deg_histogram = _estimate_degree_histogram(
                self.smiles_list,
            )

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
        query_mask_list = []

        for item in batch_plan:
            set_type = item[0]
            indices = item[1]
            set_size = len(indices)
            set_sizes.append(set_size)
            all_indices_list.append(indices)
            real_assay.extend([set_type == "assay"] * set_size)

            if set_type == "assay":
                labels = self.labels[indices].clone()
            else:
                coeffs_tensor = item[2]
                labels = self._compute_property_labels(indices, coeffs_tensor)

            if self.query_col is not None:
                set_query_mask = self.query_mask[indices]
            else:
                set_query_mask = torch.zeros(set_size, dtype=torch.bool)
                n_masked = max(1, int(set_size * self.mask_fraction))
                mask_idx = torch.from_numpy(
                    self.random.permutation(set_size)[:n_masked]
                )
                set_query_mask[mask_idx] = True

            query_mask_list.append(set_query_mask)
            labels = self._normalize_labels(labels, set_query_mask)
            all_labels_list.append(labels)

        all_indices = np.concatenate(all_indices_list)
        all_labels = torch.cat(all_labels_list)

        info = self.info[all_indices]
        query_mask = torch.concat(query_mask_list).bool()
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

        assert len(query_mask) == len(fingerprints), (
            len(query_mask),
            len(fingerprints),
        )

        return (graphs, fingerprints), all_labels, info, query_mask, metadata


class PairDataset(ActivityDataset):
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
        return self._weights

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
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


def _estimate_degree_histogram(
    smiles_list, count_h_atoms=False, sample_size=1_000_000, max_deg=5
):
    logger.info(f"Estimating degree histogram from a subsample of {sample_size}...")

    if len(smiles_list) <= sample_size:
        sample_smiles = smiles_list
        sample_size = len(smiles_list)
    else:
        sample_smiles = np.random.choice(smiles_list, sample_size, replace=False)

    all_degrees = []
    for smi in tqdm(sample_smiles, desc="Estimating degrees", leave=False):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        for atom in mol.GetAtoms():
            all_degrees.append(
                atom.GetTotalDegree() if count_h_atoms else atom.GetDegree()
            )

    all_degrees_tensor = torch.tensor(all_degrees, dtype=torch.long)
    deg_histogram = torch.bincount(all_degrees_tensor, minlength=max_deg + 1)
    logger.debug(f"complete histogram: {deg_histogram}")

    if len(deg_histogram) > max_deg + 1:
        logger.warning(
            f"Found degrees ({deg_histogram[max_deg + 1 :]}) "
            f"higher than {max_deg}, folding into last bin."
        )
        extra_degrees = deg_histogram[max_deg + 1 :].sum()
        deg_histogram = deg_histogram[: max_deg + 1]
        deg_histogram[max_deg] += extra_degrees

    deg_histogram = deg_histogram.to(torch.long)

    logger.info(
        f"Estimated degree histogram (n={sample_size}): {deg_histogram.numpy()}"
    )
    return deg_histogram
