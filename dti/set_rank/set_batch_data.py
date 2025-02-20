import numpy as np
from dti.data import ActivityDataset
from dti.constants import ASSAY
import logging
from tqdm import auto as tqdm

logger = logging.getLogger(__name__)


class SetActivityDataset(ActivityDataset):
    def __init__(
        self,
        data,
        target=...,
        info_cols=...,
        model_name="esm2_t33_650M_UR50D",
        min_batch_size: int | None = None,
        max_batch_size: int | None = None,
        random_seed: int = 0,
    ):
        super().__init__(data, target, info_cols, model_name)
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.random = np.random.default_rng(random_seed)
        self.groups_index = []
        for _, group in self.data.groupby(ASSAY):
            self.groups_index.append(group.index)
        self._make_batches()

    def _make_batches(self):
        self.groups_index = [self.random.shuffle(group) for group in self.groups_index]
        self.batches = []
        num_unused = 0
        for group in tqdm(self.groups_index, desc="Making batches..."):
            if len(group) < self.min_batch_size:
                num_unused += len(group)
                continue
            batches = np.array_split(group, len(group) // self.max_batch_size)
            self.batches.extend(batches[:-1])
            if batches[-1].size >= self.min_batch_size:
                self.batches.append(batches[-1])
                continue
            num_unused += batches[-1].shape[0]
        self.random.shuffle(self.batches)
        self.used = np.full(len(self.batches), False, dtype=bool)
        logger.info(f"Number of unused examples: {num_unused} / {len(self.data)}")

    def _get_next_batch(self, idx: int):
        if np.all(self.used):
            self._make_batches()
        self.used[idx] = True
        return self.batches[idx]

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        return (
            self.ligand_features[batch_idcs := self._get_next_batch(idx)],
            self.protein_features[batch_idcs],
            self.labels[batch_idcs],
        )
