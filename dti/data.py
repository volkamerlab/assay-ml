from typing import List, Union, Iterator, Tuple
import functools
import logging
from pathlib import Path
from multiprocessing import Pool

import pandas as pd
import numpy as np
import tqdm.auto as tqdm


import torch
from torch.utils.data import Dataset
from esm import FastaBatchedDataset, pretrained
from sklearn.preprocessing import StandardScaler

from .constants import DATA, SMILES, ACT, TID, SEQUENCE, ASSAY, COMPOUND, HODGE
from .utils import device, compute_fp
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
        target: str = ACT,
        info_cols: List[str] = [],
        model_name: str = "esm2_t33_650M_UR50D",
        n_jobs: int = 16,
    ):
        super().__init__()
        logger.info(f"creating dataset of size {len(data)}")
        logger.info("computing fingerprints")
        with Pool(n_jobs) as p:
            fps = p.map(compute_fp, data[SMILES].values)
        mask = [fp is not None for fp in fps]
        if len(mask) - sum(mask) > 0:
            logger.info(
                f"dropping {len(mask) - sum(mask)}/{len(mask)} data points w/o FP"
            )
        self.data = data[mask].copy()
        self.data.reset_index(inplace=True)
        self.ligand_features = torch.tensor(
            np.stack([fp for fp in fps if fp is not None]), dtype=torch.float32
        )
        self.protein_features = self._compute_protein_features(self.data, model_name)
        self.labels = torch.tensor(self.data[target].values, dtype=torch.float32)
        self.info_cols = info_cols
        logger.info(f"info cols: {info_cols}")
        self.info = torch.tensor(self.data[info_cols].values)

    def _compute_protein_features(self, data: pd.DataFrame, model_name: str):
        """Compute protein embeddings using the specified ESM model.

        Args:
            data (pd.DataFrame): DataFrame containing protein sequences.
            model_name (str): Name of the ESM model to use.

        Returns:
            torch.Tensor or None: Tensor of protein embeddings or None if no protein targets.
        """
        if TID not in data.columns or data[TID].isna().any():
            logger.info("missing protein target in dataset")
            return None

        logger.info(f"computing protein features: {model_name}")
        done = []
        output_dir = DATA / model_name
        output_dir.mkdir(exist_ok=True)
        emb_dir = lambda uniprot_id: output_dir / f"{uniprot_id}.pt"
        fasta_file = DATA / "data.fasta"
        with open(fasta_file, "w") as f:
            for _, row in data.iterrows():
                uniprot = row[TID]
                if emb_dir(uniprot).exists():
                    continue
                if uniprot in done:
                    continue
                f.write(f">{uniprot}\n{row[SEQUENCE]}\n")
                done.append(uniprot)

        extract_embeddings(model_name, fasta_file, output_dir)

        @functools.cache
        def load_esm(uniprot_id: str) -> torch.Tensor:
            """Load ESM embeddings for a specific protein.

            Args:
                uniprot_id (str): UniProt identifier for the protein.

            Returns:
                torch.Tensor: Protein embedding tensor.
            """
            emb = torch.load(
                emb_dir(uniprot_id), weights_only=False, map_location=device
            )
            return emb["representation"][33].cpu()

        return torch.stack([load_esm(uniprot_id) for uniprot_id in data[TID]])

    @property
    def weights(self):
        """Get sample weights for the dataset.

        Returns:
            torch.Tensor: Uniform weights for all samples.
        """
        return torch.ones(len(self.labels))

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
            torch.ones(1)
            if self.protein_features is None
            else self.protein_features[idx].cpu()
        )
        return (
            prot_feats,
            self.ligand_features[idx].cpu(),
            self.labels[idx].cpu(),
            self.info[idx].cpu(),
            torch.ones(1),
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
        assert len(weights) == len(self.pairs), (len(weights), len(self.pairs))
        self._weights = torch.tensor(weights, dtype=torch.double)
        self.info_cols = [col + "_a" for col in self.info_cols] + [
            col + "_b" for col in self.info_cols
        ]

    @property
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
        i, j = self.pairs[idx]
        prot_feats = (
            torch.ones(1)
            if self.protein_features is None
            else self.protein_features[i].cpu()
        )
        return (
            prot_feats,
            torch.stack([self.ligand_features[i].cpu(), self.ligand_features[j].cpu()]),
            self.labels[i].cpu() - self.labels[j].cpu(),
            torch.cat([self.info[i].cpu(), self.info[j].cpu()]),
            self.weights[i].cpu(),
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
        return torch.ones(len(self))

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
            torch.ones(1)
            if self.protein_features is None
            else self.protein_features[batch_idcs]
        )
        return (
            prot_feats,
            self.ligand_features[batch_idcs],
            self.labels[batch_idcs],
            self.info[batch_idcs],
            torch.ones(1),
        )


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


def extract_embeddings(
    model_name: str,
    fasta_file: Union[Path, str],
    output_dir: Path,
    tokens_per_batch: int = 4096,
    seq_length: int = 5000,
    repr_layers: List[int] = [33],
):
    # adapted from https://www.kaggle.com/code/viktorfairuschin/extracting-esm-2-embeddings-from-fasta-files

    dataset = FastaBatchedDataset.from_file(fasta_file)
    filename = lambda tid: output_dir / f"{tid}.pt"
    data = [
        (label, seq)
        for label, seq in zip(dataset.sequence_labels, dataset.sequence_strs)
        if not filename(label).exists()
    ]
    dataset.sequence_labels = [label for label, _ in data]
    dataset.sequence_strs = [seq for _, seq in data]
    if len(data) == 0:
        return

    logger.info("setting up ESM model '{model_name}'")
    model, alphabet = pretrained.load_model_and_alphabet(model_name)
    model.eval()

    if torch.cuda.is_available():
        model = model.cuda()

    logger.info(f"computing ESM embeddings for {len(data)} sequences")
    batches = dataset.get_batch_indices(tokens_per_batch, extra_toks_per_seq=1)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=alphabet.get_batch_converter(seq_length),
        batch_sampler=batches,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for _, (labels, strs, toks) in tqdm.tqdm(
            enumerate(data_loader), total=len(batches), desc="extract embeddings"
        ):
            toks = toks.to(device, non_blocking=True)

            out = model(toks, repr_layers=repr_layers, return_contacts=False)

            representations = {
                layer: t.to(device) for layer, t in out["representations"].items()
            }

            for i, label in enumerate(labels):
                entry_id = label.split()[0]
                truncate_len = min(seq_length, len(strs[i]))
                result = {"entry_id": entry_id}
                result["representation"] = {
                    layer: t[i, 1 : truncate_len + 1].mean(0).clone()
                    for layer, t in representations.items()
                }

                torch.save(result, filename(entry_id))
    logger.info("ESM embeddings written to {output_dir}")


def split_kfold_by(
    data: pd.DataFrame, k: int, column: str, seed: int = 1
) -> np.ndarray:
    """Return the k-fold partitioning of `data[column]` in shape `(k, -1)`."""
    values = data[column].unique()
    missing_modk = k - len(values) % k
    values = np.concatenate((values, [-1] * missing_modk))
    np.random.seed(seed)
    np.random.shuffle(values)
    return values.reshape(k, -1)


def split_data(
    data: pd.DataFrame,
    target_dir: Path = DATA / "processed",
    k: int = 5,
    random_valset: bool = False,
    col: str = ASSAY,
):
    logger.info(f"computing split and saving to {target_dir}")
    if (target_dir / "0").exists():
        return target_dir
    target_dir.mkdir(exist_ok=True, parents=True)

    partition = split_kfold_by(data, column=col, k=k)

    for index in range(k):
        split_dir = target_dir / f"{index}"
        split_dir.mkdir()
        data[data[ASSAY].isin(partition[index])].to_csv(split_dir / "test.csv")
        rest = data[~data[ASSAY].isin(partition[index])]
        if random_valset:
            logger.info(f"random validation set for split {index}")
            idcs = np.arange(len(rest))
            np.random.shuffle(idcs)
            split = len(rest) // 8
            rest.iloc[idcs[:split]].to_csv(split_dir / "val.csv")
            rest.iloc[idcs[split:]].to_csv(split_dir / "train.csv")
        else:
            logger.info(f"assay-split validation set for split {index}")
            val_assays = partition[(index + 1) % k][: partition.shape[1] // 2]
            rest[rest[ASSAY].isin(val_assays)].to_csv(split_dir / "val.csv")
            rest[~rest[ASSAY].isin(val_assays)].to_csv(split_dir / "train.csv")

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
) -> Iterator[
    Tuple[int, pd.DataFrame, Union[pd.DataFrame, None], pd.DataFrame, pd.DataFrame]
]:
    """Prepare train, validation, and test datasets."""
    if aggregate:
        data = aggregate_multi_measurements(data)
    split_data(data, data_dir, k=k, random_valset=random_valset)


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
    col_map = {
        "activities.standard_value": ACT,
        "compound_structures.canonical_smiles": SMILES,
        "component_sequences.sequence": SEQUENCE,
        "UniprotID": TID,
    }
    assert all(k in data.columns for k in col_map.keys()), data.columns
    return data.rename(columns=col_map)


def load_landrum(landrum_path: Path = DATA / "raw" / "landrum.csv") -> pd.DataFrame:
    logger.info(f"loading data from {landrum_path}")
    data = pd.read_csv(landrum_path, index_col=0)
    data = data[~data["canonical_smiles"].isna()]
    col_map = {
        "molregno": COMPOUND,
        "pchembl_value": ACT,
        "canonical_smiles": SMILES,
        "component_sequence": SEQUENCE,
        "tid": TID,
        "assay_id": ASSAY,
    }
    assert all(k in data.columns for k in col_map.keys()), data.columns
    return data.rename(columns=col_map)


def load_nci(path: Path = DATA / "raw" / "atcc.csv") -> pd.DataFrame:
    logger.info(f"loading NCI ATCC data from {path}")
    data = pd.read_csv(path, index_col=0)
    assay_ids = {exp: i for i, exp in enumerate(data["EXPID"].unique())}
    data[ASSAY] = data["EXPID"].map(assay_ids.get)
    col_map = {
        "NSC": COMPOUND,
        "IC50": ACT,
        "SMILES": SMILES,
    }
    assert all(k in data.columns for k in col_map.keys()), data.columns
    return data.rename(columns=col_map)
