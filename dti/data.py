from typing import List, Union, Iterator, Tuple
import functools
import logging
from pathlib import Path
from multiprocessing import Pool

import pandas as pd
import numpy as np
import tqdm.auto as tqdm

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

import torch
from torch.utils.data import Dataset
from esm import FastaBatchedDataset, pretrained
from sklearn.preprocessing import StandardScaler

from .constants import DATA, SMILES, ACT, TID, SEQUENCE, ASSAY, COMPOUND
from .utils import device
from .hodge_ranking import parallel_hodge_rank

logger = logging.getLogger(__name__)


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

    logger.info("setting up ESM model")
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
            enumerate(data_loader), total=len(batches)
        ):
            toks = toks.to(device, non_blocking=True)

            out = model(toks, repr_layers=repr_layers, return_contacts=False)

            representations = {
                layer: t.to(device="cpu") for layer, t in out["representations"].items()
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


def load_kinodata(
    kinodata_path: Path = DATA / "raw" / "activities-chembl33_v0.5.csv",
    activity_types: List[str] = ["pIC50"],
) -> pd.DataFrame:
    logger.info(f"loading kinodata activities from {kinodata_path}")
    data = pd.read_csv(kinodata_path, index_col=0)
    data = data[data["activities.standard_type"].isin(activity_types)]
    data = data[~data["compound_structures.canonical_smiles"].isna()]
    data[ASSAY] = data["assays.chembl_id"].str[6:].astype(int)
    data[COMPOUND] = data["molecule_dictionary.chembl_id"].str[6:].astype(int)
    return data.rename(
        columns={
            "activities.standard_value": ACT,
            "compound_structures.canonical_smiles": SMILES,
            "component_sequences.sequence": SEQUENCE,
            "UniprotID": TID,
        }
    )


def load_landrum(landrum_path: Path = DATA / "raw" / "landrum.csv") -> pd.DataFrame:
    logger.info(f"loading landrum data from {landrum_path}")
    data = pd.read_csv(landrum_path, index_col=0)
    data = data[~data["canonical_smiles"].isna()]
    return data.rename(
        columns={
            "molregno": COMPOUND,
            "pchembl_value": ACT,
            "canonical_smiles": SMILES,
            "component_sequence": SEQUENCE,
            "tid": TID,
            "assay_id": ASSAY,
        }
    )


def load_atcc(path: Path = DATA / "raw" / "atcc.csv") -> pd.DataFrame:
    logger.info(f"loading NCI ATCC data from {path}")
    data = pd.read_csv(path, index_col=0)
    assay_ids = {exp: i for i, exp in enumerate(data["EXPID"].unique())}
    data[ASSAY] = data["EXPID"].map(assay_ids.get)
    return data.rename(
        columns={
            "NSC": COMPOUND,
            "IC50": ACT,
            "SMILES": SMILES,
        }
    )


def aggregate_multi_measurements(data: pd.DataFrame) -> pd.DataFrame:
    keys = [COMPOUND, ASSAY]
    if TID in data.columns:
        keys += [TID]
    non_numeric_cols = data.select_dtypes(exclude=["number"]).columns
    return (
        data.groupby(keys, as_index=False)
        .agg({ACT: "mean", **{col: lambda x: x.iloc[0] for col in non_numeric_cols}})
        .reset_index()
    )


def split_kfold_by(
    kinodata: pd.DataFrame, k: int, column: str, seed: int = 1
) -> np.ndarray:
    """Return the k-fold partitioning of `kinodata[column]` in shape `(k, -1)`."""
    values = kinodata[column].unique()
    missing_modk = k - len(values) % k
    values = np.concatenate((values, [-1] * missing_modk))
    np.random.seed(seed)
    np.random.shuffle(values)
    return values.reshape(k, -1)


def compute_fp(smi: str):
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=2048)
    try:
        return mfpgen.GetFingerprintAsNumPy(Chem.MolFromSmiles(smi))
    except TypeError:
        logger.warn(f"No fp for SMILES={smi}")
        return None


class ActivityDataset(Dataset):
    def __init__(
        self,
        kinodata: pd.DataFrame,
        target: str = ACT,
        info_cols: List[str] = [],
        model_name: str = "esm2_t33_650M_UR50D",
    ):
        super().__init__()
        logger.info(f"creating dataset of size {len(kinodata)}")
        logger.info("computing fingerprints")
        with Pool(16) as p:
            fps = p.map(compute_fp, kinodata[SMILES].values)
        mask = [fp is not None for fp in fps]
        if len(mask) - sum(mask) > 0:
            logger.info(
                f"dropping {len(mask) - sum(mask)}/{len(mask)} data points w/o FP"
            )
        self.kinodata = kinodata[mask].copy()
        self.kinodata.reset_index(inplace=True)
        self.ligand_features = torch.tensor(
            np.stack([fp for fp in fps if fp is not None]), dtype=torch.float32
        )
        self.protein_features = self._compute_protein_features(
            self.kinodata, model_name
        )
        self.labels = torch.tensor(self.kinodata[target].values, dtype=torch.float32)
        self.info_cols = info_cols
        logger.info(f"info cols: {info_cols}")
        self.info = torch.tensor(self.kinodata[info_cols].values)

    def _compute_protein_features(self, data: pd.DataFrame, model_name: str):
        if TID not in data.columns:
            logger.info("no protein target in dataset")
            return None

        logger.info(f"computing protein features: {model_name}")
        done = []
        fasta_file = DATA / "data.fasta"
        if not fasta_file.exists():
            with open(fasta_file, "w") as f:
                for _, row in data.iterrows():
                    uniprot = row["TID"]
                    if uniprot in done:
                        continue
                    f.write(f">{uniprot}\n{row[SEQUENCE]}\n")
                    done.append(uniprot)

        output_dir = DATA / model_name
        output_dir.mkdir(exist_ok=True)
        extract_embeddings(model_name, fasta_file, output_dir)

        @functools.cache
        def load_esm(uniprot_id: str) -> torch.Tensor:
            return torch.load(output_dir / f"{uniprot_id}.pt", weights_only=False)[
                "representation"
            ][33]

        return torch.stack([load_esm(uniprot_id) for uniprot_id in data[TID]])

    @property
    def weights(self):
        return torch.ones(len(self.labels))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.ones(1)
            if self.protein_features is None
            else self.protein_features[idx],
            self.ligand_features[idx],
            self.labels[idx],
            self.info[idx],
        )


class PairDataset(ActivityDataset):
    def __init__(
        self,
        kinodata: pd.DataFrame,
        **kwargs,
    ):
        super().__init__(kinodata, **kwargs)
        self.pairs = list()
        weights = list()
        for assay, group in self.kinodata.groupby(ASSAY):
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
        return self._weights

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        i, j = self.pairs[idx]
        return (
            torch.ones(1)
            if self.protein_features is None
            else self.protein_features[i],
            torch.cat([self.ligand_features[i], self.ligand_features[j]]),
            self.labels[i] - self.labels[j],
            torch.cat([self.info[i], self.info[j]]),
        )


def split_data(
    data: pd.DataFrame,
    target_dir: Path = DATA / "processed",
    k: int = 5,
    random_valset: bool = False,
    col: str = ASSAY,
):
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


def normalize_activity(data: pd.DataFrame, target_col: str, scaler: StandardScaler):
    """Normalize activity data to a standard normal distribution."""
    data[target_col] = scaler.transform(data[ACT].values.reshape(-1, 1))
    return data


def prepare_datasets(
    data: pd.DataFrame,
    data_dir: Path,
    tgt_name: str,
    k: int,
    inter_assay_weight: Union[float, None],
    random_valset: bool = False,
) -> Iterator[
    Tuple[int, pd.DataFrame, Union[pd.DataFrame, None], pd.DataFrame, pd.DataFrame]
]:
    """Prepare train, validation, and test datasets."""
    data = aggregate_multi_measurements(data)
    split_data(data, data_dir, k=k, random_valset=random_valset)
    for index in range(k):
        split_dir = data_dir / f"{index}"
        logger.info(f"reading dataset from {split_dir}")

        val_data = pd.read_csv(split_dir / "val.csv", index_col=0)
        train_data = pd.read_csv(split_dir / "train.csv", index_col=0)
        test_data = pd.read_csv(split_dir / "test.csv", index_col=0)

        scaler = StandardScaler()
        train_data[tgt_name] = scaler.fit_transform(
            train_data[ACT].values.reshape(-1, 1)
        )
        test_data = normalize_activity(test_data, tgt_name, scaler)
        val_data = normalize_activity(val_data, tgt_name, scaler)

        if inter_assay_weight is not None:
            hodge_file = (
                split_dir / f"train_hodge_no_assay_norm_lam{inter_assay_weight:.2f}.csv"
            )
            if not hodge_file.exists():
                logger.info("computing Hodge ranking")
                hodge_df = parallel_hodge_rank(train_data, inter_assay_weight)
                train_data = train_data.merge(
                    hodge_df,
                    on=["compound_structures.canonical_smiles", "UniprotID"],
                    how="inner",
                )
                train_data.to_csv(hodge_file)
            else:
                logger.info(f"cached Hodge ranking data at {hodge_file}")
                train_data = pd.read_csv(hodge_file, index_col=0)

        yield index, train_data, val_data, test_data
