from typing import Sequence, List, Union, NoReturn, Callable
import functools
import logging
from pathlib import Path

import tqdm
import pandas as pd
import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem, rdFingerprintGenerator

import torch
from torch.utils.data import DataLoader, Dataset
from torch import nn
from esm import FastaBatchedDataset, pretrained
from sklearn.preprocessing import StandardScaler

from .utils import DATA, SMILES, ACT, DEVICE

import logging

logger = logging.getLogger(__name__)


def extract_embeddings(
    model_name: str,
    fasta_file: Union[Path, str],
    output_dir: Union[Path, str],
    tokens_per_batch: int = 4096,
    seq_length: int = 5000,
    repr_layers: List[int] = [33],
):

    dataset = FastaBatchedDataset.from_file(fasta_file)
    filename = lambda uniprot_id: output_dir / f"{uniprot_id}.pt"
    data = [
        (label, seq)
        for label, seq in zip(dataset.sequence_labels, dataset.sequence_strs)
        if not filename(label).exists()
    ]
    dataset.sequence_labels = [label for label, _ in data]
    dataset.sequence_strs = [seq for _, seq in data]
    if len(data) == 0:
        return

    logger.info("Setting up ESM model")
    model, alphabet = pretrained.load_model_and_alphabet(model_name)
    model.eval()

    if torch.cuda.is_available():
        model = model.cuda()

    logger.info(f"Computing ESM embeddings for {len(data)} sequences")
    batches = dataset.get_batch_indices(tokens_per_batch, extra_toks_per_seq=1)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=alphabet.get_batch_converter(seq_length),
        batch_sampler=batches,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for batch_idx, (labels, strs, toks) in tqdm.tqdm(
            enumerate(data_loader), total=len(batches)
        ):
            toks = toks.to(DEVICE, non_blocking=True)

            out = model(toks, repr_layers=repr_layers, return_contacts=False)

            logits = out["logits"].to(device="cpu")
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
    logger.info(f"Loading kinodata activities from {kinodata_path}")
    data = pd.read_csv(kinodata_path, index_col=0)
    data = data[data["activities.standard_type"].isin(activity_types)]
    data = data[~data["compound_structures.canonical_smiles"].isna()]
    return data


def split_kfold_by(
    kinodata: pd.DataFrame, k: int, column: str, seed: int = 0
) -> np.ndarray:
    """Return the k-fold partitioning of `kinodata[column]` in shape `(k, -1)`."""
    values = kinodata[column].unique()
    missing_modk = k - len(values) % k
    values = np.concatenate((values, [-1] * missing_modk))
    np.random.seed(seed)
    np.random.shuffle(values)
    return values.reshape(k, -1)


class FingerprintFactory(Callable):
    def __init__(self, mfpgen=None):
        if mfpgen is None:
            self.mfpgen = rdFingerprintGenerator.GetMorganGenerator(
                radius=2, fpSize=2048
            )

    @functools.cache
    def __call__(self, smi: str) -> Union[str, NoReturn]:
        try:
            return torch.tensor(
                self.mfpgen.GetFingerprintAsNumPy(Chem.MolFromSmiles(smi)),
                dtype=torch.float32,
            )
        except TypeError:
            logger.warning(f"No fp for SMILES={smi}")
            return None


class ActivityDataset(Dataset):
    def __init__(
        self,
        kinodata: pd.DataFrame,
        target: str = ACT,
        info_cols: List[str] = [],
        fp_gen: Union[FingerprintFactory, None] = None,
    ):
        super().__init__()
        if fp_gen is None:
            self.fp_gen = FingerprintFactory()
        else:
            self.fp_gen = fp_gen
        logger.info("Computing fingerprints")
        self.ligand_features = torch.stack(
            [self.fp_gen(smi) for smi in tqdm.tqdm(kinodata[SMILES].values)]
        )
        self._compute_protein_features(kinodata)
        self.labels = torch.tensor(kinodata[target].values, dtype=torch.float32)
        self.info_cols = info_cols
        self.info = torch.tensor(kinodata[info_cols].values)

    def _compute_protein_features(self, data: pd.DataFrame):
        logger.info("Computing ESM embeddings")

        done = []
        fasta_file = DATA / "data.fasta"
        with open(fasta_file, "w") as f:
            for i, row in data.iterrows():
                uniprot = row["UniprotID"]
                if uniprot in done:
                    continue
                f.write(f">{uniprot}\n{row['component_sequences.sequence']}\n")
                done.append(uniprot)

        model_name = "esm2_t33_650M_UR50D"
        output_dir = DATA / model_name
        output_dir.mkdir(exist_ok=True)
        extract_embeddings(model_name, fasta_file, output_dir)

        @functools.cache
        def load_esm(uniprot_id: str) -> torch.Tensor:
            return torch.load(output_dir / f"{uniprot_id}.pt", weights_only=False)[
                "representation"
            ][33]

        self.protein_features = torch.stack(
            [load_esm(uniprot_id) for uniprot_id in data["UniprotID"]]
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.protein_features[idx],
            self.ligand_features[idx],
            self.labels[idx],
            self.info[idx],
        )


def split_kinodata(target_dir: Union[Path, str] = DATA / "processed", k: int = 10):
    target_dir = target_dir / "splits"
    if target_dir.exists():
        return target_dir
    target_dir.mkdir(exist_ok=True, parents=True)
    kinodata = load_kinodata()
    kinodata["assay_id"] = kinodata["assays.chembl_id"].str[6:].astype(int)

    col = "assay_id"
    partition = split_kfold_by(kinodata, column=col, k=k)

    for index in range(k):
        kinodata[kinodata["assay_id"].isin(partition[index])].to_csv(
            target_dir / f"{index}.csv"
        )

    return target_dir


def process_kinodata_default_dti(
    target_dir: Union[Path, str] = DATA / "processed", k: int = 5
):
    kinodata = load_kinodata()
    kinodata["assay_id"] = kinodata["assays.chembl_id"].str[6:].astype(int)

    col = "assay_id"
    partition = split_kfold_by(kinodata, column=col, k=k)

    for index in range(k):
        split_dir = target_dir / str(index)
        train_file = split_dir / "train.csv"
        test_file = split_dir / "test.csv"

        if train_file.exists() and test_file.exists():
            continue
        else:
            split_dir.mkdir(exist_ok=True, parents=True)
        train = ~kinodata[col].isin(partition[index])
        test = kinodata[col].isin(partition[index])

        scaler = StandardScaler()
        tgt_name = "scaled_ic50"
        kinodata.loc[train, tgt_name] = scaler.fit_transform(
            kinodata.loc[train, ACT].values.reshape(-1, 1)
        )
        kinodata[train].to_csv(train_file)
        kinodata.loc[test, tgt_name] = scaler.transform(
            kinodata.loc[test, ACT].values.reshape(-1, 1)
        )
        kinodata[test].to_csv(test_file)

    return target_dir
