import os
import uuid
import hashlib
import pickle
import tempfile
import logging
import functools
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union, Iterable, List, Tuple
from multiprocessing import Pool
from enum import StrEnum, auto

import torch
from torch.utils.data import get_worker_info
import tqdm.auto as tqdm
from esm import FastaBatchedDataset, pretrained
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator, MACCSkeys
from transformers import AutoTokenizer, AutoModel
import portalocker

from .constants import DATA, TID, SEQUENCE
from .utils import device


logger = logging.getLogger(__name__)
_mfpgen_cache: dict[Tuple, object] = {}
_cache_root = Path(os.getenv("FP_CACHE_DIR", Path.home() / ".cache" / "mol_fps"))
logger.debug(f"cache path: {_cache_root}")


def _get_worker_cache():
    """Return a worker-specific fingerprint generator cache."""
    info = get_worker_info()
    if info is None:
        return _mfpgen_cache
    wid = info.id
    if wid not in _mfpgen_cache:
        _mfpgen_cache[wid] = {}
    return _mfpgen_cache[wid]


class MolFingerprint(StrEnum):
    MORGAN = auto()
    RDKIT = auto()
    TOPOTORSION = auto()
    ATOMPAIR = auto()
    MACCS = auto()
    CHEMBERTA = auto()
    ALL = auto()

    def _cache_path(self, smi: str) -> Path:
        """Compute cache file path for a given SMILES."""
        key = f"{self.value}-{smi}".encode("utf-8")
        hexhash = hashlib.sha1(key).hexdigest()
        subdir = _cache_root / self.value / hexhash[:2]
        subdir.mkdir(parents=True, exist_ok=True)
        return subdir / f"{hexhash}.pkl"

    def _save_to_cache(self, smi, fp):
        path = self._cache_path(smi)
        path.parent.mkdir(parents=True, exist_ok=True)

        tmp = path.parent / f"{path.name}.{uuid.uuid4().hex}.tmp"
        with open(tmp, "wb") as f:
            pickle.dump(fp, f)
            f.flush()
            f.flush()

        tmp.replace(path)
        del fp, tmp

    def _load_from_cache(self, smi):
        path = self._cache_path(smi)
        if not path.exists():
            return None
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None

    def _get_mfpgen(
        self,
        fpSize: int = 2048,
    ):
        key = (self.value, fpSize)
        cache = _get_worker_cache()
        if key not in cache:
            if self is MolFingerprint.MORGAN:
                cache[key] = rdFingerprintGenerator.GetMorganGenerator(
                    radius=3, fpSize=fpSize
                )
            elif self is MolFingerprint.RDKIT:
                cache[key] = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=fpSize)
            elif self is MolFingerprint.TOPOTORSION:
                cache[key] = rdFingerprintGenerator.GetTopologicalTorsionGenerator(
                    fpSize=fpSize
                )
            elif self is MolFingerprint.ATOMPAIR:
                cache[key] = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=fpSize)
            else:
                raise ValueError(f"{self} does not support RDKit generators")
        return cache[key]

    @property
    def dim(self):
        if self == MolFingerprint.CHEMBERTA:
            return 384
        elif self == MolFingerprint.MACCS:
            return 167
        elif self == MolFingerprint.ALL:
            # sum of all except ChemBERTa
            return (
                MolFingerprint.MORGAN.dim
                + MolFingerprint.RDKIT.dim
                + MolFingerprint.TOPOTORSION.dim
                + MolFingerprint.ATOMPAIR.dim
                + MolFingerprint.MACCS.dim
            )
        else:
            return 2048

    @functools.cache
    def compute(self, smi: str, target: str = "numpy", use_cache: bool = True):
        """Compute or load fingerprint for a single SMILES."""
        if self is MolFingerprint.CHEMBERTA:
            return smiles_to_dl_embedding(
                [smi], model_name="DeepChem/ChemBERTa-77M-MLM", pooling="mean"
            )[0]

        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            logger.warning(f"No fp for SMILES={smi}")
            return None

        if use_cache:
            cached = self._load_from_cache(smi)
            if cached is not None:
                return cached

        if self is MolFingerprint.ALL:
            fps = []
            for fp_type in [
                MolFingerprint.MORGAN,
                MolFingerprint.RDKIT,
                MolFingerprint.TOPOTORSION,
                MolFingerprint.ATOMPAIR,
                MolFingerprint.MACCS,
            ]:
                fp = fp_type.compute(smi, target="numpy", use_cache=use_cache)
                if fp is not None:
                    fps.append(fp)
            result = np.concatenate(fps, axis=-1) if fps else None

        elif self is MolFingerprint.MACCS:
            fp = MACCSkeys.GenMACCSKeys(mol)
            arr = np.zeros((fp.GetNumBits(),), dtype=np.uint8)
            DataStructs.ConvertToNumpyArray(fp, arr)
            result = arr

        else:
            mfpgen = self._get_mfpgen()
            result = mfpgen.GetFingerprintAsNumPy(mol)

        if use_cache and result is not None:
            try:
                self._save_to_cache(smi, result)
            except Exception as e:
                logger.warning(f"Could not cache fingerprint for {smi}: {e}")

        return result

    def compute_parallel(self, smiles: Iterable[str], n_jobs: int = 16, **kwargs):
        if self is MolFingerprint.CHEMBERTA:
            logger.warning("Parallel compute not supported for ChemBERTa")
            return [self.compute(s, **kwargs) for s in tqdm.tqdm(smiles)]

        with Pool(n_jobs) as p:
            return p.map(functools.partial(self.compute, **kwargs), tqdm.tqdm(smiles))


@functools.cache
def _get_tokenizer_and_model(model_name: str) -> Tuple[object, object]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return tokenizer, model


def smiles_to_dl_embedding(
    smiles_list: Iterable[str],
    model_name: str = "DeepChem/ChemBERTa-77M-MLM",
    pooling: str = "mean",
):
    """Convert a list of SMILES strings into embeddings using ChemBERTa."""
    tokenizer, model = _get_tokenizer_and_model(model_name)
    encoded = tokenizer(smiles_list, padding=True, truncation=True, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**encoded)
        hidden_states = outputs.last_hidden_state  # (batch_size, seq_len, hidden_dim)

    if pooling == "mean":
        attention_mask = encoded["attention_mask"].unsqueeze(-1)
        summed = torch.sum(hidden_states * attention_mask, dim=1)
        counts = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
        embeddings = summed / counts
    elif pooling == "cls":
        embeddings = hidden_states[:, 0]
    else:
        raise ValueError("Pooling must be 'mean' or 'cls'")

    return embeddings.detach().numpy()


def esm2_features(
    data: pd.DataFrame,
    model_name: str = "esm2_t33_650M_UR50D",
    layer: int = 33,
    output_dir: Path | None = None,
):
    """Compute protein embeddings using the specified ESM model. Store results on disk.

    Args:
        data (pd.DataFrame): A dataframe with columns for SEQUENCE and TID.
        model_name (str), optional: Name of the ESM model to use.
        layer (int), optional: Layer index of the embedding to use.
        output_dir (Path|None), optional: Location to store embeddings.

    Returns:
        torch.Tensor or None: Tensor of protein embeddings or None if no protein targets.
    """
    if TID not in data.columns or data[TID].isna().any():
        logger.info("missing protein target in dataset")
        return None

    logger.info(f"computing protein features: {model_name}")
    done = []
    if output_dir is None:
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
        emb = torch.load(emb_dir(uniprot_id), weights_only=False, map_location=device)
        return emb["representation"][layer].cpu()

    emb = torch.stack([load_esm(uniprot_id) for uniprot_id in data[TID]])
    return emb.to(device)


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

    logger.info(f"setting up ESM model '{model_name}'")
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
    logger.info(f"ESM embeddings written to {output_dir}")
