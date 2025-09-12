import logging
import functools
import pandas as pd
from pathlib import Path
from typing import Union, Iterable, List
from multiprocessing import Pool

import torch
from esm import FastaBatchedDataset, pretrained
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

from .constants import DATA, SMILES, ACT, TID, SEQUENCE, ASSAY, COMPOUND
from .utils import device


logger = logging.getLogger(__name__)


@functools.cache
def compute_fp(smi: str, radius: int = 3, fp_dim: int = 2048, target: str = "numpy"):
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=fp_dim)
    try:
        match target:
            case "numpy":
                return mfpgen.GetFingerprintAsNumPy(Chem.MolFromSmiles(smi))
            case "native":
                return mfpgen.GetFingerprint(Chem.MolFromSmiles(smi))
            case _:
                raise ValueError(f"unkown fingerprint target: '{target}'")
    except TypeError:
        logger.warn(f"No fp for SMILES={smi}")
        return None


def par_compute_fp(smiles: Iterable[str], n_jobs=16, target: str = "numpy"):
    with Pool(n_jobs) as p:
        return p.map(functools.partial(compute_fp, target=target), smiles)


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
    logger.info("ESM embeddings written to {output_dir}")
