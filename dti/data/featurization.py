import os
import logging
import hashlib
import functools
import pandas as pd
from pathlib import Path
from typing import Union, Iterable, List, Tuple
from multiprocessing import Pool
from enum import StrEnum, auto

import uuid
import torch.nn.functional as F
import torch
from torch_geometric.data import Data
import tqdm.auto as tqdm
import numpy as np
from esm import FastaBatchedDataset, pretrained
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator as fpg
from transformers import AutoTokenizer, AutoModel
from filelock import FileLock


from ..utils.constants import DATA, TID, SEQUENCE
from ..utils import device


logger = logging.getLogger(__name__)
logging.getLogger("filelock").setLevel(logging.WARNING)
_mfpgen_cache: dict[Tuple, object] = {}


# Lifted from pyg
x_map: dict[str, list] = {
    "atomic_num": list(map(str, range(0, 119))),
    "chirality": [
        "CHI_UNSPECIFIED",
        "CHI_TETRAHEDRAL_CW",
        "CHI_TETRAHEDRAL_CCW",
        "CHI_OTHER",
        "CHI_TETRAHEDRAL",
        "CHI_ALLFPENE",
        "CHI_SQUAREPLANAR",
        "CHI_TRIGONALBIPYRAMIDAL",
        "CHI_OCTAHEDRAL",
    ],
    "degree": list(map(str, range(0, 11))),
    "formal_charge": list(map(str, range(-5, 7))),
    "num_hs": list(map(str, range(0, 9))),
    "num_radical_electrons": list(map(str, range(0, 5))),
    "hybridization": [
        "UNSPECIFIED",
        "S",
        "SP",
        "SP2",
        "SP3",
        "SP3D",
        "SP3D2",
        "OTHER",
    ],
    "is_aromatic": [str(False), str(True)],
    "is_in_ring": [str(False), str(True)],
}

e_map: dict[str, list] = {
    "bond_type": [
        "UNSPECIFIED",
        "SINGLE",
        "DOUBLE",
        "TRIPLE",
        "QUADRUPLE",
        "QUINTUPLE",
        "HEXTUPLE",
        "ONEANDAHALF",
        "TWOANDAHALF",
        "THREEANDAHALF",
        "FOURANDAHALF",
        "FIVEANDAHALF",
        "AROMATIC",
        "IONIC",
        "HYDROGEN",
        "THREECENTER",
        "DATIVEONE",
        "DATIVE",
        "DATIVEL",
        "DATIVER",
        "OTHER",
        "ZERO",
    ],
    "stereo": [
        "STEREONONE",
        "STEREOANY",
        "STEREOZ",
        "STEREOE",
        "STEREOCIS",
        "STEREOTRANS",
    ],
    "is_conjugated": [str(False), str(True)],
}

NODE_FEATURE_DIM = sum(len(v) for v in x_map.values())
EDGE_FEATURE_DIM = sum(len(v) for v in e_map.values())
FP_DEFAULT_DIM = 2048
FP_SMALL_DIM = 1024


def from_rdmol_one_hot(mol) -> "torch_geometric.data.Data":
    """
    Converts an RDKit Mol instance to a PyG Data instance with
    one-hot encoded features.
    """
    assert isinstance(mol, Chem.Mol)

    xs: List[torch.Tensor] = []
    for atom in mol.GetAtoms():
        atom_feats = []
        for feat_key, accessor in [
            ("atomic_num", "GetAtomicNum"),
            ("chirality", "GetChiralTag"),
            ("degree", "GetTotalDegree"),
            ("formal_charge", "GetFormalCharge"),
            ("num_hs", "GetTotalNumHs"),
            ("num_radical_electrons", "GetNumRadicalElectrons"),
            ("hybridization", "GetHybridization"),
            ("is_aromatic", "GetIsAromatic"),
            ("is_in_ring", "IsInRing"),
        ]:
            atom_feats.append(
                F.one_hot(
                    torch.tensor(x_map[feat_key].index(str(getattr(atom, accessor)()))),
                    len(x_map[feat_key]),
                )
            )

        xs.append(torch.cat(atom_feats, dim=0).float())

    x = torch.stack(xs, dim=0)

    edge_indices, edge_attrs_list = [], []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        bond_feats = []
        for feat_key, accessor in [
            ("bond_type", "GetBondType"),
            ("stereo", "GetStereo"),
            ("is_conjugated", "GetIsConjugated"),
        ]:
            bond_feats.append(
                F.one_hot(
                    torch.tensor(e_map[feat_key].index(str(getattr(bond, accessor)()))),
                    len(e_map[feat_key]),
                )
            )

        bond_feature_vector = torch.cat(bond_feats, dim=0).float()

        edge_indices += [[i, j], [j, i]]
        edge_attrs_list += [bond_feature_vector, bond_feature_vector]

    edge_index = torch.tensor(edge_indices)
    edge_index = edge_index.t().to(torch.long).view(2, -1)

    if edge_attrs_list:
        edge_attr = torch.stack(edge_attrs_list, dim=0)
    else:
        edge_attr = torch.empty((0, EDGE_FEATURE_DIM), dtype=torch.float)

    if edge_index.numel() > 0:  # Sort indices
        perm = (edge_index[0] * x.size(0) + edge_index[1]).argsort()
        edge_index, edge_attr = edge_index[:, perm], edge_attr[perm]

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def from_smiles_one_hot(smi: str, **kwargs) -> Data:
    r"""Converts a SMILES string to a :class:`torch_geometric.data.Data`
    instance with one-hot encoded features.
    """
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        raise ValueError(f"Could not parse SMILES string: {smi}")
    return from_rdmol_one_hot(mol)


class MolFingerprint(StrEnum):
    MORGAN = auto()
    RDKIT = auto()
    TOPOTORSION = auto()
    ATOMPAIR = auto()
    CHEMBERTA = auto()
    GRAPH = auto()
    ALLFP = auto()
    ALL = auto()

    def _get_mfpgen(
        self,
        fp_size: int = FP_DEFAULT_DIM,
    ):
        key = (self.value, fp_size)
        if key not in _mfpgen_cache:
            if self is MolFingerprint.MORGAN:
                _mfpgen_cache[key] = fpg.GetMorganGenerator(radius=3, fpSize=fp_size)
            elif self is MolFingerprint.RDKIT:
                _mfpgen_cache[key] = fpg.GetRDKitFPGenerator(fpSize=fp_size)
            elif self is MolFingerprint.TOPOTORSION:
                _mfpgen_cache[key] = fpg.GetTopologicalTorsionGenerator(fpSize=fp_size)
            elif self is MolFingerprint.ATOMPAIR:
                _mfpgen_cache[key] = fpg.GetAtomPairGenerator(fpSize=fp_size)
            else:
                raise ValueError(f"{self} does not support RDKit generators")
        return _mfpgen_cache[key]

    @property
    def dim(self):
        match self:
            case MolFingerprint.CHEMBERTA:
                return 384
            case MolFingerprint.GRAPH:
                return NODE_FEATURE_DIM
            case MolFingerprint.ALLFP | MolFingerprint.ALL:
                return sum(FP_SMALL_DIM for m in self.members_all)
            case _:
                return FP_DEFAULT_DIM

    @property
    def members_all(self):
        return [
            m
            for m in MolFingerprint
            if m
            not in (
                MolFingerprint.ALLFP,
                MolFingerprint.GRAPH,
                MolFingerprint.ALL,
                MolFingerprint.CHEMBERTA,
            )
        ]

    @property
    def graph_based(self):
        return self in (MolFingerprint.ALL, MolFingerprint.GRAPH)

    @functools.lru_cache(maxsize=100_000)
    def compute(
        self,
        smi: str,
        fp_size: int = FP_DEFAULT_DIM,
        target: str = "numpy",
        cache_dir: Path = None,
    ):
        if self == MolFingerprint.ALL:
            raise ValueError("combined manually")

        if self is MolFingerprint.ALLFP:
            all_fps = []
            for feat in self.members_all:
                if feat is MolFingerprint.CHEMBERTA:
                    fp = feat.compute(smi)
                else:
                    fp = feat.compute(smi, fp_size=FP_SMALL_DIM, target="numpy")

                if fp is None:
                    logger.warning(
                        f"Failed to compute {feat.value} for ALLFP on SMILES={smi}"
                    )
                    return None
                all_fps.append(fp)
            return np.concatenate(all_fps)

        if self is MolFingerprint.CHEMBERTA:
            fp_compute = lambda s: smiles_to_dl_embedding(
                [s], model_name="DeepChem/ChemBERTa-77M-MLM", pooling="mean"
            )[0]
            return self._cache_lookup(cache_dir / str(self), smi, fp_compute)

        if self is MolFingerprint.GRAPH:
            return self._cache_lookup(cache_dir / str(self), smi, from_smiles_one_hot)

        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            logger.warning(f"No fp for SMILES={smi}")
            return None

        mfpgen = self._get_mfpgen(fp_size)

        match target:
            case "numpy":
                return mfpgen.GetFingerprintAsNumPy(mol)
            case "native":
                return mfpgen.GetFingerprint(mol)
            case _:
                raise ValueError(f"Unknown fingerprint target: '{target}'")

    def _cache_lookup(self, cache_dir: Path, smi: str, fp_compute) -> Path:
        try:
            if cache_dir is None:
                return fp_compute(smi)
            hash_str = hashlib.sha256(smi.encode()).hexdigest()

            cache_subdir = cache_dir / hash_str[0:2] / hash_str[2:4]
            cache_path = cache_subdir / f"{hash_str}.pt"
            lock_path = cache_subdir / f"{hash_str}.lock"
            temp_path = cache_subdir / f"{hash_str}{uuid.uuid1()}.tmp"

            cache_subdir.mkdir(parents=True, exist_ok=True)

            lock = FileLock(lock_path, timeout=10)
            try:
                with lock:
                    try:
                        graph = torch.load(cache_path, weights_only=False)
                    except FileNotFoundError as e:
                        graph = fp_compute(smi)
                        if graph is None:
                            logger.warning(f"Computation failed for SMILES={smi}: {e}")
                            return None
                        torch.save(graph, temp_path)
                        os.rename(temp_path, cache_path)
            except Exception as e:
                logger.warning(
                    f"Failed to load cached file {cache_path}, re-computing. Error: {e}"
                )
                graph = fp_compute(smi)
                if graph is None:
                    raise ValueError(f"Computation failed for SMILES: {smi}")

                torch.save(graph, temp_path)
                os.rename(temp_path, cache_path)
            return graph
        except Exception as e:
            logger.warning(f"Computation failed for SMILES={smi}: {e}")
            return None

    def compute_parallel(
        self,
        smiles: Iterable[str],
        n_jobs: int = 16,
        _fp_size: int = FP_DEFAULT_DIM,
        **kwargs,
    ):
        if self == MolFingerprint.ALL:
            raise ValueError("combined manually")
        elif self in {
            MolFingerprint.GRAPH,
            MolFingerprint.MORGAN,
            MolFingerprint.RDKIT,
            MolFingerprint.TOPOTORSION,
            MolFingerprint.ATOMPAIR,
        }:
            logger.info(
                f"Parallel featurization for {self.value} using {n_jobs} cores."
            )
            with Pool(n_jobs) as p:
                return p.map(
                    functools.partial(self.compute, fp_size=_fp_size, **kwargs),
                    tqdm.tqdm(smiles, desc=f"{self.value}({_fp_size})"),
                )
        elif self == MolFingerprint.ALLFP:
            embeddings = [
                np.array(
                    m.compute_parallel(smiles, fp_size=FP_SMALL_DIM, n_jobs=n_jobs)
                )
                for m in self.members_all
            ]
            return np.concatenate(embeddings, axis=1)
        elif self == MolFingerprint.CHEMBERTA:
            _batch_size = 4096
            embeddings = list()
            for batch in tqdm.tqdm(
                range(0, len(smiles), _batch_size), desc=f"{self.value}"
            ):
                smi_batch = list(smiles[batch : min(len(smiles), batch + _batch_size)])
                batch_embds = smiles_to_dl_embedding(
                    smi_batch,
                    model_name="DeepChem/ChemBERTa-77M-MLM",
                    pooling="mean",
                )
                embeddings.extend(list(batch_embds))

            assert len(smiles) == len(embeddings), (len(smiles), len(embeddings))
            return embeddings
        else:
            logger.warning(
                f"No parallel logic defined for {self.value}. Falling back to sequential."
            )
            return [self.compute(s, **kwargs) for s in tqdm.tqdm(smiles)]


@functools.cache
def _get_tokenizer_and_model(model_name: str) -> Tuple[object, object]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return tokenizer, model.to(device)


def smiles_to_dl_embedding(
    smiles_list: list[str],
    model_name: str = "DeepChem/ChemBERTa-77M-MLM",
    pooling: str = "mean",
):
    tokenizer, model = _get_tokenizer_and_model(model_name)
    encoded = tokenizer(smiles_list, padding=True, truncation=True, return_tensors="pt")
    encoded = encoded.to(device)

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

    return embeddings.detach().cpu().numpy()


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
