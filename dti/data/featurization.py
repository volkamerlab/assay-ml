import logging
import hashlib
import functools
import pandas as pd
from pathlib import Path
from typing import Union, Iterable, List, Tuple
from multiprocessing import Pool
from enum import StrEnum, auto

import torch.nn.functional as F
import torch
from torch_geometric.data import Data
from torch_geometric.utils.smiles import from_smiles
import tqdm.auto as tqdm
import numpy as np
from esm import FastaBatchedDataset, pretrained
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from transformers import AutoTokenizer, AutoModel
from filelock import FileLock


from ..utils.constants import DATA, TID, SEQUENCE
from ..utils import device


logger = logging.getLogger(__name__)
logging.getLogger("filelock").setLevel(logging.WARNING)
_mfpgen_cache: dict[Tuple, object] = {}


# Lifted from pyg
x_map: dict[str, list] = {
    "atomic_num": list(range(0, 119)),
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
    "degree": list(range(0, 11)),
    "formal_charge": list(range(-5, 7)),
    "num_hs": list(range(0, 9)),
    "num_radical_electrons": list(range(0, 5)),
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
    "is_aromatic": [False, True],
    "is_in_ring": [False, True],
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
    "is_conjugated": [False, True],
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
        atom_feats.append(
            F.one_hot(
                torch.tensor(x_map["atomic_num"].index(atom.GetAtomicNum())),
                len(x_map["atomic_num"]),
            )
        )
        atom_feats.append(
            F.one_hot(
                torch.tensor(x_map["chirality"].index(str(atom.GetChiralTag()))),
                len(x_map["chirality"]),
            )
        )
        atom_feats.append(
            F.one_hot(
                torch.tensor(x_map["degree"].index(atom.GetTotalDegree())),
                len(x_map["degree"]),
            )
        )
        atom_feats.append(
            F.one_hot(
                torch.tensor(x_map["formal_charge"].index(atom.GetFormalCharge())),
                len(x_map["formal_charge"]),
            )
        )
        atom_feats.append(
            F.one_hot(
                torch.tensor(x_map["num_hs"].index(atom.GetTotalNumHs())),
                len(x_map["num_hs"]),
            )
        )
        atom_feats.append(
            F.one_hot(
                torch.tensor(
                    x_map["num_radical_electrons"].index(atom.GetNumRadicalElectrons())
                ),
                len(x_map["num_radical_electrons"]),
            )
        )
        atom_feats.append(
            F.one_hot(
                torch.tensor(
                    x_map["hybridization"].index(str(atom.GetHybridization()))
                ),
                len(x_map["hybridization"]),
            )
        )
        atom_feats.append(
            F.one_hot(
                torch.tensor(x_map["is_aromatic"].index(atom.GetIsAromatic())),
                len(x_map["is_aromatic"]),
            )
        )
        atom_feats.append(
            F.one_hot(
                torch.tensor(x_map["is_in_ring"].index(atom.IsInRing())),
                len(x_map["is_in_ring"]),
            )
        )

        xs.append(torch.cat(atom_feats, dim=0).float())

    x = torch.stack(xs, dim=0)

    edge_indices, edge_attrs_list = [], []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        bond_feats = []
        bond_feats.append(
            F.one_hot(
                torch.tensor(e_map["bond_type"].index(str(bond.GetBondType()))),
                len(e_map["bond_type"]),
            )
        )
        bond_feats.append(
            F.one_hot(
                torch.tensor(e_map["stereo"].index(str(bond.GetStereo()))),
                len(e_map["stereo"]),
            )
        )
        bond_feats.append(
            F.one_hot(
                torch.tensor(e_map["is_conjugated"].index(bond.GetIsConjugated())),
                len(e_map["is_conjugated"]),
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
        fpSize: int = FP_DEFAULT_DIM,
    ):
        key = (self.value, fpSize)
        if key not in _mfpgen_cache:
            if self is MolFingerprint.MORGAN:
                _mfpgen_cache[key] = rdFingerprintGenerator.GetMorganGenerator(
                    radius=3, fpSize=fpSize
                )
            elif self is MolFingerprint.RDKIT:
                _mfpgen_cache[key] = rdFingerprintGenerator.GetRDKitFPGenerator(
                    fpSize=fpSize
                )
            elif self is MolFingerprint.TOPOTORSION:
                _mfpgen_cache[key] = (
                    rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=fpSize)
                )
            elif self is MolFingerprint.ATOMPAIR:
                _mfpgen_cache[key] = rdFingerprintGenerator.GetAtomPairGenerator(
                    fpSize=fpSize
                )
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

    @functools.lru_cache(maxsize=1_000_000)
    def compute(
        self,
        smi: str,
        fpSize: int = FP_DEFAULT_DIM,
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
                    fp = feat.compute(smi, fpSize=FP_SMALL_DIM, target="numpy")

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

        mfpgen = self._get_mfpgen(fpSize)

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
                        torch.save(graph, cache_path)
            except:
                logger.error(f"Failure for SMILES: {smi} and file {lock_path}")
                graph = fp_compute(smi)
            return graph
        except Exception as e:
            logger.warning(f"Computation failed for SMILES={smi}: {e}")
            return None

    def compute_parallel(self, smiles: Iterable[str], n_jobs: int = 16, **kwargs):
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
                    functools.partial(self.compute, fpSize=FP_SMALL_DIM, **kwargs),
                    tqdm.tqdm(smiles, desc=f"featurizing {self.value}"),
                )
        elif self == MolFingerprint.ALLFP:
            embeddings = [
                np.array(m.compute_parallel(smiles)) for m in self.members_all
            ]
            return np.concatenate(embeddings, axis=1)
        elif self == MolFingerprint.CHEMBERTA:
            _batch_size = 4096
            embeddings = list()
            for batch in tqdm.tqdm(
                range(0, len(smiles), _batch_size), desc=f"featurizing {self.value}"
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
