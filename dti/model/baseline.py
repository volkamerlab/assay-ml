import torch
from torch import Tensor
from torch.nn import (
    Module,
)
from torch import nn
from torch.nn import GELU

from .common import mlp

import logging

logger = logging.getLogger(__name__)


class MolecularModel(Module):
    def __init__(
        self, ligand_input_size, embedding_size, act=GELU, dropout=0.05, **kwargs
    ):
        super().__init__()

        self.molecule_input_size = ligand_input_size
        self.embedding_size = embedding_size
        self.embed = mlp(
            input_size=self.molecule_input_size,
            hidden_size=embedding_size,
            output_size=embedding_size,
            hidden_layers=4,
            act=act,
        )
        scaled_hidden_dim = embedding_size // 2
        self.readout = nn.Sequential(
            nn.Dropout(dropout),
            nn.BatchNorm1d(embedding_size),
            nn.Linear(embedding_size, embedding_size),
            act(),
            nn.Linear(embedding_size, scaled_hidden_dim),
            act(),
            nn.Linear(scaled_hidden_dim, 1),
        )

    def forward(self, _protein, molecule, **kwargs):
        # the  first argument (protein embeddings) is ignored
        if molecule.dim() == 3:
            molecule = molecule.squeeze()
        assert molecule.shape[1] == self.molecule_input_size, molecule.shape
        molecule = self.embed(molecule)
        return self.readout(molecule)


class PairMolecularModel(MolecularModel):
    def __init__(self, ligand_input_size, embedding_size, **kwargs):
        super().__init__(ligand_input_size, embedding_size, **kwargs)

    def forward(self, _protein, molecule: Tensor, **kwargs):
        x = self.embed(molecule)
        if x.dim() == 2:  # full (b, b) pairs
            n, d = molecule.size(0), self.embedding_size
            diff = (x.view(n, 1, d) - x.view(1, n, d)).reshape(n**2, d)  # (b^2, d)
        elif x.dim() == 3:  # assume pre-defined pairs (k, 2, d)
            diff = x[:, 0, :] - x[:, 1, :]  # (k, d)
        else:
            assert False
        return self.readout(diff) - self.readout(-diff)


class CombinedModel(nn.Module):
    def __init__(
        self,
        protein_input_size,
        ligand_input_size,
        embedding_size,
        hidden_layer_size=512,
        cosine_agg=True,
        act=GELU,
        **kwargs,
    ):
        super().__init__()

        self.ligand_input_size = ligand_input_size
        self.cosine_agg = cosine_agg
        self.embedding_size = embedding_size
        self.proteinmlp = mlp(
            input_size=protein_input_size,
            hidden_size=hidden_layer_size,
            output_size=embedding_size,
            hidden_layers=4,
            act=act,
        )
        self.ligandmlp = mlp(
            input_size=ligand_input_size,
            hidden_size=hidden_layer_size,
            output_size=embedding_size,
            hidden_layers=4,
            act=act,
        )

        scaled_hidden_dim = hidden_layer_size // 2
        self.combinedmlp = nn.Sequential(
            nn.Dropout(0.05),
            nn.BatchNorm1d(embedding_size),
            nn.Linear(embedding_size, hidden_layer_size),
            act(),
            nn.Linear(hidden_layer_size, scaled_hidden_dim),
            act(),
            nn.Linear(scaled_hidden_dim, 1),
        )

    def forward(self, protein, ligand, **kwargs):
        if ligand.dim() == 3:
            ligand = ligand.squeeze()
            protein = protein.squeeze()

        assert ligand.shape[1] == self.ligand_input_size, ligand.shape

        protein_emb = self.proteinmlp(protein)
        ligand_emb = self.ligandmlp(ligand)

        if self.cosine_agg:
            combined_emb = protein_emb * ligand_emb
        else:
            combined_emb = torch.cat([protein_emb, ligand_emb], dim=1)
        output = self.combinedmlp(combined_emb)
        return output


class PairCombinedModel(CombinedModel):
    def __init__(
        self,
        protein_input_size,
        ligand_input_size,
        embedding_size,
        hidden_layer_size=512,
        cosine_agg=True,
        act=GELU,
        **kwargs,
    ):
        super().__init__(
            protein_input_size,
            ligand_input_size,
            embedding_size,
            hidden_layer_size=512,
            cosine_agg=False,
            act=act,
        )

    def forward(self, protein, ligand, **kwargs):
        protein_emb = self.proteinmlp(protein)
        x = self.ligandmlp(ligand)
        if x.dim() == 2:  # all pairs
            x = protein_emb * x
            n, d = ligand.size(0), self.embedding_size
            diff = (x.view(n, 1, d) - x.view(1, n, d)).reshape(n**2, d)
        elif x.dim() == 3:  # pre-defined pairs (k, 2, d)
            x = x * protein_emb.unsqueeze(1)
            diff = x[:, 0, :] - x[:, 1, :]
        else:
            assert False

        return self.combinedmlp(diff) - self.combinedmlp(-diff)
