import torch
from torch import nn

import logging

logger = logging.getLogger(__name__)


class IndependentFeatureMapping(nn.Module):
    def __init__(self, input_dim, k, sigma=1.0):
        super().__init__()
        self.k = k
        self.input_dim = input_dim
        self.c = nn.Parameter(torch.randn(input_dim, k) * sigma)

    def forward(self, x):
        v = 2 * torch.pi * x.unsqueeze(-1) * self.c
        sin_v = torch.sin(v)
        cos_v = torch.cos(v)
        return torch.cat([sin_v, cos_v], dim=-1).flatten()


class MolecularModel(nn.Module):
    def __init__(self, ligand_input_size, embedding_size, k=5, sigma=1.0):
        super().__init__()

        self.ifm = IndependentFeatureMapping(ligand_input_size, k, sigma)
        transformed_input_size = ligand_input_size * 2 * k

        self.stack = nn.Sequential(
            nn.Linear(transformed_input_size, embedding_size),
            nn.SiLU(),
            nn.Linear(embedding_size, embedding_size),
            nn.SiLU(),
            nn.Linear(embedding_size, embedding_size),
            nn.SiLU(),
            nn.Linear(embedding_size, embedding_size),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(embedding_size, embedding_size),
        )

        scaled_hidden_dim = embedding_size // 2
        self.readout = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embedding_size, embedding_size),
            nn.SiLU(),
            nn.Linear(embedding_size, scaled_hidden_dim),
            nn.SiLU(),
            nn.Linear(scaled_hidden_dim, 1),
        )

    def forward(self, _protein, molecule):
        molecule = self.ifm(molecule)
        molecule = self.stack(molecule)
        return self.readout(molecule)


class CombinedModel(nn.Module):
    def __init__(
        self,
        protein_input_size,
        ligand_input_size,
        embedding_size,
        hidden_layer_size=512,
        k=5,
        sigma=1.0,
        cosine_agg=False,
    ):
        super().__init__()

        self.ifm = IndependentFeatureMapping(ligand_input_size, k, sigma)
        transformed_input_size = ligand_input_size * 2 * k
        self.cosine_agg = cosine_agg

        self.protein_mlp = nn.Sequential(
            nn.Linear(protein_input_size, hidden_layer_size),
            nn.SiLU(),
            nn.Linear(hidden_layer_size, hidden_layer_size),
            nn.SiLU(),
            nn.Linear(hidden_layer_size, hidden_layer_size),
            nn.SiLU(),
            nn.Linear(hidden_layer_size, hidden_layer_size),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(hidden_layer_size, embedding_size),
        )

        self.ligand_mlp = nn.Sequential(
            nn.Linear(transformed_input_size, hidden_layer_size),
            nn.SiLU(),
            nn.Linear(hidden_layer_size, hidden_layer_size),
            nn.SiLU(),
            nn.Linear(hidden_layer_size, hidden_layer_size),
            nn.SiLU(),
            nn.Linear(hidden_layer_size, hidden_layer_size),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(hidden_layer_size, embedding_size),
        )

        joint_embedding_size = embedding_size * (1 if cosine_agg else 2)
        scaled_hidden_dim = hidden_layer_size // 2
        self.combined_mlp = nn.Sequential(
            nn.Dropout(0.05),
            nn.BatchNorm1d(joint_embedding_size),
            nn.Linear(joint_embedding_size, hidden_layer_size),
            nn.SiLU(),
            nn.Linear(hidden_layer_size, scaled_hidden_dim),
            nn.SiLU(),
            nn.Linear(scaled_hidden_dim, 1),
        )

    def forward(self, protein, ligand):
        protein_emb = self.protein_mlp(protein)
        ligand = self.ifm(ligand)
        ligand_emb = self.ligand_mlp(ligand)

        if self.cosine_agg:
            combined_emb = protein_emb * ligand_emb
        else:
            combined_emb = torch.cat([protein_emb, ligand_emb], dim=1)

        return self.combined_mlp(combined_emb)
