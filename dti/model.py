import torch
import torch.nn as nn
import torch.nn.functional as F

import logging

logger = logging.getLogger(__name__)

class IndependentFeatureMapping(nn.Module):
    def __init__(self, input_dim, k=10, sigma=1.0):
        super(IndependentFeatureMapping, self).__init__()
        self.k = k
        self.sigma = sigma
        self.c = nn.Parameter(torch.randn(k) * sigma)

    def forward(self, x):
        v = 2 * torch.pi * self.c.unsqueeze(0) * x.unsqueeze(2)

        sin_v = torch.sin(v)
        cos_v = torch.cos(v)

        fx = torch.cat([sin_v, cos_v], dim=2)  # (batch_size, input_dim, 2k)

        return fx.view(x.shape[0], -1)


class MolecularModel(nn.Module):
    def __init__(self, ligand_input_size, embedding_size, ifm_k=10, ifm_sigma=1.0, **kwargs):
        super().__init__()

        self.molecule_input_size = ligand_input_size
        self.ifm = IndependentFeatureMapping(ligand_input_size, k=ifm_k, sigma=ifm_sigma)

        self.stack_input_size = ligand_input_size * 2 * ifm_k

        self.stack = nn.Sequential(
            nn.Linear(self.stack_input_size, embedding_size),
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

        if molecule.shape[1] == self.stack_input_size:
            molecule = self.stack(molecule)
            return self.readout(molecule)
        elif molecule.shape[1] == self.stack_input_size * 2:
            embedding_a = self.stack(molecule[:, :self.stack_input_size])
            embedding_b = self.stack(molecule[:, self.stack_input_size:])
            # ensure equivariance wrt. to tuple permutation
            delta_ab = self.readout(embedding_a - embedding_b)
            delta_ba = self.readout(embedding_b - embedding_a)
            return delta_ab - delta_ba
        else:
            raise ValueError(f"Unexpected molecule shape {molecule.shape}")

class CombinedModel(nn.Module):
    def __init__(
        self,
        protein_input_size,
        ligand_input_size,
        embedding_size,
        hidden_layer_size=512,
        cosine_agg=False,
        ifm_k=10,
        ifm_sigma=0.1,
    ):
        super().__init__()

        self.ligand_input_size = ligand_input_size
        self.cosine_agg = cosine_agg

        self.ligand_ifm = IndependentFeatureMapping(ligand_input_size, k=ifm_k, sigma=ifm_sigma)

        self.ligand_mlp_input_size = ligand_input_size * 2 * ifm_k

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
            nn.Linear(self.ligand_mlp_input_size, hidden_layer_size),
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

        ligand = self.ligand_ifm(ligand)

        if ligand.shape[1] == self.ligand_mlp_input_size * 2:
            ligand_a = ligand[:, :self.ligand_mlp_input_size]
            ligand_b = ligand[:, self.ligand_mlp_input_size:]
            ligand_emb_a = self.ligand_mlp(ligand_a)
            ligand_emb_b = self.ligand_mlp(ligand_b)

            if self.cosine_agg:
                combined_emb_a = protein_emb * ligand_emb_a
                combined_emb_b = protein_emb * ligand_emb_b
            else:
                combined_emb_a = torch.cat([protein_emb, ligand_emb_a], dim=1)
                combined_emb_b = torch.cat([protein_emb, ligand_emb_b], dim=1)
            pred_ab = self.combined_mlp(combined_emb_a - combined_emb_b)
            pred_ba = self.combined_mlp(combined_emb_b - combined_emb_a)
            return pred_ab - pred_ba
        elif ligand.shape[1] == self.ligand_mlp_input_size:
            ligand_emb = self.ligand_mlp(ligand)

            if self.cosine_agg:
                combined_emb = protein_emb * ligand_emb
            else:
                combined_emb = torch.cat([protein_emb, ligand_emb], dim=1)
            output = self.combined_mlp(combined_emb)
            return output
        else:
            raise ValueError(f"Unexpected ligand shape {ligand.shape}")
