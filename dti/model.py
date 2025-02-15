import torch
from torch import nn

import logging

logger = logging.getLogger(__name__)


class MolecularModel(nn.Module):
    def __init__(self, ligand_input_size, embedding_size, **kwargs):
        super().__init__()

        self.molecule_input_size = ligand_input_size
        self.stack = nn.Sequential(
            nn.Linear(self.molecule_input_size, embedding_size),
            nn.SiLU(),
            nn.Linear(embedding_size, embedding_size),
            nn.SiLU(),
            nn.Linear(embedding_size, embedding_size),
            nn.SiLU(),
            nn.Linear(embedding_size, embedding_size),
            nn.SiLU(),
            nn.Linear(embedding_size, embedding_size),
        )
        scaled_hidden_dim = embedding_size // 2
        self.readout = nn.Sequential(
            nn.Dropout(0.05),
            nn.BatchNorm1d(embedding_size),
            nn.Linear(embedding_size, embedding_size),
            nn.SiLU(),
            nn.Linear(embedding_size, scaled_hidden_dim),
            nn.SiLU(),
            nn.Linear(scaled_hidden_dim, 1),
        )

    def forward(self, _protein, molecule):
        # the  first argument (protein embeddings) is ignored
        if molecule.shape[1] == self.molecule_input_size:
            molecule = self.stack(molecule)
            return self.readout(molecule)
        elif molecule.shape[1] == self.molecule_input_size * 2:
            embedding_a = self.stack(molecule[:, : self.molecule_input_size])
            embedding_b = self.stack(molecule[:, self.molecule_input_size :])
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
    ):
        super().__init__()

        self.ligand_input_size = ligand_input_size
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
            nn.Linear(hidden_layer_size, embedding_size),
        )

        self.ligand_mlp = nn.Sequential(
            nn.Linear(self.ligand_input_size, hidden_layer_size),
            nn.SiLU(),
            nn.Linear(hidden_layer_size, hidden_layer_size),
            nn.SiLU(),
            nn.Linear(hidden_layer_size, hidden_layer_size),
            nn.SiLU(),
            nn.Linear(hidden_layer_size, hidden_layer_size),
            nn.SiLU(),
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

        if ligand.shape[1] == self.ligand_input_size * 2:
            ligand_a = ligand[:, : self.ligand_input_size]
            ligand_b = ligand[:, self.ligand_input_size :]
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
        elif ligand.shape[1] == self.ligand_input_size:
            ligand_emb = self.ligand_mlp(ligand)

            if self.cosine_agg:
                combined_emb = protein_emb * ligand_emb
            else:
                combined_emb = torch.cat([protein_emb, ligand_emb], dim=1)
            output = self.combined_mlp(combined_emb)
            return output
        else:
            raise ValueError(f"Unexpected ligand shape {ligand.shape}")
