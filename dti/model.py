import torch
from torch import nn

import logging

logger = logging.getLogger(__name__)


class CombinedModel(nn.Module):
    def __init__(
        self, protein_input_size, ligand_input_size, embedding_size, cosine_agg=False
    ):
        super(CombinedModel, self).__init__()

        self.cosine_agg = cosine_agg

        self.protein_mlp = nn.Sequential(
            nn.Linear(protein_input_size, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(512, embedding_size),
        )

        self.ligand_mlp = nn.Sequential(
            nn.Linear(ligand_input_size, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(512, embedding_size),
        )

        joint_embedding_size = embedding_size * (1 if cosine_agg else 2)
        self.combined_mlp = nn.Sequential(
            nn.Dropout(0.05),
            nn.BatchNorm1d(joint_embedding_size),
            nn.Linear(joint_embedding_size, 512),
            nn.SiLU(),
            nn.Linear(512, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
        )

    def forward(self, protein, ligand):
        protein_embedding = self.protein_mlp(protein)

        ligand_embedding = self.ligand_mlp(ligand)

        if self.cosine_agg:
            combined_embedding = protein_embedding * ligand_embedding
        else:
            combined_embedding = torch.cat([protein_embedding, ligand_embedding], dim=1)
        output = self.combined_mlp(combined_embedding)
        return output


class PairModel(nn.Module):
    def __init__(self, protein_input_size, ligand_input_size, embedding_size, **kwargs):
        super().__init__()

        self.ligand_input_size = ligand_input_size

        self.protein_mlp = nn.Sequential(
            nn.Linear(protein_input_size, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(512, embedding_size),
        )

        self.ligand_mlp = nn.Sequential(
            nn.Linear(ligand_input_size, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(512, embedding_size),
        )

        self.combined_mlp = nn.Sequential(
            nn.Dropout(0.05),
            nn.BatchNorm1d(embedding_size),
            nn.Linear(embedding_size, 512),
            nn.SiLU(),
            nn.Linear(512, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
        )

    def forward(self, protein, ligands):
        protein_embedding = self.protein_mlp(protein)

        ligand_a = ligands[:, : self.ligand_input_size]
        ligand_b = ligands[:, self.ligand_input_size :]
        ligand_embedding_a = self.ligand_mlp(ligand_a)
        ligand_embedding_b = self.ligand_mlp(ligand_b)

        combined_embedding_a = protein_embedding * ligand_embedding_a
        combined_embedding_b = protein_embedding * ligand_embedding_b

        combined_embedding = combined_embedding_a - combined_embedding_b
        output = self.combined_mlp(combined_embedding)

        return output
