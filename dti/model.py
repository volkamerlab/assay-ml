import torch
from torch import nn

import logging

logger = logging.getLogger(__name__)


class CombinedModel(nn.Module):
    def __init__(self, protein_input_size, ligand_input_size, embedding_size, cosine_agg=False):
        super(CombinedModel, self).__init__()

        self.cosing_agg = cosine_agg

        # Protein sequence transformer
        self.protein_mlp = nn.Sequential(
            nn.Linear(protein_input_size, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, embedding_size),
        )

        # Ligand MLP
        self.ligand_mlp = nn.Sequential(
            nn.Linear(ligand_input_size, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, embedding_size),
        )

        # Combined MLP
        self.combined_mlp = nn.Sequential(
            nn.Dropout(0.1),
            nn.BatchNorm1d(embedding_size * (1 if cosine_agg else 2)),
            nn.Linear(embedding_size, 512),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 128),
            nn.SiLU(),
            nn.Dropout(0.1),
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
