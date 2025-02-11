import torch
from torch import nn

import logging

logger = logging.getLogger(__name__)


class MolecularModel(nn.Module):
    def __init__(self, molecule_input_size, embedding_size):
        self.molecule_input_size = molecule_input_size
        self.stack = nn.Sequential(
            nn.Linear(molecule_input_size, embedding_size),
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
        self.readout = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embedding_size, embedding_size),
            nn.SiLU(),
            nn.Linear(embedding_size, 1),
        )

    def forward(self, x):
        x = self.stack(x)
        return self.readout(x)


class MolecularPairModel(MolecularModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, x):
        molecule_a = x[:, : self.molecule_input_size]
        molecule_b = x[:, self.molecule_input_size :]
        embedding_a = self.stack(molecule_a)
        embedding_b = self.stack(molecule_b)
        return self.readout(embedding_a - embedding_b)


class CombinedModel(nn.Module):
    def __init__(
        self, protein_input_size, ligand_input_size, embedding_size, cosine_agg=False
    ):
        super(CombinedModel, self).__init__()

        self.ligand_input_size = ligand_input_size
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
            nn.Linear(self.ligand_input_size, 512),
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

        if ligand.shape[1] == self.ligand_input_size * 2:
            ligand_a = ligand[:, : self.ligand_input_size]
            ligand_b = ligand[:, self.ligand_input_size :]
            ligand_embedding_a = self.ligand_mlp(ligand_a)
            ligand_embedding_b = self.ligand_mlp(ligand_b)

            if self.cosine_agg:
                combined_embedding_a = protein_embedding * ligand_embedding_a
                combined_embedding_b = protein_embedding * ligand_embedding_b
            else:
                combined_embedding_a = torch.cat([protein_embedding, ligand_embedding_a], dim=1)
                combined_embedding_b = torch.cat([protein_embedding, ligand_embedding_b], dim=1)
            pred_a = self.combined_mlp(combined_embedding_a)
            pred_b = self.combined_mlp(combined_embedding_b)
            return pred_a - pred_b

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
