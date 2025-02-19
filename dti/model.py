import torch
from torch import nn, Tensor

import logging

logger = logging.getLogger(__name__)


class MolecularModel(nn.Module):
    def __init__(self, ligand_input_size, embedding_size, **kwargs):
        super().__init__()

        self.molecule_input_size = ligand_input_size
        self.embedding_size = embedding_size
        self.embed = nn.Sequential(
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
        assert molecule.shape[1] == self.molecule_input_size, molecule.shape
        molecule = self.embed(molecule)
        return self.readout(molecule)


class PairMolecularModel(MolecularModel):
    def __init__(self, ligand_input_size, embedding_size, **kwargs):
        super().__init__(ligand_input_size, embedding_size, **kwargs)

    def forward(self, _protein, molecule: Tensor):
        x = self.embed(molecule)
        if x.dim() == 2:  # full (b, b) pairs
            n, d = molecule.size(0), self.embedding_size
            diff_a = (x.view(n, 1, d) - x.view(1, n, d)).reshape(-1, d)  # (b, b, d)
            diff_b = (x.view(1, n, d) - x.view(n, 1, d)).reshape(-1, d)  # (b, b, d)
        elif x.dim() == 3:  # assume pre-defined pairs (k, 2, d)
            diff_a = x[:, 0, :] - x[:, 1, :]  # (k, 2, d)
            diff_b = x[:, 1, :] - x[:, 0, :]
        return self.readout(diff_a) - self.readout(diff_b)


class CombinedModel(nn.Module):
    def __init__(
        self,
        protein_input_size,
        ligand_input_size,
        embedding_size,
        hidden_layer_size=512,
        cosine_agg=True,
    ):
        super().__init__()

        self.ligand_input_size = ligand_input_size
        self.cosine_agg = cosine_agg
        self.embedding_size = embedding_size
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

        scaled_hidden_dim = hidden_layer_size // 2
        self.combined_mlp = nn.Sequential(
            nn.Dropout(0.05),
            nn.BatchNorm1d(embedding_size),
            nn.Linear(embedding_size, hidden_layer_size),
            nn.SiLU(),
            nn.Linear(hidden_layer_size, scaled_hidden_dim),
            nn.SiLU(),
            nn.Linear(scaled_hidden_dim, 1),
        )

    def forward(self, protein, ligand):
        assert ligand.shape[1] == self.ligand_input_size, ligand.shape

        protein_emb = self.protein_mlp(protein)
        ligand_emb = self.ligand_mlp(ligand)

        if self.cosine_agg:
            combined_emb = protein_emb * ligand_emb
        else:
            assert False  # FIXME: debugging only
            combined_emb = torch.cat([protein_emb, ligand_emb], dim=1)
        output = self.combined_mlp(combined_emb)
        return output


class PairCombinedModel(CombinedModel):
    def __init__(
        self,
        protein_input_size,
        ligand_input_size,
        embedding_size,
        hidden_layer_size=512,
        cosine_agg=True,
    ):
        super().__init__(
            protein_input_size,
            ligand_input_size,
            embedding_size,
            hidden_layer_size=512,
            cosine_agg=False,
        )

    def forward(self, protein, ligand):
        protein_emb = self.protein_mlp(protein)
        x = self.ligand_mlp(ligand)
        if x.dim() == 2:  # all pairs
            x = protein_emb * x
            n, d = ligand.size(0), self.embedding_size
            diff_a = (x.view(n, 1, d) - x.view(1, n, d)).reshape(-1, d)  # (b, b, d)
            diff_b = (x.view(1, n, d) - x.view(n, 1, d)).reshape(-1, d)  # (b, b, d)
        elif x.dim() == 3:  # pre-defined pairs (k, 2, d)
            x = x * protein_emb.unsqueeze(1)
            xa = protein_emb * x[:, 0, :]
            xb = protein_emb * x[:, 1, :]
            diff_a = xa - xb
            diff_b = xb - xa
        else:
            assert False

        return self.combined_mlp(diff_a) - self.combined_mlp(diff_b)

    # def forward(self, protein, ligand):
    #     protein_emb = self.protein_mlp(protein)
    #
    #     if ligand.shape[1] == self.ligand_input_size * 2:
    #         ligand_a = ligand[:, : self.ligand_input_size]
    #         ligand_b = ligand[:, self.ligand_input_size :]
    #         ligand_emb_a = self.ligand_mlp(ligand_a)
    #         ligand_emb_b = self.ligand_mlp(ligand_b)
    #
    #         if self.cosine_agg:
    #             combined_emb_a = protein_emb * ligand_emb_a
    #             combined_emb_b = protein_emb * ligand_emb_b
    #         else:
    #             combined_emb_a = torch.cat([protein_emb, ligand_emb_a], dim=1)
    #             combined_emb_b = torch.cat([protein_emb, ligand_emb_b], dim=1)
    #         pred_ab = self.combined_mlp(combined_emb_a - combined_emb_b)
    #         pred_ba = self.combined_mlp(combined_emb_b - combined_emb_a)
    #         return pred_ab - pred_ba
    #     elif ligand.shape[1] == self.ligand_input_size:
    #         ligand_emb = self.ligand_mlp(ligand)
    #
    #         if self.cosine_agg:
    #             combined_emb = protein_emb * ligand_emb
    #         else:
    #             combined_emb = torch.cat([protein_emb, ligand_emb], dim=1)
    #         output = self.combined_mlp(combined_emb)
    #         return output
    #     else:
    #         raise ValueError(f"Unexpected ligand shape {ligand.shape}")
