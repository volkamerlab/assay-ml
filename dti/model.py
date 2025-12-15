import torch
from torch import nn, Tensor
from torch.nn import Dropout, Linear, Module, SiLU, Sequential, BatchNorm1d, LayerNorm

from .set_rank.set_transformer import SetTransformer, _mlp

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
        if molecule.dim() == 3:
            molecule = molecule.squeeze()
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
    ):
        super().__init__()

        self.ligand_input_size = ligand_input_size
        self.cosine_agg = cosine_agg
        self.embedding_size = embedding_size
        self.protein_mlp = nn.Sequential(
            nn.Linear(protein_input_size, hidden_layer_size),
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
        if ligand.dim() == 3:
            ligand = ligand.squeeze()
            protein = protein.squeeze()

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
            diff = (x.view(n, 1, d) - x.view(1, n, d)).reshape(n**2, d)
        elif x.dim() == 3:  # pre-defined pairs (k, 2, d)
            x = x * protein_emb.unsqueeze(1)
            diff = x[:, 0, :] - x[:, 1, :]
        else:
            assert False

        return self.combined_mlp(diff) - self.combined_mlp(-diff)


class MoleculeSetRank(Module):
    def __init__(
        self,
        ligand_input_size: int,
        hidden_channels: int = 512,
        p_dropout: float = 0.05,
        num_heads: int = 8,
        **kwargs,
    ):
        super().__init__()
        self.embed_ligand =nn.Sequential(
            nn.Linear(ligand_input_size, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.num_heads = num_heads
        self.set_transformer = SetTransformer(
            hidden_channels=hidden_channels,
            num_heads=self.num_heads,
            ffn_hidden_layers=2,
            num_blocks=8,
            dropout=0.0,
        )
        self.ouput = Sequential(
            Linear(hidden_channels, hidden_channels),
            SiLU(),
            LayerNorm(hidden_channels),
            Dropout(p_dropout),
            Linear(hidden_channels, hidden_channels),
            SiLU(),
            Linear(hidden_channels, 1),
        )

    def forward(
        self,
        _protein: Tensor,
        ligand: Tensor,
        attn_mask: Tensor | None = None,
    ) -> Tensor:
        """
        Only supports batch size 1 (ie 1 intra assay group of molecule)

        Args:
            ligand (Tensor): shape (N, ligand_input_size)
            protein (Tensor): shape (N, protein_input_size)

        Returns:
            Tensor: unnormalized ranking scores (N, 1)
        """
        x_ligand = self.embed_ligand(ligand.squeeze())
        h = self.set_transformer(x_ligand, attn_mask=attn_mask)
        return self.ouput(h).squeeze()


class SetRankModel(MoleculeSetRank):
    def __init__(
        self,
        ligand_input_size: int,
        protein_input_size: int,
        hidden_channels: int = 512,
        p_dropout: float = 0.05,
        **kwargs,
    ):
        super().__init__(ligand_input_size, hidden_channels, p_dropout)
        self.embed_protein = nn.Sequential(
            nn.Linear(protein_input_size, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )

    def combine_with_query(self, x: Tensor, query: Tensor) -> Tensor:
        return x * query

    def forward(
        self,
        protein: Tensor,
        ligand: Tensor,
        attn_mask: Tensor | None = None,
    ) -> Tensor:
        """
        Only supports batch size 1 (ie 1 intra assay group of molecule)

        Args:
            ligand (Tensor): shape (N, ligand_input_size)
            protein (Tensor): shape (N, protein_input_size)

        Returns:
            Tensor: unnormalized ranking scores (N, 1)
        """
        x_ligand = self.embed_ligand(ligand)
        x_protein = self.embed_protein(protein)
        x = self.combine_with_query(x_ligand, x_protein)
        h = self.set_transformer(x, attn_mask=attn_mask)
        return self.ouput(h).squeeze()
