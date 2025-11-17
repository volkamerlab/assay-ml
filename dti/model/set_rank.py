import torch
from torch import Tensor
from torch import nn
from torch.nn import (
    Dropout,
    LayerNorm,
    Linear,
    Module,
    Sequential,
)
from torch.nn import GELU


from .set_transformer import SetTransformer
from .common import make_block_diag_mask, mlp

import logging

logger = logging.getLogger(__name__)


class MoleculeSetRank(Module):
    def __init__(
        self,
        ligand_input_size: int,
        hidden_channels: int = 512,
        p_dropout: float = 0.05,
        num_heads: int = 8,
        act=GELU,
        **kwargs,
    ):
        super().__init__()
        self.embed_ligand = mlp(
            input_size=ligand_input_size,
            hidden_size=hidden_channels,
            output_size=hidden_channels,
            hidden_layers=4,
            act=act,
        )
        self.num_heads = num_heads
        self.set_transformer = SetTransformer(
            hidden_channels=hidden_channels,
            num_heads=self.num_heads,
            ffn_hidden_layers=2,
            num_blocks=8,
            dropout=p_dropout,
            act=act,
        )
        scaled_hidden_dim = hidden_channels // 2
        self.output = Sequential(
            nn.Dropout(p_dropout),
            nn.LayerNorm(hidden_channels),
            nn.Linear(hidden_channels, hidden_channels),
            act(),
            nn.Linear(hidden_channels, scaled_hidden_dim),
            act(),
            nn.Linear(scaled_hidden_dim, 1),
        )

    def forward(
        self,
        _protein: Tensor,
        ligand: Tensor,
        set_ids: Tensor,
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
        attn_mask = make_block_diag_mask(set_ids, num_heads=self.num_heads)
        h = self.set_transformer(x_ligand, attn_mask=attn_mask)
        return self.output(h).squeeze()


class ComplexSetRank(MoleculeSetRank):
    def __init__(
        self,
        ligand_input_size: int,
        protein_input_size: int,
        hidden_channels: int = 512,
        p_dropout: float = 0.05,
        act=GELU,
        cosine_agg: bool = False,
        **kwargs,
    ):
        super().__init__(ligand_input_size, hidden_channels, p_dropout, act=act)
        self.embed_protein = mlp(
            input_size=protein_input_size,
            hidden_size=hidden_channels,
            output_size=hidden_channels,
            hidden_layers=4,
            act=act,
        )
        self.cosine_agg = cosine_agg
        if not self.cosine_agg:
            self.combine = Linear(hidden_channels * 2, hidden_channels)

    def combine_with_query(self, x: Tensor, query: Tensor) -> Tensor:
        if self.cosine_agg:
            return x * query
        else:
            return self.combine(torch.cat([x, query], dim=1))

    def forward(self, protein, ligand, set_ids):
        logger.debug(
            f"protein: {protein.device}, ligand: {ligand.device}, set_ids: {set_ids.device}"
        )
        attn_mask = make_block_diag_mask(set_ids, num_heads=self.num_heads).to(
            set_ids.device
        )
        logger.debug(f"mask: {attn_mask.device}")

        x_ligand = self.embed_ligand(ligand)
        logger.debug(f"x_ligand: {x_ligand.device}")

        x_protein = self.embed_protein(protein)
        logger.debug(f"x_protein: {x_protein.device}")

        x = self.combine_with_query(x_ligand, x_protein)
        logger.debug(f"x (combined): {x.device}")

        h = self.set_transformer(x, attn_mask=attn_mask)
        logger.debug(f"h: {h.device}")

        out = self.output(h)
        logger.debug(f"output: {out.device}")
        return out.squeeze()

    # def forward(
    #     self,
    #     protein: Tensor,
    #     ligand: Tensor,
    #     set_ids: Tensor,
    # ) -> Tensor:
    #     """
    #     Only supports batch size 1 (ie 1 intra assay group of molecule)
    #
    #     Args:
    #         ligand (Tensor): shape (N, ligand_input_size)
    #         protein (Tensor): shape (N, protein_input_size)
    #
    #     Returns:
    #         Tensor: unnormalized ranking scores (N, 1)
    #     """
    #     x_ligand = self.embed_ligand(ligand)
    #     x_protein = self.embed_protein(protein)
    #     x = self.combine_with_query(x_ligand, x_protein)
    #     attn_mask = make_block_diag_mask(set_ids, num_heads=self.num_heads)
    #     h = self.set_transformer(x, attn_mask=attn_mask)
    #     return self.output(h).squeeze()
