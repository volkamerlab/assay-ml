import torch
from torch import Tensor, randn
from torch.nn import (
    Parameter,
    LayerNorm,
    Linear,
    Module,
    Sequential,
)
from torch.nn import GELU


from ..utils import device
from .bin_distribution import BinDistribution
from .set_transformer import SetTransformer
from .common import make_block_diag_mask, make_asymmetric_mask, mlp

import logging

logger = logging.getLogger(__name__)


class MoleculeBayesianSetRankModel(Module):
    def __init__(
        self,
        ligand_input_size: int = 1024,
        n_bins: int = 20,
        hidden_channels: int = 512,
        p_dropout: float = 0.05,
        num_heads: int = 8,
        smoothing: bool = True,
        act=GELU,
        **kwargs,
    ):
        super().__init__()
        self.n_bins = n_bins
        self.bin_dist = BinDistribution(
            n_bins=n_bins, exp_tails=False, normalization="minmax"
        )
        self.distribution_encoder = mlp(
            input_size=1,
            hidden_size=hidden_channels,
            output_size=hidden_channels,
            hidden_layers=2,
            act=act,
        )
        self.embed_ligand = mlp(
            input_size=ligand_input_size,
            hidden_size=hidden_channels,
            output_size=hidden_channels,
            hidden_layers=2,
            act=act,
        )
        self.num_heads = num_heads
        self.default_dist_emb = Parameter(randn(hidden_channels, device=device) * 0.02)
        self.combine_repr = Sequential(
            Linear(hidden_channels * 2, hidden_channels * 3),
            act(),
            Linear(hidden_channels * 3, hidden_channels),
        )
        self.set_transformer = SetTransformer(
            hidden_channels=hidden_channels,
            num_heads=self.num_heads,
            ffn_hidden_layers=2,
            num_blocks=8,
            dropout=p_dropout,
            act=act,
        )
        self.output = Sequential(
            Linear(hidden_channels, hidden_channels),
            act(),
            LayerNorm(hidden_channels),
            Linear(hidden_channels, hidden_channels * 2),
            act(),
            Linear(hidden_channels * 2, hidden_channels),
            act(),
        )
        self.readout = Linear(hidden_channels, n_bins)

    def forward(
        self,
        ligand: Tensor,
        _protein: Tensor,  # ignored
        y: Tensor,  # raw regression targets
        sample_mask: Tensor,
        set_ids: Tensor,
    ) -> Tensor:
        attn_mask = make_block_diag_mask(set_ids, num_heads=self.num_heads)
        attn_mask = torch.logical_or(attn_mask, make_asymmetric_mask(sample_mask))
        sample_mask = sample_mask.float().unsqueeze(1)
        x_ligand = self.embed_ligand(ligand)
        y = y.unsqueeze(1)
        x_dist = self.distribution_encoder(y)
        x_dist = (1 - sample_mask) * x_dist + sample_mask * self.default_dist_emb
        h = self.combine_repr(torch.cat((x_ligand, x_dist), 1)) + x_ligand
        h = self.set_transformer(h, attn_mask=attn_mask)
        h = self.output(h) + h
        return self.readout(h)


class ComplexBayesianSetRankModel(MoleculeBayesianSetRankModel):
    pass
