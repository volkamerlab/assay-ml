import torch
from torch import Tensor, randn
from torch.nn import (
    Dropout,
    Parameter,
    LayerNorm,
    Linear,
    Module,
    Sequential,
    ModuleList,
)
from torch.nn import GELU
import torch_geometric.nn as gnn
from torch_geometric.data import Batch


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
        self.distribution_encoder = Sequential(
            Linear(1, hidden_channels),
            act(),
            Linear(hidden_channels, hidden_channels),
            act(),
            Dropout(p_dropout),
            Linear(hidden_channels, hidden_channels),
        )
        self.embed_ligand = Sequential(
            Linear(ligand_input_size, hidden_channels),
            act(),
            Linear(hidden_channels, hidden_channels),
            act(),
            Dropout(p_dropout),
            Linear(hidden_channels, hidden_channels),
        )
        self.default_dist_emb = Parameter(randn(hidden_channels, device=device) * 0.02)
        self.combine_repr = Sequential(
            Linear(hidden_channels * 2, hidden_channels),
            act(),
            LayerNorm(hidden_channels),
            Linear(hidden_channels, hidden_channels),
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
        self.output = Sequential(
            Linear(hidden_channels, hidden_channels),
            act(),
            LayerNorm(hidden_channels),
            Linear(hidden_channels, hidden_channels),
            act(),
            Linear(hidden_channels, hidden_channels),
            act(),
            Linear(hidden_channels, n_bins),
        )

    def _embed_ligand(self, ligand: Tensor) -> Tensor:
        return self.embed_ligand(ligand)

    def forward(
        self,
        ligand: Tensor | Batch,
        _protein: Tensor,  # ignored
        y: Tensor,  # raw regression targets
        sample_mask: Tensor,
        set_ids: Tensor,
    ) -> Tensor:
        attn_mask = make_block_diag_mask(set_ids, num_heads=None)
        attn_mask = torch.logical_or(attn_mask, make_asymmetric_mask(sample_mask))
        attn_mask = attn_mask.unsqueeze(0).expand(self.num_heads, -1, -1)
        sample_mask = sample_mask.float().unsqueeze(1)
        x_ligand = self._embed_ligand(ligand)
        y = y.unsqueeze(1)
        x_dist = self.distribution_encoder(y)
        x_dist = (1 - sample_mask) * x_dist + sample_mask * self.default_dist_emb
        h = self.combine_repr(torch.cat((x_ligand, x_dist), 1))
        h = self.set_transformer(h, attn_mask=attn_mask)
        h = self.output(h)
        return h


class GraphMoleculeBayesianSetRankModel(MoleculeBayesianSetRankModel):
    def __init__(
        self,
        ligand_input_size: int,
        num_gnn_layers: int = 3,
        n_bins: int = 20,
        hidden_channels: int = 512,
        p_dropout: float = 0.05,
        num_heads: int = 8,
        smoothing: bool = True,
        act=GELU,
        **kwargs,
    ):
        super().__init__(
            ligand_input_size=1,
            n_bins=n_bins,
            hidden_channels=hidden_channels,
            p_dropout=p_dropout,
            num_heads=num_heads,
            smoothing=smoothing,
            act=act,
            **kwargs,
        )

        self.gnn_layers = ModuleList()
        in_channels = ligand_input_size

        for _ in range(num_gnn_layers):
            mlp = Sequential(
                Linear(in_channels, hidden_channels),
                act(),
                Linear(hidden_channels, hidden_channels),
            )
            self.gnn_layers.append(gnn.GINConv(mlp, train_eps=True))
            in_channels = hidden_channels

        self.pooling_layer = gnn.global_add_pool

        del self.embed_ligand

    def _embed_ligand(
        self,
        ligand: Batch,
    ) -> Tensor:
        x, edge_index, batch_idx = ligand.x, ligand.edge_index, ligand.batch

        for gnn_layer in self.gnn_layers:
            x = gnn_layer(x, edge_index)

        return self.pooling_layer(x, batch_idx)


class ComplexBayesianSetRankModel(MoleculeBayesianSetRankModel):
    pass
