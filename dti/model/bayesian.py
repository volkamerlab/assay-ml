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
from ..data.featurization import EDGE_FEATURE_DIM, NODE_FEATURE_DIM
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
        ligand: Tensor | Batch | tuple,
        y: Tensor,
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
        edge_input_size: int = EDGE_FEATURE_DIM,
        num_gnn_layers: int = 3,
        hidden_channels: int = 512,
        p_dropout: float = 0.05,
        act=GELU,
        num_heads: int = 4,
        **kwargs,
    ):
        super().__init__(
            ligand_input_size=ligand_input_size,
            hidden_channels=hidden_channels,
            p_dropout=p_dropout,
            act=act,
            num_heads=num_heads,
            **kwargs,
        )
        del self.embed_ligand
        self.act = act

        if hidden_channels % num_heads != 0:
            raise ValueError(
                f"hidden_channels ({hidden_channels}) must be divisible "
                f"by num_heads ({num_heads})"
            )
        head_channels = hidden_channels // num_heads

        self.gnn_layers = ModuleList()
        self.norms = ModuleList()
        in_channels = ligand_input_size

        for i in range(num_gnn_layers):
            self.gnn_layers.append(
                gnn.GATv2Conv(
                    in_channels=in_channels,
                    out_channels=head_channels,
                    heads=num_heads,
                    concat=True,
                    edge_dim=edge_input_size,
                )
            )
            self.norms.append(LayerNorm(hidden_channels))
            in_channels = hidden_channels

        gate_nn = Sequential(
            Linear(hidden_channels, hidden_channels // 2),
            self.act(),
            Linear(hidden_channels // 2, 1),
        )

        self.pooling = gnn.GlobalAttention(gate_nn=gate_nn, nn=None)

    def _embed_ligand(self, ligand: Batch) -> Tensor:
        x, edge_index, batch_idx = ligand.x, ligand.edge_index, ligand.batch
        edge_attr = ligand.edge_attr.float()

        for gnn_layer, bn in zip(self.gnn_layers, self.norms):
            x = gnn_layer(x, edge_index, edge_attr)
            x = bn(x)

        h_graph = self.pooling(x, batch_idx)
        return h_graph


class AllMoleculeBayesianSetRankModel(GraphMoleculeBayesianSetRankModel):
    def __init__(
        self,
        ligand_input_size: int,
        node_input_size: int = NODE_FEATURE_DIM,
        hidden_channels: int = 512,
        p_dropout: float = 0.05,
        act=GELU,
        **kwargs,
    ):
        super().__init__(
            node_input_size,
            hidden_channels=hidden_channels,
            p_dropout=p_dropout,
            act=act,
            **kwargs,
        )
        self.embed_fp_ligand = Sequential(
            Linear(ligand_input_size, hidden_channels),
            act(),
            Linear(hidden_channels, hidden_channels),
            act(),
            Dropout(p_dropout),
            Linear(hidden_channels, hidden_channels),
        )
        self.ln_graph = LayerNorm(hidden_channels)
        self.ln_fps = LayerNorm(hidden_channels)
        self.combine_fp_graph = Linear(hidden_channels * 2, hidden_channels)

    def _embed_ligand(self, ligand: tuple) -> Tensor:
        ligand_graph, ligand_fp = ligand
        graph_emb = super()._embed_ligand(ligand_graph)
        fp_emb = self.embed_fp_ligand(ligand_fp)
        h = torch.cat([self.ln_graph(graph_emb), self.ln_fps(fp_emb)], dim=1)
        return self.combine_fp_graph(h)


class ComplexBayesianSetRankModel(MoleculeBayesianSetRankModel):
    pass
