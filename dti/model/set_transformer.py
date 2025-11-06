from typing import Literal

from torch import Tensor
from torch.nn import (
    Dropout,
    LayerNorm,
    Module,
    ModuleList,
)
from torch.nn import MultiheadAttention as MHA
from torch.nn import GELU

from .common import mlp

import logging

logger = logging.getLogger(__name__)


class MHABlock(Module):
    def __init__(
        self,
        hidden_channels: int,
        num_heads: int,
        ffn_hidden_layers: int,
        dropout: float = 0.1,
        act=GELU,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads
        self.ffn_hidden_layers = ffn_hidden_layers
        self.dropout = dropout
        self.attn = MHA(hidden_channels, num_heads, dropout=dropout)
        self.dropout = Dropout(dropout)
        self.ffn = mlp(
            hidden_channels,
            hidden_channels,
            hidden_channels,
            ffn_hidden_layers,
            act=act,
        )
        self.ln1 = LayerNorm(hidden_channels)
        self.ln2 = LayerNorm(hidden_channels)
        self.ln3 = LayerNorm(hidden_channels)

    def forward(
        self,
        x: Tensor,
        y: Tensor | None = None,
        attn_mask: Tensor | None = None,
        need_weights: bool = False,
    ) -> Tensor:
        x = self.ln1(x)
        if y is None:
            y = x
        x_, attn_weights = self.attn(x, y, y, attn_mask=attn_mask, need_weights=True)
        x = self.ln2(x + x_)
        x = self.ln3(x + self.ffn(x))
        if need_weights:
            return x, attn_weights
        return x


class SetAttentionBlock(MHABlock):
    def forward(
        self, x: Tensor, attn_mask: Tensor | None = None, need_weights: bool = False
    ) -> Tensor:
        return super().forward(x, attn_mask=attn_mask, need_weights=need_weights)


class SetTransformer(Module):
    def __init__(
        self,
        hidden_channels: int,
        num_heads: int,
        ffn_hidden_layers: int,
        num_blocks: int,
        num_seeds: int = 1,
        dropout: float = 0.1,
        layer_type: Literal["full"] = "full",
        act=GELU,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.num_seeds = num_seeds
        self.dropout = dropout
        self.layer_type = layer_type
        match layer_type:
            case "full":
                self.blocks = ModuleList(
                    [
                        SetAttentionBlock(
                            hidden_channels,
                            num_heads,
                            ffn_hidden_layers,
                            dropout,
                            act=act,
                        )
                        for _ in range(num_blocks)
                    ]
                )

    def forward(
        self,
        x: Tensor,
        attn_mask: Tensor | None = None,
        need_intermediate_activations: bool = False,
    ) -> Tensor:
        if attn_mask is not None and self.layer_type == "induced":
            raise ValueError("Induced attention does not support attention mask")
        activations = [x] + [(x := block(x, attn_mask)) for block in self.blocks]
        if need_intermediate_activations:
            return activations
        return activations[-1]
