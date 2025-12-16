from typing import Literal
import torch
from torch import Tensor, tensor
from torch.nn import (
    Dropout,
    LayerNorm,
    Linear,
    Module,
    Parameter,
    SiLU,
    Sequential,
    ModuleList,
    init,
)
from torch.nn import MultiheadAttention as MHA


def _mlp(
    input_size: int,
    hidden_size: int,
    output_size: int,
    hidden_layers: int,
    act=SiLU,
) -> Module:
    if hidden_layers == 0:
        return Linear(input_size, output_size)
    layers = [Linear(input_size, hidden_size), act()]
    for _ in range(hidden_layers - 1):
        layers.extend([Linear(hidden_size, hidden_size), act()])
    layers.append(Linear(hidden_size, output_size))
    return Sequential(*layers)


class GaussianFourierProjection(Module):
    def __init__(self, embed_dim: int, scale: float = 2.0):
        super().__init__()
        self.W = Parameter(torch.randn(1, embed_dim // 2) * scale, requires_grad=False)
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_proj = x * 2 * torch.pi * self.W
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class MHABlock(Module):
    def __init__(
        self,
        hidden_channels: int,
        num_heads: int,
        ffn_hidden_layers: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads
        self.ffn_hidden_layers = ffn_hidden_layers
        self.dropout = dropout
        self.attn = MHA(hidden_channels, num_heads, dropout=dropout)
        self.dropout = Dropout(dropout)
        self.ffn = _mlp(
            hidden_channels, hidden_channels, hidden_channels, ffn_hidden_layers
        )
        self.ln1 = LayerNorm(hidden_channels)
        self.ln2 = LayerNorm(hidden_channels)

    def forward(
        self, x: Tensor, y: Tensor | None = None, attn_mask: Tensor | None = None
    ) -> Tensor:
        if y is None:
            y = x
        x_, attn_weigths = self.attn(x, y, y, attn_mask=attn_mask)
        x = self.ln1(x + x_)
        x = self.ln2(x + self.ffn(x))
        return x


class SetAttentionBlock(MHABlock):
    def forward(self, x: Tensor, attn_mask: Tensor | None = None) -> Tensor:
        return super().forward(x, x, attn_mask)


class InducedSetAttentionBlock(Module):
    def __init__(
        self,
        hidden_channels: int,
        num_heads: int,
        ffn_hidden_layers: int,
        dropout: float = 0.1,
        num_seeds: int = 1,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads
        self.ffn_hidden_layers = ffn_hidden_layers
        self.dropout = dropout
        self.num_seeds = num_seeds
        self.seed_query_attention = MHABlock(
            hidden_channels, num_heads, ffn_hidden_layers, dropout
        )
        self.query_seed_attention = MHABlock(
            hidden_channels, num_heads, ffn_hidden_layers, dropout
        )
        self.seeds = Parameter(tensor(num_seeds, hidden_channels), requires_grad=True)
        init.xavier_normal_(self.seeds)

    def forward(self, x: Tensor, attn_mask: Tensor | None = None) -> Tensor:
        h = self.seed_query_attention(self.seeds, x)
        return self.query_seed_attention(x, h)


class SetTransformer(Module):
    def __init__(
        self,
        hidden_channels: int,
        num_heads: int,
        ffn_hidden_layers: int,
        num_blocks: int,
        num_seeds: int = 1,
        dropout: float = 0.1,
        layer_type: Literal["full", "induced"] = "full",
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
                            hidden_channels, num_heads, ffn_hidden_layers, dropout
                        )
                        for _ in range(num_blocks)
                    ]
                )
            case "induced":
                self.blocks = ModuleList(
                    [
                        InducedSetAttentionBlock(
                            hidden_channels,
                            num_heads,
                            ffn_hidden_layers,
                            dropout,
                            num_seeds,
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


class SelfConditionedSetTransformer(SetTransformer):
    """
    Iterative refinement transformer.
    Feeds the ranking score from Layer K as a positional encoding into Layer K+1.
    """

    def __init__(
        self,
        hidden_channels: int,
        num_heads: int,
        ffn_hidden_layers: int,
        num_blocks: int,
        num_seeds: int = 1,
        dropout: float = 0.1,
        layer_type="full",
    ):
        super().__init__(
            hidden_channels,
            num_heads,
            ffn_hidden_layers,
            num_blocks,
            num_seeds,
            dropout,
            layer_type,
        )

        self.pos_enc = GaussianFourierProjection(embed_dim=hidden_channels)

        self.readouts = ModuleList(
            [
                Sequential(
                    Linear(hidden_channels, hidden_channels // 2),
                    SiLU(),
                    Linear(hidden_channels // 2, 1),
                )
                for _ in range(num_blocks + 1)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        return_all_layers: bool = False,
    ) -> torch.Tensor:
        predictions = []

        curr_pred = self.readouts[0](x)
        predictions.append(curr_pred)

        pos_emb = self.pos_enc(curr_pred)

        for block, readout in zip(self.blocks, self.readouts[1:]):
            x = block(x + pos_emb, attn_mask)

            curr_pred = readout(x)
            predictions.append(curr_pred)

            pos_emb = self.pos_enc(curr_pred)

        if return_all_layers:
            return torch.stack(predictions).squeeze(-1)

        return predictions[-1]


if __name__ == "__main__":
    from torch import randn

    x = randn(10, 32)
    sab = SetAttentionBlock(32, 4, 1)
    x = sab(x)

    isab = InducedSetAttentionBlock(32, 4, 1, num_seeds=5)
    x = isab(x)
