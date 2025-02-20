from typing import Literal
from torch import Tensor, tensor, stack, zeros
from torch.nn import (
    Dropout,
    LayerNorm,
    Linear,
    Module,
    Parameter,
    ReLU,
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
    act=ReLU,
) -> Module:
    if hidden_layers == 0:
        return Linear(input_size, output_size)
    layers = [Linear(input_size, hidden_size), act()]
    for _ in range(hidden_layers - 1):
        layers.extend([Linear(hidden_size, hidden_size), act()])
    layers.append(Linear(hidden_size, output_size))
    return Sequential(*layers)


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
    def _readout_ffn(self) -> Module:
        return Sequential(
            Linear(self.hidden_channels, self.hidden_channels // 2),
            ReLU(),
            Linear(self.hidden_channels // 2, 1),
        )

    def __init__(
        self,
        hidden_channels: int,
        num_heads: int,
        ffn_hidden_layers: int,
        num_blocks: int,
        num_seeds: int = 1,
        dropout: float = 0.1,
        layer_type: Literal["full", "induced"] = "full",
        pos_enc_type: str = "rbf",
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
        self.pos_enc: Module = ...  # TODO
        self.readouts = ModuleList([self._readout_ffn() for _ in range(num_blocks + 1)])

    def forward(
        self,
        x: Tensor,
        attn_mask: Tensor | None,
        force_initial_pos: Tensor | None = None,
        need_encoded_positions: bool = False,
    ) -> Tensor:
        predictions = []
        encoded_predictions = []
        if force_initial_pos is not None:
            assert force_initial_pos.size(0) == x.size(0)
            assert force_initial_pos.size(1) == self.hidden_channels
            predictions.append(zeros(x.size(0), 1))
            encoded_predictions.append(force_initial_pos)
        else:
            predictions.append(self.readouts[0](x))
            encoded_predictions.append(self.pos_enc(predictions[0]))
        for block, readout in zip(self.blocks, self.readouts[1:]):
            x = block(x + encoded_predictions[-1], attn_mask)
            predictions.append(readout(x))
            encoded_predictions.append(self.pos_enc(predictions[-1]))
        if need_encoded_positions:
            return stack(predictions), stack(encoded_predictions)
        return stack(predictions)


if __name__ == "__main__":
    from torch import randn

    x = randn(10, 32)
    sab = SetAttentionBlock(32, 4, 1)
    x = sab(x)

    isab = InducedSetAttentionBlock(32, 4, 1, num_seeds=5)
    x = isab(x)
