import torch
from torch import nn, Tensor, tensor
from typing import Literal
from torch.nn import (
    Parameter,
    Dropout,
    Linear,
    Module,
    SiLU,
    Sequential,
    LayerNorm,
    ModuleList,
    init,
)
from torch.nn import MultiheadAttention as MHA


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
        hidden_channels=512,
        p_dropout=0.05,
        **kwargs,
    ):
        super().__init__()

        self.ligand_input_size = ligand_input_size
        self.embedding_size = embedding_size
        self.embed_protein = nn.Sequential(
            nn.Linear(protein_input_size, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )

        self.embed_ligand = nn.Sequential(
            nn.Linear(ligand_input_size, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )

        self.output = Sequential(
            nn.BatchNorm1d(hidden_channels * 2),
            Linear(hidden_channels * 2, hidden_channels),
            SiLU(),
            Linear(hidden_channels, hidden_channels),
            SiLU(),
            Dropout(p_dropout),
            Linear(hidden_channels, 1),
        )

    def forward(self, protein, ligand):
        if ligand.dim() == 3:
            ligand = ligand.squeeze()
            protein = protein.squeeze()

        assert ligand.shape[1] == self.ligand_input_size, ligand.shape

        protein_emb = self.embed_protein(protein)
        ligand_emb = self.embed_ligand(ligand)
        combined_emb = torch.cat([protein_emb, ligand_emb], dim=1)
        output = self.output(combined_emb)
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
        self.hidden_channels = hidden_channels
        self.embed_ligand = nn.Sequential(
            nn.Linear(ligand_input_size, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.num_heads = num_heads

        self.set_transformer = SelfConditionedSetTransformer(
            hidden_channels=hidden_channels,
            num_heads=self.num_heads,
            ffn_hidden_layers=2,
            num_blocks=4,
            dropout=p_dropout,
        )

        self.output = Sequential(
            LayerNorm(hidden_channels),
            Linear(hidden_channels, hidden_channels),
            SiLU(),
            Dropout(p_dropout),
            Linear(hidden_channels, 1),
        )

    def forward(self, protein, ligand, attn_mask=None):
        pass


class SetRankModel(MoleculeSetRank):
    def __init__(
        self,
        ligand_input_size: int,
        protein_input_size: int,
        hidden_channels: int = 512,
        p_dropout: float = 0.05,
        **kwargs,
    ):
        super().__init__(ligand_input_size, hidden_channels, p_dropout, **kwargs)

        self.embed_protein = nn.Sequential(
            nn.Linear(protein_input_size, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )

        self.fusion_proj = nn.Linear(hidden_channels * 2, hidden_channels)

    def forward(
        self,
        protein: torch.Tensor,
        ligand: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        return_all_layers: bool = False,
    ) -> torch.Tensor:
        x_prot = self.embed_protein(protein)
        x_lig = self.embed_ligand(ligand)

        if x_prot.size(0) != x_lig.size(0):
            x_prot = x_prot.expand(x_lig.size(0), -1)

        fused = torch.cat([x_lig, x_prot], dim=-1)
        x = self.fusion_proj(fused)

        h = self.set_transformer(
            x, return_all_layers=return_all_layers, attn_mask=attn_mask
        )
        if h.shape[-1] == self.hidden_channels:
            return self.output(h)
        return h


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
            return torch.stack(predictions)

        return predictions[-1]
