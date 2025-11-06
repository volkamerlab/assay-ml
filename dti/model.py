from typing import Literal

import torch
from torch import Tensor, randn
from torch.nn import (
    Dropout,
    Parameter,
    LayerNorm,
    Linear,
    Module,
    ReLU,
    Sequential,
    ModuleList,
)
from torch.nn import MultiheadAttention as MHA
from torch import nn
from torch.nn import GELU, BatchNorm1d


from .utils import device
from .bin_distribution import BinDistribution

import logging

logger = logging.getLogger(__name__)

def _mlp(
    input_size: int,
    hidden_size: int,
    output_size: int,
    hidden_layers: int,
    act=GELU,
) -> Module:
    if hidden_layers == 0:
        return Linear(input_size, output_size)
    layers = [Linear(input_size, hidden_size), act()]
    for _ in range(hidden_layers - 1):
        layers.extend([Linear(hidden_size, hidden_size), act()])
    layers.append(Linear(hidden_size, output_size))
    return Sequential(*layers)


class MolecularModel(nn.Module):
    def __init__(self, ligand_input_size, embedding_size, act=GELU, **kwargs):
        super().__init__()

        self.molecule_input_size = ligand_input_size
        self.embedding_size = embedding_size
        self.embed = _mlp(
            input_size=self.molecule_input_size,
            hidden_size=embedding_size,
            output_size=embedding_size,
            hidden_layers=4,
            act=act,
        )
        scaled_hidden_dim = embedding_size // 2
        self.readout = nn.Sequential(
            nn.Dropout(0.05),
            nn.BatchNorm1d(embedding_size),
            nn.Linear(embedding_size, embedding_size),
            act(),
            nn.Linear(embedding_size, scaled_hidden_dim),
            act(),
            nn.Linear(scaled_hidden_dim, 1),
        )

    def forward(self, _protein, molecule, **kwargs):
        # the  first argument (protein embeddings) is ignored
        if molecule.dim() == 3:
            molecule = molecule.squeeze()
        assert molecule.shape[1] == self.molecule_input_size, molecule.shape
        molecule = self.embed(molecule)
        return self.readout(molecule)


class PairMolecularModel(MolecularModel):
    def __init__(self, ligand_input_size, embedding_size, **kwargs):
        super().__init__(ligand_input_size, embedding_size, **kwargs)

    def forward(self, _protein, molecule: Tensor, **kwargs):
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
        act=GELU,
        **kwargs,
    ):
        super().__init__()

        self.ligand_input_size = ligand_input_size
        self.cosine_agg = cosine_agg
        self.embedding_size = embedding_size
        self.protein_mlp = _mlp(
            input_size=protein_input_size,
            hidden_size=hidden_layer_size,
            output_size=embedding_size,
            hidden_layers=4,
            act=act,
        )
        self.ligand_mlp = _mlp(
            input_size=ligand_input_size,
            hidden_size=hidden_layer_size,
            output_size=embedding_size,
            hidden_layers=4,
            act=act,
        )

        scaled_hidden_dim = hidden_layer_size // 2
        self.combined_mlp = nn.Sequential(
            nn.Dropout(0.05),
            nn.BatchNorm1d(embedding_size),
            nn.Linear(embedding_size, hidden_layer_size),
            act(),
            nn.Linear(hidden_layer_size, scaled_hidden_dim),
            act(),
            nn.Linear(scaled_hidden_dim, 1),
        )

    def forward(self, protein, ligand, **kwargs):
        if ligand.dim() == 3:
            ligand = ligand.squeeze()
            protein = protein.squeeze()

        assert ligand.shape[1] == self.ligand_input_size, ligand.shape

        protein_emb = self.protein_mlp(protein)
        ligand_emb = self.ligand_mlp(ligand)

        if self.cosine_agg:
            combined_emb = protein_emb * ligand_emb
        else:
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
        act=GELU,
        **kwargs,
    ):
        super().__init__(
            protein_input_size,
            ligand_input_size,
            embedding_size,
            hidden_layer_size=512,
            cosine_agg=False,
            act=act,
        )

    def forward(self, protein, ligand, **kwargs):
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
        act=GELU,
        **kwargs,
    ):
        super().__init__()
        self.embed_ligand = _mlp(
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
        self.ouput = Sequential(
            Linear(hidden_channels, hidden_channels),
            act(),
            LayerNorm(hidden_channels),
            Dropout(p_dropout),
            Linear(hidden_channels, hidden_channels),
            act(),
            Linear(hidden_channels, 1),
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
        x_ligand = self.embed_ligand(ligand.squeeze())
        attn_mask = make_block_diag_mask(set_ids, num_heads=self.num_heads)
        h = self.set_transformer(x_ligand, attn_mask=attn_mask)
        return self.ouput(h).squeeze()


class MoleculeBayesianSetRankModel(MoleculeSetRank):
    def __init__(
        self,
        ligand_input_size: int,
        n_bins: int = 20,
        hidden_channels: int = 512,
        p_dropout: float = 0.05,
        num_heads: int = 8,
        smoothing: bool = True,
        act=GELU,
        **kwargs,
    ):
        super().__init__(
            ligand_input_size=ligand_input_size,
            hidden_channels=hidden_channels,
            p_dropout=p_dropout,
            num_heads=num_heads,
        )
        self.n_bins = n_bins
        self.bin_dist = BinDistribution(
            n_bins=n_bins, exp_tails=False, normalization="minmax"
        )
        self.distribution_encoder = _mlp(
            input_size=1,
            hidden_size=hidden_channels,
            output_size=hidden_channels,
            hidden_layers=2,
            act=act,
        )
        self.embed_ligand = _mlp(
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


class ComplexSetRank(MoleculeSetRank):
    def __init__(
        self,
        ligand_input_size: int,
        protein_input_size: int,
        hidden_channels: int = 512,
        p_dropout: float = 0.05,
        **kwargs,
    ):
        super().__init__(ligand_input_size, hidden_channels, p_dropout, act=act)
        self.embed_protein = _mlp(
            input_size=protein_input_size,
            hidden_size=hidden_channels,
            output_size=hidden_channels,
            hidden_layers=1,
            act=act,
        )

    def combine_with_query(self, x: Tensor, query: Tensor) -> Tensor:
        return x * query

    def forward(
        self,
        protein: Tensor,
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
        x_protein = self.embed_protein(protein)
        x = self.combine_with_query(x_ligand, x_protein)
        attn_mask = make_block_diag_mask(set_ids, num_heads=self.num_heads)
        h = self.set_transformer(x, attn_mask=attn_mask)
        return self.ouput(h).squeeze()



class ComplexBayesianSetRankModel(MoleculeBayesianSetRankModel):
    pass


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
        self.ffn = _mlp(
            hidden_channels, hidden_channels, hidden_channels, ffn_hidden_layers, act=act
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
                            hidden_channels, num_heads, ffn_hidden_layers, dropout, act=act
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


def make_block_diag_mask(set_ids: torch.Tensor, num_heads: int = None) -> torch.Tensor:
    """
    Create a mask where elements can only attend within their set.

    Args:
        set_ids: Tensor of shape (N,) with set identifier for each element
        num_heads: Number of attention heads (if None, returns 2D mask)

    Returns:
        Attention mask of shape (N, N) or (num_heads, N, N) where True means "mask out" (no attention)
    """
    set_ids = set_ids.flatten()
    mask = set_ids.unsqueeze(0) != set_ids.unsqueeze(1)  # (N, N)

    if num_heads is not None:
        mask = mask.unsqueeze(0).expand(num_heads, -1, -1)  # (num_heads, N, N)

    return mask


def make_asymmetric_mask(masked: torch.Tensor) -> torch.Tensor:
    """
    Build an asymmetric attention mask:
    - masked tokens (True) can attend to unmasked tokens (False) and themselves
    - unmasked tokens cannot attend to masked tokens
    Args:
        masked: Bool tensor of shape (N,), True if the token is masked
    Returns:
        attn_mask: Bool tensor of shape (N, N)
    """
    N = masked.shape[0]
    attn_mask = torch.zeros(N, N, dtype=torch.bool, device=device)

    for i in range(N):  # query index
        attn_mask[i] = masked  # disallow attending masked tokens
        if masked[i]:
            attn_mask[i, i] = False  # allow self

    return attn_mask
