from typing import Literal

import torch
from torch import Tensor, tensor, randn
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
from torch.nn import SiLU, BatchNorm1d

import pytest

from .utils import device

import logging

logger = logging.getLogger(__name__)


class MolecularModel(nn.Module):
    def __init__(self, ligand_input_size, embedding_size, **kwargs):
        super().__init__()

        self.molecule_input_size = ligand_input_size
        self.embedding_size = embedding_size
        self.embed = _mlp(
            input_size=self.molecule_input_size,
            hidden_size=embedding_size,
            output_size=embedding_size,
            hidden_layers=4,
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
        )
        self.ligand_mlp = _mlp(
            input_size=ligand_input_size,
            hidden_size=hidden_layer_size,
            output_size=embedding_size,
            hidden_layers=4,
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
        **kwargs,
    ):
        super().__init__()
        self.embed_ligand = _mlp(
            input_size=ligand_input_size,
            hidden_size=hidden_channels,
            output_size=hidden_channels,
            hidden_layers=4,
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
            BatchNorm1d(hidden_channels),
            Dropout(p_dropout),
            Linear(hidden_channels, hidden_channels),
            SiLU(),
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
        n_bins: int = 10,
        hidden_channels: int = 512,
        p_dropout: float = 0.05,
        num_heads: int = 8,
        **kwargs,
    ):
        super().__init__(
            ligand_input_size=ligand_input_size,
            hidden_channels=hidden_channels,
            p_dropout=p_dropout,
            num_heads=num_heads,
        )
        self.n_bins = n_bins
        self.distribution_encoder = _mlp(
            input_size=self.n_bins,
            hidden_size=self.n_bins * 2,
            output_size=hidden_channels,
            hidden_layers=1,
        )
        self.embed_ligand = _mlp(
            input_size=ligand_input_size,
            hidden_size=hidden_channels,
            output_size=hidden_channels,
            hidden_layers=4,
        )
        self.num_heads = num_heads
        self.default_dist_emb = Parameter(torch.zeros(hidden_channels))
        self.combine_repr = Sequential(
            Linear(hidden_channels * 2, hidden_channels * 2),
            SiLU(),
            Linear(hidden_channels * 2, hidden_channels),
        )
        self.set_transformer = SetTransformer(
            hidden_channels=hidden_channels,
            num_heads=self.num_heads,
            ffn_hidden_layers=2,
            num_blocks=8,
            dropout=p_dropout,
        )
        self.ouput = Sequential(
            Linear(hidden_channels, hidden_channels),
            SiLU(),
            BatchNorm1d(hidden_channels),
            Linear(hidden_channels, hidden_channels),
            SiLU(),
            Linear(hidden_channels, n_bins),
        )

    def forward(
        self,
        ligand: Tensor,
        dist: Tensor,
        sample_mask: Tensor,
        set_ids: Tensor,
    ) -> Tensor:
        attn_mask = make_block_diag_mask(set_ids, num_heads=self.num_heads)
        attn_mask = torch.logical_or(make_asymmetric_mask(sample_mask))
        x_ligand = self.embed_ligand(ligand)
        x_dist = self.distribution_encoder(dist)
        x_dist[~mask] = self.default_dist_emb
        x = self.combine_repr(torch.cat((x_ligand, x_dist), 1))
        h = self.set_transformer(x, attn_mask=attn_mask)
        return self.ouput(h).squeeze()


class ComplexSetRank(MoleculeSetRank):
    def __init__(
        self,
        ligand_input_size: int,
        protein_input_size: int,
        hidden_channels: int = 512,
        p_dropout: float = 0.05,
        **kwargs,
    ):
        super().__init__(ligand_input_size, hidden_channels, p_dropout)
        self.embed_protein = _mlp(
            input_size=protein_input_size,
            hidden_size=hidden_channels,
            output_size=hidden_channels,
            hidden_layers=1,
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


class ComplexBayesianSetRankModel(MoleculeBayesianSetRankModel):
    def __init__(
        self,
        ligand_input_size: int,
        protein_input_size: int,
        hidden_channels: int,
        n_bins: int = 10,
        **kwargs,
    ):
        super().__init__(
            n_bins=n_bins,
            ligand_input_size=ligand_input_size,
            protein_input_size=protein_input_size,
            **kwargs,
        )
        self.embed_protein = _mlp(
            input_size=protein_input_size,
            hidden_size=hidden_channels,
            output_size=hidden_channels,
            hidden_layers=4,
        )
        self.combine_repr = Sequential(
            Linear(hidden_channels * 3, hidden_channels * 3),
            SiLU(),
            Linear(hidden_channels * 3, hidden_channels),
        )

    def forward(
        self,
        ligand: Tensor,
        protein: Tensor,
        dist: Tensor,
        sample_mask: Tensor,
        set_ids: Tensor,
    ) -> Tensor:
        attn_mask = make_block_diag_mask(set_ids, num_heads=self.num_heads)
        attn_mask = torch.logical_or(attn_mask, make_asymmetric_mask(sample_mask))
        x_ligand = self.embed_ligand(ligand)
        x_protein = self.embed_protein(protein)
        x_dist = self.distribution_encoder(dist)
        x_dist[sample_mask] = self.default_dist_emb
        x = self.combine_repr(torch.cat((x_ligand, x_protein, x_dist), 1))
        h = self.set_transformer(x, attn_mask=attn_mask)
        return self.ouput(h).squeeze()


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
        self,
        x: Tensor,
        y: Tensor | None = None,
        attn_mask: Tensor | None = None,
        need_weights: bool = False,
    ) -> Tensor:
        if y is None:
            y = x
        x_, attn_weights = self.attn(x, y, y, attn_mask=attn_mask, need_weights=True)
        x = self.ln1(x + x_)
        x = self.ln2(x + self.ffn(x))
        if need_weights:
            return x, attn_weights
        return x


class SetAttentionBlock(MHABlock):
    def forward(
        self, x: Tensor, attn_mask: Tensor | None = None, need_weights: bool = False
    ) -> Tensor:
        return super().forward(x, x, attn_mask, need_weights=need_weights)


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


def fit_quantile_bins(y: torch.Tensor, num_classes: int) -> torch.Tensor:
    if y.ndim != 1:
        y = y.flatten()
    probs = torch.linspace(0, 1, num_classes + 1, device=device)
    return torch.quantile(y, probs, interpolation="linear")


def digitize_labels(y: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
    return torch.bucketize(y, edges[1:-1], right=False).long()


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
    attn_mask = torch.zeros(N, N, dtype=torch.bool)

    for i in range(N):  # query index
        if masked[i]:
            # masked query: can attend to unmasked + itself
            attn_mask[i] = masked  # disallow attending masked tokens
            attn_mask[i, i] = False  # allow self
        else:
            # unmasked query: cannot attend to masked tokens
            attn_mask[i] = masked

    return attn_mask


def test_asymmetric_mask_shape_and_behavior():
    masked = torch.tensor([False, True, False, True])
    mask = make_asymmetric_mask(masked)
    print(mask)

    # Shape check
    assert mask.shape == (4, 4)

    # Unmasked (0) should not attend to masked (1, 3)
    assert mask[0, 1] and mask[0, 3]
    assert mask[1, 3]  # cannot attend to other masked
    # Masked token must still see itself
    assert mask[1, 1] == False
    # Masked (1) should not attend to other masked (1, 3) but can attend to unmasked (0, 2)
    assert not mask[1, 0]


def test_asymmetric_mask_in_attention():
    torch.manual_seed(0)
    x = randn(4, 8)
    block = SetAttentionBlock(8, 2, 1, dropout=0)

    masked = torch.tensor([False, True, False, True])
    mask = make_asymmetric_mask(masked)

    out, weights = block(x, attn_mask=mask, need_weights=True)

    for i, m in enumerate(masked):
        if not m:
            assert torch.allclose(
                weights[i, masked], torch.zeros_like(weights[i, masked]), atol=1e-6
            )


def test_no_mask_all_to_all():
    x = randn(4, 8)
    block = SetAttentionBlock(8, 4, 1, dropout=0)
    out_no_mask = block(x, attn_mask=None)
    mask = make_block_diag_mask([1, 1, 1, 1])
    out_masked = block(x, attn_mask=mask)
    torch.testing.assert_close(out_no_mask, out_masked)


def test_block_diag_mask_independent_sets():
    x = randn(8, 16)
    block = SetAttentionBlock(16, 2, 1, dropout=0)
    mask = make_block_diag_mask([1, 1, 1, 1, 2, 2, 2, 2])
    out, weights = block(x, attn_mask=mask, need_weights=True)
    torch.isclose(
        (mask.type(torch.float64) * weights).sum(), tensor(0.0, dtype=torch.float64)
    )

    # embeddings of set A should not directly depend on set B
    set_a_out = out[:4]
    set_b_out = out[4:]
    # Compute mean embedding of each set
    mean_a = set_a_out.mean(dim=0)
    mean_b = set_b_out.mean(dim=0)
    # If masking works, mean_a and mean_b should not be almost identical
    # (without mask, they'd mix more strongly)
    assert not torch.allclose(mean_a, mean_b, rtol=1e-2, atol=1e-2)


def test_invalid_mask_shape():
    x = randn(6, 10)
    block = SetAttentionBlock(10, 2, 1, dropout=0)
    wrong_mask = torch.zeros(3, 3, dtype=torch.bool)  # wrong shape
    with pytest.raises(RuntimeError):
        block(x, attn_mask=wrong_mask)


def test_settransformer_blocks_with_mask():
    x = randn(7, 12)
    model = SetTransformer(12, 2, 1, num_blocks=2, dropout=0)
    mask = make_block_diag_mask([3, 4])
    out = model(x, attn_mask=mask)
    assert out.shape == (7, 12)


if __name__ == "__main__":
    test_asymmetric_mask_shape_and_behavior()
    # test_block_diag_mask_independent_sets()
    # from torch import randn
    #
    # x = randn(10, 32)
    # sab = SetAttentionBlock(32, 4, 1)
    # x = sab(x)
    #
    # isab = InducedSetAttentionBlock(32, 4, 1, num_seeds=5)
    # x = isab(x)
