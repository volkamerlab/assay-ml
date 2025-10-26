from typing import Literal

import torch
from torch import Tensor
from torch.nn import (
    Dropout,
    Parameter,
    LayerNorm,
    Linear,
    Module,
    Sequential,
)
from torch.nn import MultiheadAttention as MHA
from torch import nn
from torch.nn import SiLU, BatchNorm1d


from .utils import device
from .bin_distribution import BinDistribution

import logging

logger = logging.getLogger(__name__)

_act = SiLU


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
            _act(),
            nn.Linear(embedding_size, scaled_hidden_dim),
            _act(),
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
        )
        self.ligand_mlp = _mlp(
            input_size=ligand_input_size,
            hidden_size=hidden_layer_size,
            output_size=embedding_size,
            hidden_layers=4,
        )

        scaled_hidden_dim = hidden_layer_size // 2
        self.combined_mlp = nn.Sequential(
            nn.Dropout(p_dropout),
            nn.BatchNorm1d(embedding_size),
            nn.Linear(embedding_size, hidden_layer_size),
            _act(),
            nn.Linear(hidden_layer_size, scaled_hidden_dim),
            _act(),
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
        **kwargs,
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
            _act(),
            BatchNorm1d(hidden_channels),
            Dropout(p_dropout),
            Linear(hidden_channels, hidden_channels),
            _act(),
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


class SparseBinSmoothing(nn.Module):
    def __init__(self, bin_centers, sigma=0.1, max_neighbors=10):
        super().__init__()

        n_bins = len(bin_centers)

        indices = []
        values = []

        for i in range(n_bins):
            distances = torch.abs(bin_centers - bin_centers[i])
            mask = distances < 3 * sigma

            if mask.sum() > max_neighbors:
                _, top_k = torch.topk(-distances, max_neighbors)
                mask = torch.zeros(n_bins, dtype=torch.bool)
                mask[top_k] = True

            weights = torch.exp(-distances[mask].pow(2) / (2 * sigma**2))
            weights = weights / weights.sum()

            for j, w in zip(torch.where(mask)[0], weights):
                indices.append([i, j.item()])
                values.append(w.item())

        indices = torch.tensor(indices).T
        values = torch.tensor(values)
        sparse_kernel = torch.sparse_coo_tensor(indices, values, (n_bins, n_bins))

        self.register_buffer("kernel", sparse_kernel)

    def forward(self, x):
        return torch.sparse.mm(self.kernel, x.T).T


class MoleculeBayesianSetRankModel(MoleculeSetRank):
    def __init__(
        self,
        ligand_input_size: int,
        n_bins: int = 20,
        hidden_channels: int = 512,
        p_dropout: float = 0.0,
        num_heads: int = 8,
        smoothing: bool = True,
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
            n_bins=n_bins, tail_mode="dirac", normalization="minmax"
        )
        self.distribution_encoder = _mlp(
            input_size=self.n_bins,
            hidden_size=hidden_channels,
            output_size=hidden_channels,
            hidden_layers=1,
        )
        self.embed_ligand = _mlp(
            input_size=ligand_input_size,
            hidden_size=hidden_channels,
            output_size=hidden_channels,
            hidden_layers=1,
            dropout=p_dropout,
        )
        self.num_heads = num_heads
        self.default_dist_emb = Parameter(torch.zeros(hidden_channels, device=device))
        self.combine_repr = Sequential(
            Linear(hidden_channels * 2, hidden_channels * 2),
            _act(),
            Linear(hidden_channels * 2, hidden_channels),
        )
        self.set_transformer = SetTransformer(
            hidden_channels=hidden_channels,
            num_heads=self.num_heads,
            ffn_hidden_layers=2,
            num_blocks=8,
            dropout=p_dropout,
        )
        self.output = Sequential(
            Linear(hidden_channels, hidden_channels),
            _act(),
            nn.LayerNorm(hidden_channels),
            Linear(hidden_channels, hidden_channels),
            _act(),
            Linear(hidden_channels, n_bins),
        )
        self.smoother = (
            SparseBinSmoothing(
                self.bin_dist.bucket_centers(), sigma=0.1, max_neighbors=20
            )
            if smoothing
            else nn.Identity(n_bins)
        )
        logger.debug(f"Smoothing module: {self.smoother}")

    def forward(
        self,
        ligand: Tensor,
        _protein: Tensor,  # ignored
        y: Tensor,  # raw regression targets
        sample_mask: Tensor,  # [n_sets, set_size]
        padding_mask: Tensor,  # [n_sets, set_size]
    ) -> Tensor:
        x_ligand = self.embed_ligand(ligand)
        class_labels = self.bin_dist.labels(y)
        dist_onehot = self.bin_dist.dist(class_labels)  # [N, n_bins]
        x_dist = self.distribution_encoder(dist_onehot)
        attn_mask = make_attn_mask(sample_mask, self.num_heads)
        sample_mask = sample_mask.float().unsqueeze(-1)
        x_def_dist = self.default_dist_emb.view(1, 1, -1)
        x_dist = (1 - sample_mask) * x_dist + sample_mask * x_def_dist
        x = self.combine_repr(torch.cat((x_ligand, x_dist), -1))
        h = self.set_transformer(x, attn_mask=attn_mask, key_padding_mask=padding_mask)
        logits = self.output(h)
        return self.smoother(logits)


def make_attn_mask(sample_mask, num_heads):
    n_sets, set_size = sample_mask.shape
    q_mask = sample_mask.unsqueeze(-1)  # [n_sets, set_size, 1]
    k_mask = sample_mask.unsqueeze(-2)  # [n_sets, 1, set_size]

    mask = torch.zeros(n_sets, set_size, set_size, dtype=torch.bool, device=device)

    # non-query cannot attend to query
    mask |= (~q_mask) & k_mask

    # query cannot attend to other queries (except itself)
    same_idx = torch.eye(set_size, dtype=torch.bool, device=device).unsqueeze(0)
    mask |= q_mask & k_mask & (~same_idx)

    mask = mask.unsqueeze(1).expand(-1, num_heads, -1, -1)
    mask = mask.reshape(n_sets * num_heads, set_size, set_size)

    return mask


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
            dropout=0.1,
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
    act=_act,
    dropout: float = 0.0,
) -> Module:
    if hidden_layers == 0:
        return Linear(input_size, output_size)

    layers = [Linear(input_size, hidden_size), act(), Dropout(dropout)]
    for _ in range(hidden_layers - 1):
        layers.extend([Linear(hidden_size, hidden_size), act(), Dropout(dropout)])
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
            _act(),
            Linear(hidden_channels * 3, hidden_channels),
        )

    def forward(
        self,
        ligand: Tensor,
        protein: Tensor,
        y: Tensor,
        sample_mask: Tensor,
        set_ids: Tensor,
    ) -> Tensor:
        attn_mask = make_block_diag_mask(set_ids, num_heads=self.num_heads)
        attn_mask = torch.logical_or(attn_mask, make_asymmetric_mask(sample_mask))
        sample_mask = sample_mask.float().unsqueeze(1)
        x_ligand = self.embed_ligand(ligand)
        x_protein = self.embed_protein(protein)
        class_labels = self.bin_dist.labels(y)
        dist_onehot = self.bin_dist.dist(class_labels)  # [N, n_bins]
        x_dist = self.distribution_encoder(dist_onehot)
        x_dist = (1 - sample_mask) * x_dist + sample_mask * self.default_dist_emb
        x = self.combine_repr(torch.cat((x_ligand, x_protein, x_dist), 1))
        h = self.set_transformer(x, attn_mask=attn_mask)
        logits = self.output(h)
        return self.smoother(logits)


class MAB(Module):
    def __init__(
        self,
        hidden_channels: int,
        num_heads: int,
        ffn_hidden_layers: int,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads
        self.ffn_hidden_layers = ffn_hidden_layers
        self.dropout = dropout
        self.attn = MHA(hidden_channels, num_heads, dropout=dropout, batch_first=True)
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
        key_padding_mask: Tensor | None = None,
        need_weights: bool = False,
    ) -> Tensor:
        if y is None:
            y = x
        x_, attn_weights = self.attn(
            x,
            y,
            y,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=True,
        )
        x = self.ln1(x + x_)
        x = self.ln2(x + self.ffn(x))
        if need_weights:
            return x, attn_weights
        return x


class SAB(MAB):
    def forward(
        self,
        x: Tensor,
        attn_mask: Tensor | None = None,
        key_padding_mask: Tensor | None = None,
        need_weights: bool = False,
    ) -> Tensor:
        return super().forward(
            x,
            x,
            attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
        )


class ISAB(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        num_heads: int,
        ffn_hidden_layers: int,
        dropout: float = 0.1,
        num_inducing_points: int = 10,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads
        self.m = num_inducing_points
        self.ffn_hidden_layers = ffn_hidden_layers
        self.dropout = dropout

        self.ind_points = Parameter(torch.randn(self.m, hidden_channels))

        self.mab1 = MAB(hidden_channels, num_heads, ffn_hidden_layers, dropout=dropout)
        self.mab2 = MAB(hidden_channels, num_heads, ffn_hidden_layers, dropout=dropout)

    def forward(
        self,
        x: Tensor,
        attn_mask: Tensor | None = None,
        key_padding_mask: Tensor | None = None,
        need_weights: bool = False,
    ) -> Tensor:
        if need_weights:
            raise NotImplementedError("Returning weights not supported by ISAB.")

        B = x.size(0)
        I = self.ind_points.expand(B, -1, -1)
        H = self.mab1(
            I,
            x,
            attn_mask=None,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        out = self.mab2(
            x,
            H,
            attn_mask=None,
            key_padding_mask=None,  # K=H has no padding
            need_weights=False,
        )
        return out


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(self.norm(x), *args, **kwargs)


class SetTransformer(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        num_heads: int,
        ffn_hidden_layers: int,
        num_blocks: int,
        num_seeds: int = 1,
        dropout: float = 0.05,
        layer_type: Literal["full", "induced"] = "full",
        prenorm: bool = False,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.num_seeds = num_seeds
        self.dropout = dropout
        self.layer_type = layer_type

        BlockType = SAB if layer_type == "full" else ISAB

        def maybe_norm(block):
            return PreNorm(hidden_channels, block) if prenorm else block

        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            if BlockType == ISAB:
                block = ISAB(
                    hidden_channels,
                    num_heads,
                    ffn_hidden_layers,
                    dropout=dropout,
                    num_inducing_points=16,
                )
            else:
                block = SAB(
                    hidden_channels,
                    num_heads,
                    ffn_hidden_layers,
                    dropout=dropout,
                )
            self.blocks.append(maybe_norm(block))

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
        set_ids: torch.Tensor | None = None,
        need_intermediate_activations: bool = False,
    ) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)

        return x


def make_block_diag_mask(set_ids: torch.Tensor, num_heads: int = None) -> torch.Tensor:
    set_ids = set_ids.flatten()
    mask = set_ids.unsqueeze(0) != set_ids.unsqueeze(1)  # (N, N)

    if num_heads is not None:
        mask = mask.unsqueeze(0).expand(num_heads, -1, -1)  # (num_heads, N, N)

    return mask


def make_asymmetric_mask(masked: torch.Tensor) -> torch.Tensor:
    N = masked.shape[0]
    attn_mask = torch.zeros(N, N, dtype=torch.bool, device=device)

    for i in range(N):  # query index
        attn_mask[i] = masked  # disallow attending masked tokens
        if masked[i]:
            attn_mask[i, i] = False  # allow self

    return attn_mask
