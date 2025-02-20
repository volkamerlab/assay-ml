from torch import Tensor
from torch.nn import Dropout, Linear, Module, ReLU, Sequential, BatchNorm1d

from dti.set_rank.set_transformer import SetTransformer, _mlp


class SetRankModel(Module):
    def __init__(
        self,
        ligand_input_size: int,
        protein_input_size: int,
        hidden_channels: int,
        p_dropout: float = 0.05,
        **kwargs,
    ):
        super().__init__()
        self.embed_ligand = _mlp(
            input_size=ligand_input_size,
            hidden_size=hidden_channels,
            output_size=hidden_channels,
            hidden_layers=1,
        )
        self.embed_protein = _mlp(
            input_size=protein_input_size,
            hidden_size=hidden_channels,
            output_size=hidden_channels,
            hidden_layers=1,
        )
        self.set_transformer = SetTransformer(
            hidden_channels=hidden_channels,
            num_heads=8,
            ffn_hidden_layers=1,
            num_blocks=3,
            dropout=0.0,
        )
        self.ouput = Sequential(
            Linear(hidden_channels, hidden_channels),
            ReLU(),
            BatchNorm1d(hidden_channels),
            Dropout(p_dropout),
            Linear(hidden_channels, hidden_channels),
            ReLU(),
            BatchNorm1d(hidden_channels),
            Dropout(p_dropout),
            Linear(hidden_channels, 1),
        )

    def combine_with_query(self, x: Tensor, query: Tensor) -> Tensor:
        return x * query

    def forward(self, ligand: Tensor, protein: Tensor) -> Tensor:
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
        h = self.set_transformer(x)
        return self.ouput(h)
