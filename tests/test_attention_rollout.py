from typing import Tuple
import torch

from dti.utils.attention_rollout import attention_rollout, get_attention_weights
from dti.model.common import unstack_block_diagonal_tensor
from dti.model.bayesian import MoleculeBayesianSetRankModel


def _mock_model_and_input() -> Tuple[MoleculeBayesianSetRankModel, list, dict]:
    model = MoleculeBayesianSetRankModel(
        ligand_input_size=16,
        n_bins=20,
        hidden_channels=16,
        p_dropout=0.0,
        num_heads=2,
    )
    x_ligand = torch.randn(7, 16)
    y = torch.rand(7)
    sample_mask = ~torch.tensor([True, True, False, True, True, False, False])
    set_ids = torch.tensor([0, 0, 0, 1, 1, 1, 1])
    args = [x_ligand, y, sample_mask, set_ids]
    return model, args, dict()


@torch.no_grad()
def test_compute(
    batch_size: int = 4,
    heads: int = 2,
    tokens: int = 5,
    layers: int = 3,
):
    random_attention_weights = [
        torch.randn(batch_size, heads, tokens, tokens) for _ in range(layers)
    ]
    random_attention_weights = [
        torch.softmax(w, dim=-1) for w in random_attention_weights
    ]

    result = attention_rollout(random_attention_weights, agg=None)
    assert result.shape == random_attention_weights[0].shape

    result = attention_rollout(random_attention_weights, agg="mean")
    assert result.shape == (batch_size, tokens, tokens)


def test_get_attention_scores():
    model, args, kwargs = _mock_model_and_input()
    attention_weights = get_attention_weights(model, args, kwargs)
    assert len(attention_weights) == len(model.set_transformer.blocks)
    return attention_weights


def test_integration():
    model, args, kwargs = _mock_model_and_input()
    x_ligand, y, sample_mask, set_ids = args
    scores = get_attention_weights(model, (x_ligand, y, sample_mask, set_ids), dict())
    unstacked = [unstack_block_diagonal_tensor(score, set_ids) for score in scores]
    num_layers = len(unstacked)
    num_examples = len(unstacked[0])
    attention_weights = [
        [unstacked[j_layer][j_example].unsqueeze_(0) for j_layer in range(num_layers)]
        for j_example in range(num_examples)
    ]
    global_attention = [attention_rollout(att) for att in attention_weights]
    assert len(global_attention) == 2


if __name__ == "__main__":
    test_compute()
    test_get_attention_scores()
    test_integration()
