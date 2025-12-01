from functools import partial
from typing import Literal

import torch
import torchlens
from torch import Tensor

from dti.model.bayesian import MoleculeBayesianSetRankModel
from dti.model.common import unstack_block_diagonal_tensor
import re

_agg_functions = {
    "mean": partial(torch.mean, dim=1),
    "max": partial(torch.max, dim=1),
}


def _identity(x: Tensor) -> Tensor:
    return x


def attention_rollout(
    attention_weights: list[Tensor],  # (B, H, N, N)
    add_identity: bool = False,
    agg: Literal["mean", "max"] | None = "mean",
    agg_when: Literal["first", "last"] = "first",
) -> Tensor:
    """
    Perform attention rollout by iteratively multiplying
    attenion weight matrices from different layers.

    Attention across different heads is aggregated by default.
    Aggregation can be suppressed by passing `agg=None`,

    Args:
        attention_weights (list[Tensor]): list of attention weights across model layers.
        agg (Literal[&quot;mean&quot;, &quot;max&quot;] | None, optional): How to aggregate (mean, max, or None). Defaults to "mean".
        agg_when (Literal[&quot;first&quot;, &quot;last&quot;], optional): When to aggregate (first (before rollout) or last (after rollout). Defaults to "first".

    Returns:
        Tensor: attention rollout matrix.
    """
    B, H, N, N_ = attention_weights[0].shape
    assert N == N_, "Attenion weight matrices should be square!"
    fn_agg = _identity
    if agg in _agg_functions:
        fn_agg = _agg_functions[agg]
    if add_identity:
        eye = torch.eye(N, N).view(1, 1, N, N)
        attention_weights = [0.5 * w + 0.5 * eye for w in attention_weights]
    if agg_when == "first":
        attention_weights = [fn_agg(w) for w in attention_weights]
    result = attention_weights[0].clone()
    for w in attention_weights[1:]:
        result = torch.matmul(w, result)
    if agg_when == "last":
        result = fn_agg(result)
    return result


def get_attention_weights(
    model: MoleculeBayesianSetRankModel,
    model_fwd_args: list,
    model_fwd_kwargs: dict,
    attn_layer_pattern: str = "set_transformer.blocks.{}.attn",
):
    _stored = model.set_transformer._need_weights
    model.set_transformer.need_weights(True)
    history = torchlens.log_forward_pass(model, model_fwd_args, model_fwd_kwargs)
    model.set_transformer.need_weights(_stored)
    return [
        history[attn_layer_pattern.format(j)].tensor_contents
        for j in range(len(model.set_transformer.blocks))
    ]


if __name__ == "__main__":
    # x = torch.randn(2, 2, 2, 3, 3)
    # y = torch.randn(2, 2, 2, 3, 3)
    # xy = torch.matmul(x, y)
    # assert torch.allclose(xy[0, 1, 0], torch.matmul(x[0, 1, 0], y[0, 1, 0]))

    model = MoleculeBayesianSetRankModel(
        ligand_input_size=16,
        n_bins=20,
        hidden_channels=16,
        p_dropout=0.0,
        num_heads=2,
    )
    x_ligand = torch.randn(7, 16)
    y = torch.rand(7)
    sample_mask = torch.tensor([True, True, False, True, True, False, False])
    set_ids = torch.tensor([0, 0, 0, 1, 1, 1, 1])

    scores = get_attention_weights(model, (x_ligand, y, sample_mask, set_ids), dict())
    unstacked = [unstack_block_diagonal_tensor(score, set_ids) for score in scores]
    for block in blocks:
        rollout_scores = attention_rollout(block.unsqueeze(0))
