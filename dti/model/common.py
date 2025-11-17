import torch
from torch import Tensor
from torch.nn import (
    Linear,
    Module,
    Sequential,
)
from torch.nn import GELU

from ..utils import device


def mlp(
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


def make_asymmetric_mask(masked: Tensor) -> Tensor:
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


def make_block_diag_mask(set_ids: Tensor, num_heads: int = None) -> Tensor:
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

    return mask.to(device)
