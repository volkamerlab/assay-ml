import torch
from torch.nn import functional as F
from torch import nn


from .utils import device

import logging

logger = logging.getLogger(__name__)


class BinDistribution(nn.Module):
    """Learnable bin distribution with positive interval widths."""

    def __init__(
        self,
        n_bins: int,
        exp_tails: bool = True,
        widths: torch.Tensor | None = None,
        learnable_widths: bool = False,
    ):
        super().__init__()
        self.n_bins = n_bins
        self.exp_tails = exp_tails

        self.widths_unconstrained = (
            torch.ones(n_bins, device=device) if widths is None else widths
        )
        if learnable_widths:
            self.widths_unconstrained = nn.Parameter(self.widths_unconstrained)

        self._side_normals = None

    def _construct_edges(self) -> torch.Tensor:
        """Reconstruct bin edges from parameters."""
        widths = F.softplus(self.widths_unconstrained)
        edges = torch.cat(
            [
                torch.tensor([0.0], device=device),
                torch.cumsum(widths, dim=0),
            ]
        )
        return edges / widths.sum()

    def _init_side_normals(self):
        edges = self._construct_edges()
        bucket_widths = edges[1:] - edges[:-1]

        self._side_normals = (
            self._halfnormal(bucket_widths[0].item(), p=0.5),
            self._halfnormal(bucket_widths[-1].item(), p=0.5),
        )

    @staticmethod
    def _halfnormal(
        range_max: float, p: float = 0.5
    ) -> torch.distributions.Distribution:
        if range_max <= 0:
            range_max = 1e-8
        standard_half_normal = torch.distributions.HalfNormal(
            torch.tensor(1.0, device=device)
        )
        scale = range_max / standard_half_normal.icdf(torch.tensor(p, device=device))
        return torch.distributions.HalfNormal(scale.item())

    def labels(self, y: torch.Tensor) -> torch.Tensor:
        edges = self._construct_edges()

        bucket_indices = torch.searchsorted(edges, y) - 1
        bucket_indices[y == edges[0]] = 0
        bucket_indices[y == edges[-1]] = self.n_bins - 1
        bucket_indices = bucket_indices.clamp(0, self.n_bins - 1)

        return bucket_indices.long()

    def dist(self, class_labels: torch.Tensor) -> torch.Tensor:
        return F.one_hot(class_labels, num_classes=self.n_bins).float()

    def log_prob(self, y: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        edges = self._construct_edges()
        bucket_indices = self.labels(y)
        bucket_log_probs = F.log_softmax(logits, dim=-1)
        bucket_widths = edges[1:] - edges[:-1]

        scaled_log_probs = bucket_log_probs - torch.log(bucket_widths)
        log_probs = scaled_log_probs.gather(-1, bucket_indices.unsqueeze(-1)).squeeze(
            -1
        )

        if self.exp_tails and self._side_normals is not None:
            left_boundary_mask = bucket_indices == 0
            right_boundary_mask = bucket_indices == self.n_bins - 1

            if left_boundary_mask.any():
                distances = (edges[1] - y[left_boundary_mask]).clamp(min=1e-8)
                half_normal_log_prob = self._side_normals[0].log_prob(distances)
                log_probs[left_boundary_mask] += half_normal_log_prob + torch.log(
                    bucket_widths[0]
                )

            if right_boundary_mask.any():
                distances = (y[right_boundary_mask] - edges[-2]).clamp(min=1e-8)
                half_normal_log_prob = self._side_normals[1].log_prob(distances)
                log_probs[right_boundary_mask] += half_normal_log_prob + torch.log(
                    bucket_widths[-1]
                )

        return log_probs

    def bucket_centers(self) -> torch.Tensor:
        edges = self._construct_edges()
        return (edges[:-1] + edges[1:]) / 2

    def moment(self, logits: torch.Tensor, n: float = 1.0):
        """Compute the n-th moment of the distribution."""
        loc = self.bucket_centers()
        probs = F.softmax(logits, dim=-1)
        return (probs * (loc.pow(n))).sum(dim=-1)

    def mean(self, logits: torch.Tensor):
        return self.moment(logits, n=1.0)

    def var(self, logits: torch.Tensor):
        mu = self.mean(logits)
        mu2 = self.moment(logits, n=2.0)
        return mu2 - mu**2

    def std(self, logits: torch.Tensor):
        return self.var(logits).sqrt()

    def icdf(self, logits: torch.Tensor, q: torch.Tensor | float):
        if isinstance(q, float):
            q = torch.tensor([q])
        q = q.clamp(1e-6, 1 - 1e-6)
        assert q.dim() == 1, "Expected 1D tensor of quantiles"
        assert logits.dim() == 2, "Expected 2D tensor of logits"
        q = q.to(logits.device)

        loc = self._construct_edges()
        probs = F.softmax(logits, dim=-1)
        cumprobs = torch.cumsum(probs, dim=-1)
        bin_indices = torch.searchsorted(
            cumprobs, q.expand(logits.size(0), -1), right=True
        )
        left_index = bin_indices
        right_index = bin_indices + 1
        zero_padded_cumprobs = torch.cat(
            (cumprobs.new_zeros(cumprobs.size(0), 1), cumprobs), dim=-1
        )
        P_left = zero_padded_cumprobs.gather(-1, left_index)
        P_right = zero_padded_cumprobs.gather(-1, right_index)
        loc_left = loc[left_index]
        loc_right = loc[right_index]
        slope = (q - P_left) / (P_right - P_left).clamp(min=1e-8)
        xq = loc_left + slope * (loc_right - loc_left)
        return xq

    def median(self, logits: torch.Tensor):
        return self.icdf(logits, q=0.5).squeeze(-1)
