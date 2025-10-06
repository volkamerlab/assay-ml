import torch
from torch.nn import functional as F
from torch import nn

import tqdm.auto as tqdm


from .utils import device

import logging

logger = logging.getLogger(__name__)


class BinDistribution(nn.Module):
    """Bin distribution with empirical quantile bins fitted to z-score normalized data."""

    def __init__(
        self,
        n_bins: int,
        exp_tails: bool = True,
    ):
        super().__init__()
        self.n_bins = n_bins
        self.exp_tails = exp_tails
        self._side_normals = None

        self.register_buffer("edges", torch.zeros(n_bins + 1, device=device))

    @torch.no_grad()
    def fit(self, loader, max_samples: int = 1_000_000):
        """
        Fit quantile bin edges from z-score normalized training data.

        Args:
            loader: Training data loader
            mask_fraction: Fraction of samples to mask (should match training)
        """
        all_normed_values = []

        if hasattr(loader.dataset, "property_set_ratio"):
            prop_set_ratio = loader.dataset.property_set_ratio
            loader.dataset.property_set_ratio = 0

        for protein_features, ligand_features, labels, _, metadata in tqdm.tqdm(
            loader, desc="fitting bin distribution"
        ):
            set_boundaries = metadata["set_boundaries"].squeeze()
            num_sets = metadata["num_sets"].squeeze().item()

            labels = labels.squeeze().to(device)

            # Apply the same z-score normalization as in training
            for i in range(num_sets):
                start_idx = set_boundaries[i]
                end_idx = set_boundaries[i + 1]
                set_size = end_idx - start_idx
                set_labels = labels[start_idx:end_idx]

                mean_val = set_labels.mean()
                std_val = set_labels.std(unbiased=True)
                std_val = std_val.clamp_min(1e-6)
                normed_set = (set_labels - mean_val) / std_val

                all_normed_values.append(normed_set)

        all_normed = torch.cat(all_normed_values)
        if all_normed.numel() > max_samples:
            idx = torch.randperm(all_normed.numel(), device=device)[:max_samples]
            all_normed = all_normed[idx]

        probabilities = torch.linspace(0, 1, self.n_bins + 1, device=device)
        quantiles = torch.quantile(all_normed, probabilities)

        self.edges.copy_(quantiles)
        if self.exp_tails:
            self._init_side_normals()

        logger.info(
            f"fitted {self.n_bins} bins with edges: "
            f"[{self.edges[0]:.3f}, ..., {self.edges[-1]:.3f}]"
        )
        logger.info(
            f"Median bin width: {(self.edges[1:] - self.edges[:-1]).median():.3f}"
        )
        if hasattr(loader.dataset, "property_set_ratio"):
            loader.dataset.property_set_ratio = prop_set_ratio

    def _init_side_normals(self):
        """Initialize half-normal distributions for the tails."""
        bucket_widths = self.edges[1:] - self.edges[:-1]
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
        """Assign bin labels to input values."""
        bucket_indices = torch.searchsorted(self.edges, y) - 1
        bucket_indices[y == self.edges[0]] = 0
        bucket_indices[y == self.edges[-1]] = self.n_bins - 1
        bucket_indices = bucket_indices.clamp(0, self.n_bins - 1)
        return bucket_indices.long()

    def dist(self, class_labels: torch.Tensor) -> torch.Tensor:
        """Convert class labels to one-hot distribution."""
        return F.one_hot(class_labels, num_classes=self.n_bins).float()

    def log_prob(self, y: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        """Compute log probability of values given logits."""
        bucket_indices = self.labels(y)
        bucket_log_probs = F.log_softmax(logits, dim=-1)
        bucket_widths = self.edges[1:] - self.edges[:-1]

        # Scale log probs by bucket width (uniform density within bucket)
        scaled_log_probs = bucket_log_probs - torch.log(bucket_widths)
        log_probs = scaled_log_probs.gather(-1, bucket_indices.unsqueeze(-1)).squeeze(
            -1
        )

        # Add exponential tail corrections if enabled
        if self.exp_tails and self._side_normals is not None:
            left_boundary_mask = bucket_indices == 0
            right_boundary_mask = bucket_indices == self.n_bins - 1

            if left_boundary_mask.any():
                distances = (self.edges[1] - y[left_boundary_mask]).clamp(min=1e-8)
                half_normal_log_prob = self._side_normals[0].log_prob(distances)
                log_probs[left_boundary_mask] += half_normal_log_prob + torch.log(
                    bucket_widths[0]
                )

            if right_boundary_mask.any():
                distances = (y[right_boundary_mask] - self.edges[-2]).clamp(min=1e-8)
                half_normal_log_prob = self._side_normals[1].log_prob(distances)
                log_probs[right_boundary_mask] += half_normal_log_prob + torch.log(
                    bucket_widths[-1]
                )

        return log_probs

    def bucket_centers(self) -> torch.Tensor:
        return (self.edges[:-1] + self.edges[1:]) / 2

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
        loc_left = self.edges[left_index]
        loc_right = self.edges[right_index]
        slope = (q - P_left) / (P_right - P_left).clamp(min=1e-8)
        xq = loc_left + slope * (loc_right - loc_left)
        return xq

    def median(self, logits: torch.Tensor):
        return self.icdf(logits, q=0.5).squeeze(-1)
