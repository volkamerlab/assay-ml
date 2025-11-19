from functools import cached_property

import torch
from torch.nn import functional as F
from torch import nn
import tqdm.auto as tqdm
import numpy as np
import logging
from ..utils import device

logger = logging.getLogger(__name__)


class BinDistribution(nn.Module):
    def __init__(
        self,
        n_bins: int,
        tail_type: str = "exponential",
        tail_bound_prob: float | None = None,
        normalization: str = "zscore",
    ):
        super().__init__()
        self.n_bins = n_bins

        assert tail_type in ["none", "exponential", "half_normal"]
        self.tail_type = tail_type

        if tail_type != "none":
            if tail_bound_prob is None:
                self.tail_bound_prob = 2 / n_bins
            else:
                self.tail_bound_prob = tail_bound_prob
        else:
            self.tail_bound_prob = 0.0

        self.device_str = device
        assert normalization in ["zscore", "minmax"]
        self.normalization = normalization

        self.register_buffer("edges", torch.linspace(0, 1, n_bins + 1, device=device))

        self.register_buffer("tail_scales", torch.tensor([1.0, 1.0], device=device))

        logger.info(
            f"BinDistribution initialized: {n_bins} bins, "
            f"tail_type='{self.tail_type}', tail_prob={self.tail_bound_prob}"
        )

    @cached_property
    def bounded_support(self) -> bool:
        return self.tail_type == "none"

    @torch.no_grad()
    def fit(self, loader, max_samples: int = 1_000_000):
        if self.normalization == "minmax":
            return
        assert self.normalization == "zscore"

        all_normed_values = []

        prop_set_ratio = None
        if hasattr(loader.dataset, "property_set_ratio"):
            prop_set_ratio = loader.dataset.property_set_ratio
            loader.dataset.property_set_ratio = 0

        for ligand_features, labels, _, _, metadata in tqdm.tqdm(
            loader, desc="fitting bin distribution"
        ):
            set_boundaries = metadata["set_boundaries"].squeeze()
            num_sets = metadata["num_sets"]
            if isinstance(num_sets, torch.Tensor):
                num_sets = num_sets.squeeze().item()

            real_assay = metadata["real_assay"].squeeze()
            labels = labels.squeeze()[real_assay].to(device)
            set_ids = metadata["set_ids_tensor"].to(device).squeeze()[real_assay]

            num_sets_tensor = set_ids.max().item() + 1

            sum_per_set = torch.zeros(num_sets_tensor, device=device).scatter_add_(
                0, set_ids, labels
            )
            count_per_set = torch.bincount(set_ids, minlength=num_sets_tensor).to(
                device
            )
            mean_per_set = sum_per_set / count_per_set.clamp_min(1)

            var_per_set = torch.zeros(num_sets_tensor, device=device).scatter_add_(
                0, set_ids, (labels - mean_per_set[set_ids]) ** 2
            )
            std_per_set = (
                (var_per_set / count_per_set.clamp_min(1).sub(1)).sqrt().clamp_min(1e-6)
            )

            normed_labels = (labels - mean_per_set[set_ids]) / std_per_set[set_ids]
            all_normed_values.append(normed_labels)

        all_normed = torch.cat(all_normed_values)
        all_normed = all_normed[torch.isfinite(all_normed)]

        if all_normed.numel() > max_samples:
            idx = torch.randperm(all_normed.numel())[:max_samples]
            all_normed = all_normed[idx]

        p_lower = self.tail_bound_prob
        p_upper = 1.0 - self.tail_bound_prob

        probabilities = torch.linspace(p_lower, p_upper, self.n_bins + 1).to(device)
        quantiles = torch.quantile(all_normed, probabilities).to(device)
        self.edges.copy_(quantiles)

        if self.tail_type != "none":
            left_boundary = self.edges[0]
            right_boundary = self.edges[-1]

            left_tail_data = all_normed[all_normed < left_boundary]
            right_tail_data = all_normed[all_normed > right_boundary]

            self._fit_single_tail(left_tail_data, left_boundary, is_left=True)
            self._fit_single_tail(right_tail_data, right_boundary, is_left=False)

        logger.info(
            f"Fitted {self.n_bins} bins with edges: "
            f"[{self.edges[0]:.3f}, ..., {self.edges[-1]:.3f}]"
        )
        if self.tail_type != "none":
            logger.info(
                f"Tail Scales ({self.tail_type}): Left={self.tail_scales[0]:.3f}, "
                f"Right={self.tail_scales[1]:.3f}"
            )

        if prop_set_ratio is not None and hasattr(loader.dataset, "property_set_ratio"):
            loader.dataset.property_set_ratio = prop_set_ratio

    def _fit_single_tail(self, data: torch.Tensor, boundary: float, is_left: bool):
        idx = 0 if is_left else 1

        if data.numel() < 2:
            fallback_width = (
                (self.edges[1] - self.edges[0])
                if is_left
                else (self.edges[-1] - self.edges[-2])
            )
            self.tail_scales[idx] = fallback_width
            logger.warning(
                f"{'Left' if is_left else 'Right'} tail has insufficient data. "
                f"Using fallback scale: {fallback_width:.3f}"
            )
            return

        distances = (boundary - data) if is_left else (data - boundary)
        distances = distances.clamp_min(1e-6)

        if self.tail_type == "exponential":
            scale = distances.mean()
        elif self.tail_type == "half_normal":
            scale = torch.sqrt(distances.pow(2).mean())
        else:
            scale = torch.tensor(1.0, device=device)

        self.tail_scales[idx] = scale

    @property
    def widths(self):
        return self.edges[1:] - self.edges[:-1]

    @property
    def effective_widths(self):
        w = self.widths.clone()
        if self.tail_type != "none":
            w[0] = self.tail_scales[0]
            w[-1] = self.tail_scales[1]
        return w

    def labels(self, y: torch.Tensor) -> torch.Tensor:
        bucket_indices = torch.searchsorted(self.edges, y) - 1
        bucket_indices[y == self.edges[0]] = 0
        bucket_indices[y == self.edges[-1]] = self.n_bins - 1
        return bucket_indices.clamp(0, self.n_bins - 1).long()

    def dist(self, class_labels: torch.Tensor) -> torch.Tensor:
        return F.one_hot(class_labels, num_classes=self.n_bins).float()

    def log_prob(self, y: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        bucket_idcs = self.labels(y)
        bucket_log_ps = F.log_softmax(logits, dim=-1)

        scaled_log_ps = bucket_log_ps - torch.log(self.effective_widths.clamp_min(1e-8))
        log_ps = scaled_log_ps.gather(-1, bucket_idcs.unsqueeze(-1)).squeeze(-1)

        if self.tail_type == "none":
            return log_ps

        left_mask = bucket_idcs == 0
        right_mask = bucket_idcs == self.n_bins - 1

        if left_mask.any():
            y_sub = y[left_mask]
            dist = (self.edges[0] - y_sub).clamp(min=0.0)
            scale = self.tail_scales[0]

            log_p_bin = bucket_log_ps[left_mask, 0]

            if self.tail_type == "exponential":
                tail_log_pdf = -torch.log(scale + 1e-8) - (dist / scale)
            elif self.tail_type == "half_normal":
                const = 0.5 * np.log(2 / np.pi)
                tail_log_pdf = (
                    const - torch.log(scale + 1e-8) - 0.5 * (dist / scale).pow(2)
                )

            log_ps[left_mask] = log_p_bin + tail_log_pdf

        if right_mask.any():
            y_sub = y[right_mask]
            dist = (y_sub - self.edges[-1]).clamp(min=0.0)
            scale = self.tail_scales[1]

            log_p_bin = bucket_log_ps[right_mask, -1]

            if self.tail_type == "exponential":
                tail_log_pdf = -torch.log(scale + 1e-8) - (dist / scale)
            elif self.tail_type == "half_normal":
                const = 0.5 * np.log(2 / np.pi)
                tail_log_pdf = (
                    const - torch.log(scale + 1e-8) - 0.5 * (dist / scale).pow(2)
                )

            log_ps[right_mask] = log_p_bin + tail_log_pdf

        return log_ps

    def wasserstein(self, y: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=-1)
        true_bins = self.labels(y)
        one_hot = F.one_hot(true_bins, num_classes=self.n_bins).float()

        cdf_pred = torch.cumsum(probs, dim=-1)
        cdf_true = torch.cumsum(one_hot, dim=-1)

        wass = torch.sum(torch.abs(cdf_pred - cdf_true) * self.effective_widths, dim=-1)

        total_width = self.edges[-1] - self.edges[0]
        if self.tail_type != "none":
            total_width = total_width + self.tail_scales.sum()

        wass = wass / total_width.clamp_min(1e-6)
        return wass

    def crps(self, y: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=-1)
        true_bins = self.labels(y)
        one_hot = F.one_hot(true_bins, num_classes=self.n_bins).float()

        cdf_pred = torch.cumsum(probs, dim=-1)
        cdf_true = torch.cumsum(one_hot, dim=-1)

        crps = torch.sum((cdf_pred - cdf_true).pow(2) * self.effective_widths, dim=-1)
        return crps

    def bucket_centers(self) -> torch.Tensor:
        centers = (self.edges[:-1] + self.edges[1:]) / 2

        if self.tail_type == "none":
            return centers

        if self.tail_type == "exponential":
            left_offset = self.tail_scales[0]
            right_offset = self.tail_scales[1]
        elif self.tail_type == "half_normal":
            factor = np.sqrt(2 / np.pi)
            left_offset = self.tail_scales[0] * factor
            right_offset = self.tail_scales[1] * factor

        centers[0] = self.edges[0] - left_offset
        centers[-1] = self.edges[-1] + right_offset

        return centers

    def moment(self, logits: torch.Tensor, n: float = 1.0):
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
        assert q.dim() == 1
        assert logits.dim() == 2
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

        loc_left = self.edges[left_index.clamp(max=self.n_bins - 1)]
        loc_right = self.edges[right_index.clamp(max=self.n_bins)]

        slope = (q - P_left) / (P_right - P_left).clamp(min=1e-8)
        xq = loc_left + slope * (loc_right - loc_left)
        return xq

    def median(self, logits: torch.Tensor):
        return self.icdf(logits, q=0.5).squeeze(-1)
