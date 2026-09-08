# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""
Per-channel activation statistics — for activation-aware methods.

`AccumActStats` collects amax / E[x^2] / per-channel histogram quantiles / a
capped sample cache during the calibration forward. Any activation-aware method
(jointfix, awq, gptq) consumes these.

Decoupling note: this takes a small `StatsConfig` (collection knobs only), NOT a
method's full config object. The smooth method maps its JointSearchConfig down to
a StatsConfig — so `core` never imports `methods`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch


@dataclass
class StatsConfig:
    """Knobs governing activation-stat collection (method-agnostic)."""
    n_hist_bins: int = 256
    hist_min: float = 1e-8
    hist_max: float = 1e4
    sample_limit: int = 128
    tokens_per_sample: int = 8
    # The per-channel histogram only feeds p99.9/median, which only the "outlier"
    # channel weight reads. The output-recon objective (and hessian weighting) never
    # touch them, yet building it moves the WHOLE activation to host + does a float64
    # CPU scatter_add every forward — the dominant cost of stat collection on big MoE
    # layers. Off-by-default callers keep it; the method turns it off when unneeded.
    collect_histogram: bool = True
    # E[x²] (sum_x2) feeds only the "hessian" channel weight — again weight-error-only,
    # unused by output-recon. Its per-forward float64 .double() + .cpu() sync is the rest
    # of the stat-collection cost once the histogram is gone, so it's gated the same way.
    collect_moments: bool = True
    # MDMixQ mode keeps modality-separated candidate samples.  The public
    # statistics remain compatible with JointFix; only X_sample is assembled
    # with a text-first budget at finalize time.
    modality_aware: bool = False
    text_sample_ratio: float = 0.8
    modality_candidate_multiplier: int = 4


class AccumActStats:
    """Per-channel activation statistics: amax, E[X^2], histogram, sample cache."""

    def __init__(self, in_features: int, config: StatsConfig):
        self.in_features = in_features
        self.config = config
        # Allocate the histogram (and its bin edges) only when it will be collected —
        # [in_features, n_bins] float64 per site is GBs of dead CPU RAM across the
        # hundreds of per-expert collectors of a big MoE layer otherwise.
        if config.collect_histogram:
            self.bin_edges = torch.logspace(
                torch.log10(torch.tensor(config.hist_min)).item(),
                torch.log10(torch.tensor(config.hist_max)).item(),
                steps=config.n_hist_bins + 1,
            )
            self.histogram = torch.zeros(in_features, config.n_hist_bins, dtype=torch.float64)
        else:
            self.bin_edges = None
            self.histogram = None
        self.sum_x2 = torch.zeros(in_features, dtype=torch.float64)  # stays 0 if !collect_moments
        self.amax = torch.zeros(in_features, dtype=torch.float32)
        self.count = 0
        self.samples: List[torch.Tensor] = []
        self._sample_rows = 0
        self.text_samples: List[torch.Tensor] = []
        self.nontext_samples: List[torch.Tensor] = []
        self.text_priorities: List[torch.Tensor] = []
        self.nontext_priorities: List[torch.Tensor] = []
        self.text_count = 0
        self.nontext_count = 0

    def update(self, x: torch.Tensor, token_is_text: torch.Tensor | None = None,
               priority: torch.Tensor | None = None) -> None:
        """Accumulate statistics from activation tensor x."""
        if x.ndim == 3:
            x = x.reshape(-1, x.shape[-1])
        elif x.ndim != 2:
            raise ValueError(f"Expected 2D or 3D input, got shape {x.shape}")

        x = x.detach().float()
        abs_x = x.abs()

        # E[x²] (only the hessian channel weight reads it) — skip the float64 materialize
        # + host sync when unneeded. amax is always needed (it sets the smooth scale).
        if self.config.collect_moments:
            self.sum_x2 += (x.double() ** 2).sum(dim=0).cpu()
        self.amax = torch.maximum(self.amax, abs_x.amax(dim=0).cpu())
        self.count += x.shape[0]

        if token_is_text is not None:
            token_is_text = token_is_text.detach().reshape(-1).to(dtype=torch.bool)
            if token_is_text.numel() != x.shape[0]:
                raise ValueError(
                    f"token modality rows {token_is_text.numel()} != activation rows {x.shape[0]}"
                )
            self.text_count += int(token_is_text.sum().item())
            self.nontext_count += int((~token_is_text).sum().item())

        # Histogram (skipped unless an outlier-style channel weight needs the quantiles)
        if self.histogram is not None:
            bin_idx = torch.bucketize(abs_x.cpu(), self.bin_edges) - 1
            bin_idx = bin_idx.clamp(0, self.config.n_hist_bins - 1)
            flat_idx = bin_idx.transpose(0, 1).contiguous()
            ones = torch.ones_like(flat_idx, dtype=torch.float64)
            self.histogram.scatter_add_(1, flat_idx, ones)

        # Sample cache for J(a,b) computation
        if self.config.modality_aware and token_is_text is not None:
            scores = (priority.detach().reshape(-1).float().cpu()
                      if priority is not None else torch.ones(x.shape[0]))
            if scores.numel() != x.shape[0]:
                raise ValueError(
                    f"priority rows {scores.numel()} != activation rows {x.shape[0]}"
                )
            mask_cpu = token_is_text.cpu()
            x_cpu = x.cpu()
            cap = max(self.config.sample_limit,
                      self.config.sample_limit * self.config.modality_candidate_multiplier)
            for mask, values, priorities in (
                (mask_cpu, self.text_samples, self.text_priorities),
                (~mask_cpu, self.nontext_samples, self.nontext_priorities),
            ):
                selected = torch.nonzero(mask, as_tuple=False).flatten()
                if selected.numel() == 0:
                    continue
                values.append(x_cpu.index_select(0, selected))
                priorities.append(scores.index_select(0, selected))
                # Bound memory while retaining the highest-confidence routed
                # candidates. Stable sorting makes runs deterministic.
                rows = sum(v.shape[0] for v in values)
                if rows > cap * 2:
                    all_x = torch.cat(values, dim=0)
                    all_p = torch.cat(priorities, dim=0)
                    keep = torch.argsort(all_p, descending=True, stable=True)[:cap]
                    values[:] = [all_x.index_select(0, keep)]
                    priorities[:] = [all_p.index_select(0, keep)]
        elif self._sample_rows < self.config.sample_limit:
            take = min(self.config.tokens_per_sample, x.shape[0],
                       self.config.sample_limit - self._sample_rows)
            self.samples.append(x[:take].cpu())
            self._sample_rows += take

    @staticmethod
    def _top_rows(values: List[torch.Tensor], priorities: List[torch.Tensor],
                  limit: int, in_features: int) -> torch.Tensor:
        if limit <= 0 or not values:
            return torch.zeros(0, in_features)
        x = torch.cat(values, dim=0)
        p = torch.cat(priorities, dim=0)
        keep = torch.argsort(p, descending=True, stable=True)[:limit]
        return x.index_select(0, keep)

    def _finalize_samples(self) -> torch.Tensor:
        if not self.config.modality_aware or not (self.text_samples or self.nontext_samples):
            return (torch.cat(self.samples, dim=0) if self.samples
                    else torch.zeros(1, self.in_features))
        limit = self.config.sample_limit
        text_budget = int(round(limit * self.config.text_sample_ratio))
        nontext_budget = limit - text_budget
        text = self._top_rows(
            self.text_samples, self.text_priorities, text_budget, self.in_features)
        nontext = self._top_rows(
            self.nontext_samples, self.nontext_priorities, nontext_budget, self.in_features)
        remaining = limit - text.shape[0] - nontext.shape[0]
        if remaining > 0:
            if text.shape[0] < text_budget:
                extra = self._top_rows(
                    self.nontext_samples, self.nontext_priorities,
                    nontext.shape[0] + remaining, self.in_features)[nontext.shape[0]:]
                nontext = torch.cat((nontext, extra), dim=0)
            else:
                extra = self._top_rows(
                    self.text_samples, self.text_priorities,
                    text.shape[0] + remaining, self.in_features)[text.shape[0]:]
                text = torch.cat((text, extra), dim=0)
        result = torch.cat((text, nontext), dim=0)
        return result if result.shape[0] else torch.zeros(1, self.in_features)

    def _quantile(self, q: float) -> torch.Tensor:
        """Per-channel quantile via histogram CDF interpolation."""
        total = self.histogram.sum(dim=1, keepdim=True).clamp(min=1.0)
        cdf = torch.cumsum(self.histogram, dim=1) / total
        target = torch.full((self.in_features, 1), q, dtype=cdf.dtype)
        bin_idx = (cdf >= target).float().argmax(dim=1)
        edges_lo = self.bin_edges[bin_idx]
        edges_hi = self.bin_edges[bin_idx + 1]
        return (edges_lo * edges_hi).sqrt()

    def finalize(self, device: torch.device) -> dict:
        """Finalize statistics and move to target device."""
        E_x2 = (self.sum_x2 / max(self.count, 1)).float().to(device)
        if self.histogram is not None:
            p99_9 = self._quantile(0.999).to(device)
            median = self._quantile(0.5).to(device)
        else:
            # histogram skipped — p99.9/median feed only the 'outlier' channel weight,
            # which the output-recon objective never reads. Return zero placeholders so
            # dict consumers (e.g. refresh_stats_after_smooth) still work.
            p99_9 = torch.zeros(self.in_features, device=device)
            median = torch.zeros(self.in_features, device=device)
        return {
            "amax": self.amax.to(device),
            "E_x2": E_x2,
            "p99_9": p99_9,
            "median": median,
            "X_sample": self._finalize_samples().to(device),
            "text_count": self.text_count,
            "nontext_count": self.nontext_count,
            "Y_refs": [],   # populated by caller when objective=output-recon
        }


def _merge_into(base: AccumActStats, other: AccumActStats) -> None:
    """
    Merge `other`'s accumulators into `base` (multi-device collector merge).

    sum accumulators, max the amax, sum the histogram/count, append samples up to
    the limit — order-faithful so multi-device stats are identical to a single
    run with the same device count + chunking.
    """
    base.sum_x2 += other.sum_x2
    base.amax = torch.maximum(base.amax, other.amax)
    if base.histogram is not None and other.histogram is not None:
        base.histogram += other.histogram
    base.count += other.count
    base.text_count += other.text_count
    base.nontext_count += other.nontext_count
    base.text_samples.extend(other.text_samples)
    base.nontext_samples.extend(other.nontext_samples)
    base.text_priorities.extend(other.text_priorities)
    base.nontext_priorities.extend(other.nontext_priorities)
    if base.config.modality_aware:
        cap = max(base.config.sample_limit,
                  base.config.sample_limit * base.config.modality_candidate_multiplier)
        for values, priorities in (
            (base.text_samples, base.text_priorities),
            (base.nontext_samples, base.nontext_priorities),
        ):
            rows = sum(v.shape[0] for v in values)
            if rows > cap:
                all_x = torch.cat(values, dim=0)
                all_p = torch.cat(priorities, dim=0)
                keep = torch.argsort(all_p, descending=True, stable=True)[:cap]
                values[:] = [all_x.index_select(0, keep)]
                priorities[:] = [all_p.index_select(0, keep)]
    if base._sample_rows < base.config.sample_limit:
        for s in other.samples:
            if base._sample_rows >= base.config.sample_limit:
                break
            base.samples.append(s)
            base._sample_rows += s.shape[0]


def merge_collectors(parts: "list[dict]") -> dict:
    """
    Merge a list of per-device collector dicts into one. parts[0] is the base
    for each key; the rest merge in device order.
    """
    merged: dict = {}
    keys = set()
    for cp in parts:
        keys.update(cp)
    for key in keys:
        cs = [cp[key] for cp in parts if key in cp]
        base = cs[0]
        for other in cs[1:]:
            if isinstance(base, AccumActStats):
                _merge_into(base, other)
            elif hasattr(base, "merge"):
                base.merge(other)
            else:
                raise TypeError(f"collector {key!r} does not support merge")
        merged[key] = base
    return merged


def refresh_stats_after_smooth(stats: dict, divide_by: torch.Tensor) -> dict:
    """
    Analytically rescale per-channel stats after smoothing applied scale `s` to
    the upstream weight, so the activation reaching this Linear is x_new = x / s.

    Mathematically equivalent to re-running the calibration forward and
    re-collecting stats — but for free, because float smoothing is identity in the
    layer forward AND the activation amax / E[x^2] / quantile / sample all scale
    linearly with s. Used between iterations of coordinate-descent search.

    `divide_by` shape must be [in_features], same as stats arrays.
    """
    s = divide_by.to(stats["amax"].device).clamp(min=1e-12)
    s_sq = s ** 2
    return {
        "amax": stats["amax"] / s,
        "E_x2": stats["E_x2"] / s_sq,
        "p99_9": stats["p99_9"] / s,
        "median": stats["median"] / s,
        "X_sample": stats["X_sample"] / s.unsqueeze(0),
    }
