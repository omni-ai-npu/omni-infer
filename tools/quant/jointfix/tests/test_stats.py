# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""
Activation-stat collection — and the opt-out of the per-forward stats output-recon never reads.

Two stats cost the forward but only the weight-error objective consumes them:
  - the histogram  -> p99.9/median -> the "outlier" channel weight
  - sum_x2         -> E[x²]        -> the "hessian" channel weight
output-recon is a pure output MSE and reads neither. These tests pin the contract: skipping
them leaves every *used* statistic (amax, X_sample) bit-identical, and the method skips both
for the default output-recon path.
"""
import torch

from jointfix.core.stats import AccumActStats, StatsConfig, merge_collectors
from jointfix.methods.jointfix import JointSearchConfig


def _feed(cfg, xs):
    c = AccumActStats(xs[0].shape[-1], cfg)
    for x in xs:
        c.update(x)
    return c


def test_skipping_unused_stats_leaves_used_stats_bit_identical():
    """
    amax / X_sample (what output-recon actually uses) are unchanged when the unused
    histogram + moments are skipped.
    """
    torch.manual_seed(0)
    xs = [torch.randn(7, 4) for _ in range(5)]
    full = _feed(StatsConfig(collect_histogram=True, collect_moments=True), xs).finalize(torch.device("cpu"))
    lean = _feed(StatsConfig(collect_histogram=False, collect_moments=False), xs).finalize(torch.device("cpu"))
    assert torch.equal(full["amax"], lean["amax"])
    assert torch.equal(full["X_sample"], lean["X_sample"])


def test_moments_collected_match_and_skipped_are_zero():
    torch.manual_seed(1)
    xs = [torch.randn(7, 4) for _ in range(3)]
    with_m = _feed(StatsConfig(collect_moments=True), xs).finalize(torch.device("cpu"))
    no_m = _feed(StatsConfig(collect_moments=False), xs).finalize(torch.device("cpu"))
    assert torch.count_nonzero(with_m["E_x2"]) > 0          # really collected
    assert torch.count_nonzero(no_m["E_x2"]) == 0           # zero placeholder when skipped
    assert torch.equal(with_m["amax"], no_m["amax"])        # amax untouched either way


def test_skipped_histogram_returns_zero_placeholder_quantiles():
    c = _feed(StatsConfig(collect_histogram=False), [torch.randn(7, 4)])
    assert c.histogram is None and c.bin_edges is None
    out = c.finalize(torch.device("cpu"))
    assert "p99_9" in out and "median" in out
    assert torch.count_nonzero(out["p99_9"]) == 0
    assert torch.count_nonzero(out["median"]) == 0


def test_dead_sum_abs_x_is_removed():
    c = _feed(StatsConfig(), [torch.randn(7, 4)])
    assert not hasattr(c, "sum_abs_x")                      # it was never read — gone


def test_default_statsconfig_still_collects_everything():
    c = _feed(StatsConfig(), [torch.randn(7, 4)])           # default preserves old behavior
    assert c.histogram is not None
    out = c.finalize(torch.device("cpu"))
    assert torch.count_nonzero(out["p99_9"]) > 0
    assert torch.count_nonzero(out["E_x2"]) > 0


def test_merge_collectors_works_without_histogram_or_moments():
    xs = [torch.randn(5, 4)]
    a = {"k": _feed(StatsConfig(collect_histogram=False, collect_moments=False), xs)}
    b = {"k": _feed(StatsConfig(collect_histogram=False, collect_moments=False), xs)}
    out = merge_collectors([a, b])["k"].finalize(torch.device("cpu"))
    assert out["amax"].shape == (4,)


def test_method_skips_unused_stats_for_output_recon():
    # default = output-recon -> neither channel-weight stat is collected
    sc = JointSearchConfig().stats_config()
    assert sc.collect_histogram is False and sc.collect_moments is False
    # outlier weighting under output-recon still skips (output-recon ignores channel weights)
    assert JointSearchConfig(channel_weight="outlier").stats_config().collect_histogram is False
    # only weight-error actually consumes them
    we_out = JointSearchConfig(objective="weight-error", channel_weight="outlier").stats_config()
    we_hes = JointSearchConfig(objective="weight-error", channel_weight="hessian").stats_config()
    assert we_out.collect_histogram is True and we_out.collect_moments is False
    assert we_hes.collect_moments is True and we_hes.collect_histogram is False


def test_mdmixq_samples_are_text_first_and_confidence_ranked():
    cfg = StatsConfig(
        sample_limit=5, modality_aware=True, text_sample_ratio=0.8,
        collect_histogram=False, collect_moments=False,
    )
    c = AccumActStats(1, cfg)
    x = torch.arange(8, dtype=torch.float32).unsqueeze(1)
    is_text = torch.tensor([1, 1, 1, 1, 1, 0, 0, 0], dtype=torch.bool)
    priority = torch.tensor([0.1, 0.9, 0.3, 0.8, 0.2, 0.2, 0.7, 0.5])
    c.update(x, token_is_text=is_text, priority=priority)
    out = c.finalize(torch.device("cpu"))

    assert out["text_count"] == 5 and out["nontext_count"] == 3
    assert out["X_sample"].squeeze(1).tolist() == [1.0, 3.0, 2.0, 4.0, 6.0]
