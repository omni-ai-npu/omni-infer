# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Tests for V2 runner wiring and sampling fallback."""

from __future__ import annotations

import os
import subprocess
import sys
from types import SimpleNamespace

import pytest


@pytest.fixture
def clean_cuda_aliases():
    """torch.cuda 会被永久改写成 torch.npu，退出时还原现场。"""
    from omni_npu.worker.npu.utils import restore_cuda_attrs, snapshot_cuda_attrs

    saved = snapshot_cuda_attrs()
    try:
        yield
    finally:
        restore_cuda_attrs(saved)


# 类体里除了我们定义的名字，Python 还会放这些；3.13 起多了后两个。
_CLASS_NOISE = {
    "__module__", "__qualname__", "__doc__", "__dict__", "__weakref__",
    "__firstlineno__", "__static_attributes__",
}


def test_runner_installs_cuda_aliases_and_pins_override_surface(
    clean_cuda_aliases, monkeypatch
):
    """构造 = 装 torch.cuda 别名 + 零覆写子类（vllm_config/device 原样透传）。"""
    import torch
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    from omni_npu.compilation import npugraph_ex_config
    from omni_npu.worker.npu.model_runner import NPUModelRunnerV2
    from omni_npu.worker.npu import utils as npu_utils

    created = {}

    # 类是模块级声明的，基类换不掉，改钉住上游 __init__ 收到的实参。
    def fake_init(self, vllm_config, device):
        created.update(vllm_config=vllm_config, device=device)

    monkeypatch.setattr(GPUModelRunner, "__init__", fake_init)
    monkeypatch.setattr(npugraph_ex_config, "init_aclgraph_config", lambda cfg: None)
    monkeypatch.setattr(
        npu_utils.logger,
        "info_once",
        lambda *args, **kwargs: None,
        raising=False,
    )

    cfg, dev = object(), object()
    runner = NPUModelRunnerV2(cfg, dev)

    # 别名必须在上游 __init__ 之前装好（上游第三行就建 torch.cuda.Stream）；
    # 模块 patch 不在这里，由插件期的 patch_mrv2_*.py 负责。
    assert torch.cuda.Stream is torch.npu.Stream

    # 上游基类的子类实例，参数原样透传
    assert isinstance(runner, GPUModelRunner)
    assert created == {"vllm_config": cfg, "device": dev}

    # 钉住覆写面：多出来说明有人加了覆写没说明原因，少了说明某个适配被删掉了。
    own = set(vars(type(runner))) - _CLASS_NOISE
    assert own == {
        "__init__",                  # torch.cuda 别名，必须早于上游 __init__ 建 Stream
        "capture_model",             # 裁剪守卫（内存 API 已由 !2496 接管）
        "prepare_inputs",            # 为 prepare_attn 暂存本步 cudagraph 模式
        "prepare_attn",              # MRv1 的 pad_attn 开关：非 FULL 不 pad slot
        "prepare_dummy_attn",        # dummy forward 清零 block table，别写脏已交接的块
        "_omni_has_separate_kv_update",
        "_sync_dummy_main_compute_logits",
        "execute_model",             # 在 dummy target forward 后补 lm_head logits
    }, f"NPUModelRunnerV2 覆写面变化，实际定义了: {own}"


def test_runner_is_the_only_mrv2_entry_class():
    """MRv2 只暴露显式顶层 NPUModelRunnerV2，不保留旧别名或类工厂。"""
    from omni_npu.worker.npu import model_runner
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    assert issubclass(model_runner.NPUModelRunnerV2, GPUModelRunner)
    assert model_runner.NPUModelRunnerV2.__qualname__ == "NPUModelRunnerV2"
    # 没有类工厂，也不额外暴露 NPUModelRunner —— 那个名字属于 MRv1
    # 的 omni_npu.worker.npu_model_runner，同名会让 import 来源变得含糊。
    assert not hasattr(model_runner, "_get_runner_cls")
    assert not hasattr(model_runner, "NPUModelRunner")


def test_dummy_run_joins_dp_lmhead_before_speculator(monkeypatch):
    """Idle ranks must join target lm_head collectives before drafter sync."""
    import torch
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    from omni_npu.worker.npu import model_runner as npu_model_runner

    calls = []
    hidden_states = torch.arange(12).reshape(4, 3)
    logits_indices = torch.tensor([1, 3])

    monkeypatch.setattr(npu_model_runner, "_dp_lmhead_enabled", lambda: True)

    def fake_super_execute_model(self, *args, **kwargs):
        calls.append("execute_model")
        self.execute_model_state = SimpleNamespace(
            input_batch=SimpleNamespace(logits_indices=logits_indices),
            hidden_states=hidden_states,
        )
        return "forward-output"

    monkeypatch.setattr(
        GPUModelRunner, "execute_model", fake_super_execute_model)

    def fake_super_dummy(self, num_tokens, *args, **kwargs):
        assert self.execute_model(
            "scheduler", dummy_run=True, is_profile=False) == "forward-output"
        calls.append("speculator")
        return hidden_states, hidden_states[logits_indices]

    monkeypatch.setattr(GPUModelRunner, "_dummy_run", fake_super_dummy)

    runner = npu_model_runner.NPUModelRunnerV2.__new__(
        npu_model_runner.NPUModelRunnerV2
    )
    runner.model = SimpleNamespace(
        compute_logits=lambda sample_hs: calls.append(
            ("compute_logits", sample_hs.tolist())
        )
    )

    result = runner._dummy_run(4)

    assert torch.equal(result[0], hidden_states)
    assert torch.equal(result[1], hidden_states[logits_indices])
    assert calls == [
        "execute_model",
        ("compute_logits", hidden_states[logits_indices].tolist()),
        "speculator",
    ]


def test_dp_lmhead_enabled_reads_model_parallel_config(monkeypatch):
    from omni_npu.model_config.config_loader.loader import model_extra_config
    from omni_npu.worker.npu import model_runner as npu_model_runner

    parallel_config = SimpleNamespace(
        ena_dp_lmhead_parallel=True,
        ena_local_lmhead_parallel=False,
    )
    monkeypatch.setattr(model_extra_config, "parall_config", parallel_config)

    assert npu_model_runner._dp_lmhead_enabled()


def _make_inputs(num_tokens, vocab, temperatures, device="cpu"):
    """Shared logits/idx_mapping/temperature/seed/pos fixtures for sample tests.

    device defaults to "cpu" for tests that only exercise Python-level control
    flow (e.g. the use_fp64 raise); NPU kernel tests pass device="npu".
    """
    import torch

    torch.manual_seed(0)
    logits = torch.randn(num_tokens, vocab, device=device)
    idx_mapping = torch.arange(num_tokens, dtype=torch.int32, device=device)
    temperature = torch.tensor(temperatures, dtype=torch.float32, device=device)
    seeds = torch.zeros(num_tokens, dtype=torch.int64, device=device)
    pos = torch.arange(num_tokens, dtype=torch.int32, device=device)
    return logits, idx_mapping, temperature, seeds, pos


def test_gumbel_sample_use_fp64_rejected():
    """use_fp64 must fail-loud at the Python entry (no kernel launch).

    triton-ascend does not support fp64 here; this is a CPU-safe branch test.
    """
    from omni_npu.worker.npu.sampler import gumbel_sample

    num_tokens, vocab = 2, 64
    logits, idx_mapping, temperature, seeds, pos = _make_inputs(
        num_tokens, vocab, [0.0] * num_tokens
    )

    with pytest.raises(NotImplementedError, match="FP64"):
        gumbel_sample(
            logits, idx_mapping, temperature, seeds, pos,
            apply_temperature=False, use_fp64=True,
        )


def test_apply_top_k_top_p_without_filters_returns_original_logits():
    import torch

    from omni_npu.worker.npu.sampler import apply_top_k_top_p

    logits = torch.randn(2, 8)

    assert apply_top_k_top_p(logits, None, None) is logits


def _npu_available() -> bool:
    try:
        import torch
        import torch_npu  # noqa: F401
    except ImportError:
        return False
    return hasattr(torch, "npu") and torch.npu.is_available()


requires_npu = pytest.mark.skipif(not _npu_available(), reason="NPU is unavailable")


def _npu_sampling_available() -> bool:
    if not _npu_available():
        return False
    from omni_npu.worker.npu import sampler

    # Some UT shards run with vLLM's no-Triton fallback.  In that fallback
    # @triton.jit is an identity decorator, so the kernel is a plain function
    # and cannot be launched with kernel[grid](...).
    return hasattr(sampler._gumbel_sample_kernel, "__getitem__")


requires_npu_sampling = pytest.mark.xfail(
    _npu_available() and not _npu_sampling_available(),
    reason="the active vLLM Triton backend cannot launch the NPU sampler kernel",
    raises=TypeError,
    strict=True,
)



@requires_npu_sampling
@requires_npu
def test_gumbel_sample_pure_greedy():
    """纯 greedy：apply_temperature=False 或全 temp=0，返回 argmax + int64。"""
    import torch

    from omni_npu.worker.npu.sampler import gumbel_sample

    dev = "npu"
    num_tokens, vocab = 5, 1024
    logits, idx_mapping, temperature, seeds, pos = _make_inputs(
        num_tokens, vocab, [0.0] * num_tokens, device=dev
    )

    # 主采样路径：apply_temperature=False（温度已由 states 提前作用）
    got = gumbel_sample(
        logits, idx_mapping, temperature, seeds, pos, apply_temperature=False
    )
    torch.npu.synchronize()
    assert got.dtype == torch.int64
    assert torch.equal(got.cpu(), logits.cpu().argmax(dim=-1))

    # 全 greedy 批即使 apply_temperature=True（全 temp=0）也不该被误伤
    got = gumbel_sample(
        logits.clone(), idx_mapping, temperature, seeds, pos,
        apply_temperature=True,
    )
    torch.npu.synchronize()
    assert torch.equal(got.cpu(), logits.cpu().argmax(dim=-1))


@requires_npu_sampling
@requires_npu
def test_gumbel_sample_random_sampling():
    """纯 random：apply_temperature=True 且全 temp!=0，应正常采样（不再 raise）。

    M4 P0 曾让此场景 fail-loud；P1 平移 vllm-ascend kernel 后，random 采样可用。
    断言：合法 token id + 同 seed 确定性复现。
    """
    import torch

    from omni_npu.worker.npu.sampler import gumbel_sample

    dev = "npu"
    num_tokens, vocab = 5, 1024
    logits, idx_mapping, temperature, seeds, pos = _make_inputs(
        num_tokens, vocab, [0.7] * num_tokens, device=dev
    )

    r1 = gumbel_sample(
        logits.clone(), idx_mapping, temperature, seeds, pos,
        apply_temperature=True,
    )
    torch.npu.synchronize()
    r2 = gumbel_sample(
        logits.clone(), idx_mapping, temperature, seeds, pos,
        apply_temperature=True,
    )
    torch.npu.synchronize()

    # 合法 token id
    assert (r1.cpu() >= 0).all() and (r1.cpu() < vocab).all()
    # 同 seed 两次调用 bitwise 一致（G2 确定性）
    assert torch.equal(r1.cpu(), r2.cpu())


@requires_npu_sampling
@requires_npu
def test_gumbel_sample_mixed_rows():
    """mixed rows：批内 greedy 行（temp=0）返回 argmax，random 行（temp!=0）返回合法 token。

    MRv2 把不同 req 的采样参数经 expanded_idx_mapping 间接寻址到同一批 token 行，
    所以"半 greedy 半 random"是最易漏的覆盖。平移后的 kernel 逐行按各自 temp 决策。
    """
    import torch

    from omni_npu.worker.npu.sampler import gumbel_sample

    dev = "npu"
    num_tokens, vocab = 4, 1024
    # req 0/1 greedy（temp=0），req 2/3 random（temp=0.7）；经 idx_mapping 映射到行
    logits, idx_mapping, temperature, seeds, pos = _make_inputs(
        num_tokens, vocab, [0.0, 0.0, 0.7, 0.7], device=dev
    )

    got = gumbel_sample(
        logits.clone(), idx_mapping, temperature, seeds, pos,
        apply_temperature=True,
    )
    torch.npu.synchronize()
    got_cpu = got.cpu()

    # greedy 行（0/1）必须是 argmax
    expected_greedy = logits.cpu().argmax(dim=-1)
    assert got_cpu[0].item() == expected_greedy[0].item(), (
        f"greedy row 0 mismatch: got {got_cpu[0].item()} expected {expected_greedy[0].item()}"
    )
    assert got_cpu[1].item() == expected_greedy[1].item(), (
        f"greedy row 1 mismatch: got {got_cpu[1].item()} expected {expected_greedy[1].item()}"
    )
    # random 行（2/3）合法 token id（不要求等于 argmax）
    assert 0 <= got_cpu[2].item() < vocab
    assert 0 <= got_cpu[3].item() < vocab


@requires_npu_sampling
@requires_npu
def test_gumbel_sample_processed_logits():
    """processed_logits 回写：apply_temperature=True 时回写 logits/temp。

    EAGLE speculator 路径（speculator.py:282-284）会传 output_processed_logits。
    平移后的 kernel 支持回写（参照 vllm-ascend gumbel.py:132-142）。
    """
    import torch

    from omni_npu.worker.npu.sampler import gumbel_sample

    dev = "npu"
    num_tokens, vocab = 5, 1024
    logits, idx_mapping, temperature, seeds, pos = _make_inputs(
        num_tokens, vocab, [0.8] * num_tokens, device=dev
    )

    out_logits = torch.zeros(num_tokens, vocab, dtype=torch.float32, device=dev)
    gumbel_sample(
        logits, idx_mapping, temperature, seeds, pos,
        apply_temperature=True,
        output_processed_logits=out_logits,
    )
    torch.npu.synchronize()

    for tok in range(num_tokens):
        temp = temperature[tok].item()
        expected = logits[tok].float().cpu() / temp
        assert torch.allclose(out_logits[tok].cpu(), expected, atol=1e-4, rtol=1e-4), (
            f"processed_logits mismatch at token {tok} (temp={temp:.3f}): "
            f"max_diff={(out_logits[tok].float().cpu() - expected).abs().max().item():.6f}"
        )


@requires_npu
def test_apply_top_k_top_p_mask_layout():
    """NPU apply_top_k_top_p（PyTorch sort）mask 布局与 numpy sort oracle 一致。

    上游 Qrita Triton kernel 在 triton-ascend 3.2.2 编译失败，omni setattr 强制走
    PyTorch sort。本测试直接调 omni 注入版，验证三种 case 的 mask 布局（哪些位置变 -inf）
    与 numpy fp64 sort oracle 完全一致——top-k/top-p 是纯过滤，mask 布局是正确性核心。
    """
    import numpy as np
    import torch

    from omni_npu.worker.npu.sampler import apply_top_k_top_p

    dev = "npu"

    def oracle_mask(logits_np, k_np, p_np):
        out = logits_np.astype(np.float64).copy()
        for r in range(out.shape[0]):
            order = np.argsort(out[r])
            row_sort = out[r][order]
            if k_np is not None:
                kk = int(k_np[r])
                thresh = row_sort[-kk] if kk < len(row_sort) else row_sort[0]
                row_sort = np.where(row_sort < thresh, -np.inf, row_sort)
            if p_np is not None:
                probs = np.exp(row_sort - row_sort.max())
                probs = probs / probs.sum()
                csum = np.cumsum(probs)
                m = csum <= (1.0 - p_np[r])
                m[-1] = False
                row_sort = np.where(m, -np.inf, row_sort)
            out[r] = row_sort[np.argsort(order)]
        return np.isinf(out)

    torch.manual_seed(0)
    num_tokens, vocab = 8, 1024
    logits = torch.randn(num_tokens, vocab, dtype=torch.float32, device=dev)
    logits[0, :50] = float("-inf")  # grammar-mask 模拟

    # Case 1: top-k only
    k = torch.full((num_tokens,), 10, dtype=torch.int32, device=dev)
    out = apply_top_k_top_p(logits.clone(), k, None)
    torch.npu.synchronize()
    out_inf = np.isinf(out.cpu().numpy())
    orc_inf = oracle_mask(logits.cpu().numpy(), k.cpu().numpy(), None)
    assert (out_inf == orc_inf).all(), "top-k only mask layout mismatch"

    # Case 2: top-p only
    p = torch.full((num_tokens,), 0.9, dtype=torch.float32, device=dev)
    out = apply_top_k_top_p(logits.clone(), None, p)
    torch.npu.synchronize()
    out_inf = np.isinf(out.cpu().numpy())
    orc_inf = oracle_mask(logits.cpu().numpy(), None, p.cpu().numpy())
    assert (out_inf == orc_inf).all(), "top-p only mask layout mismatch"

    # Case 3: per-row varied k/p（MRv2 实际形态：req 各自参数）
    k_v = torch.tensor([5, 10, 20, 50, 100, 1, vocab, 10], dtype=torch.int32, device=dev)
    p_v = torch.tensor([0.8, 0.9, 0.95, 1.0, 0.5, 0.99, 1.0, 0.9],
                       dtype=torch.float32, device=dev)
    out = apply_top_k_top_p(logits.clone(), k_v, p_v)
    torch.npu.synchronize()
    out_inf = np.isinf(out.cpu().numpy())
    orc_inf = oracle_mask(logits.cpu().numpy(), k_v.cpu().numpy(), p_v.cpu().numpy())
    assert (out_inf == orc_inf).all(), "per-row k/p mask layout mismatch"


@requires_npu_sampling
@requires_npu
def test_gumbel_sample_with_top_k_top_p():
    """组合参数端到端：apply_top_k_top_p → gumbel_sample 完整调度。

    验证 top-k/top-p 过滤后的 logits 进入 gumbel 采样：采出的 token 必落在
    未被 mask 的候选集内（-inf 位置不可能被 argmax 选中）。
    """
    import torch

    from omni_npu.worker.npu.sampler import apply_top_k_top_p, gumbel_sample

    dev = "npu"
    num_tokens, vocab = 8, 1024
    logits, idx_mapping, temperature, seeds, pos = _make_inputs(
        num_tokens, vocab, [0.7] * num_tokens, device=dev
    )
    # top_k=10 过滤：每行只留 top-10 候选
    k = torch.full((num_tokens,), 10, dtype=torch.int32, device=dev)
    filtered = apply_top_k_top_p(logits.clone(), k, None)
    torch.npu.synchronize()
    # 采样（apply_temperature=False，温度已在 logits 之外处理）
    sampled = gumbel_sample(
        filtered, idx_mapping, temperature, seeds, pos, apply_temperature=False
    )
    torch.npu.synchronize()
    sampled_cpu = sampled.cpu()

    # 每个采出的 token 必须在未被 mask（非 -inf）的位置
    filtered_cpu = filtered.cpu()
    for tok in range(num_tokens):
        tid = sampled_cpu[tok].item()
        assert filtered_cpu[tok, tid] != float("-inf"), (
            f"token {tok} 采样到 {tid}，但该位置已被 top-k mask 成 -inf"
        )
        assert 0 <= tid < vocab


@requires_npu_sampling
@requires_npu
def test_rejection_sample_greedy_no_draft_logits():
    """Greedy one-hot draft path: accept prefix, then stop at first mismatch."""
    import torch

    from omni_npu.worker.npu.ops.rejection_sampler_utils import rejection_sample

    dev = "npu"
    vocab = 1024
    target_logits = torch.full((3, vocab), -20.0, dtype=torch.float32, device=dev)
    target_logits[0, 1] = 20.0
    target_logits[1, 3] = 20.0
    target_logits[2, 7] = 20.0
    draft_sampled = torch.tensor([0, 1, 2], dtype=torch.int64, device=dev)
    cu_num_logits = torch.tensor([0, 3], dtype=torch.int64, device=dev)
    pos = torch.arange(3, dtype=torch.int32, device=dev)
    idx_mapping = torch.tensor([0], dtype=torch.int64, device=dev)
    expanded_idx_mapping = torch.zeros(3, dtype=torch.int64, device=dev)
    expanded_local_pos = torch.tensor([0, 1, 2], dtype=torch.int64, device=dev)
    temperature = torch.tensor([0.0], dtype=torch.float32, device=dev)
    seed = torch.tensor([0], dtype=torch.int64, device=dev)

    sampled, num_sampled = rejection_sample(
        target_logits,
        None,
        draft_sampled,
        cu_num_logits,
        pos,
        idx_mapping,
        expanded_idx_mapping,
        expanded_local_pos,
        temperature,
        seed,
        num_speculative_steps=2,
    )
    torch.npu.synchronize()

    assert num_sampled.cpu().tolist() == [2]
    assert sampled.cpu()[0, :2].tolist() == [1, 3]


@requires_npu_sampling
@requires_npu
def test_rejection_sample_block_verification_with_draft_logits():
    """Block verification path accepts matching draft distributions."""
    import torch

    from omni_npu.worker.npu.ops.rejection_sampler_utils import rejection_sample

    dev = "npu"
    vocab = 1024
    target_logits = torch.full((3, vocab), -20.0, dtype=torch.float32, device=dev)
    target_logits[0, 5] = 20.0
    target_logits[1, 9] = 20.0
    target_logits[2, 13] = 20.0
    draft_logits = torch.full((1, 2, vocab), -20.0, dtype=torch.float32, device=dev)
    draft_logits[0, 0, 5] = 20.0
    draft_logits[0, 1, 9] = 20.0
    draft_sampled = torch.tensor([0, 5, 9], dtype=torch.int64, device=dev)
    cu_num_logits = torch.tensor([0, 3], dtype=torch.int64, device=dev)
    pos = torch.arange(3, dtype=torch.int32, device=dev)
    idx_mapping = torch.tensor([0], dtype=torch.int64, device=dev)
    expanded_idx_mapping = torch.zeros(3, dtype=torch.int64, device=dev)
    expanded_local_pos = torch.tensor([0, 1, 2], dtype=torch.int64, device=dev)
    temperature = torch.tensor([0.7], dtype=torch.float32, device=dev)
    seed = torch.tensor([11], dtype=torch.int64, device=dev)

    sampled, num_sampled = rejection_sample(
        target_logits,
        draft_logits,
        draft_sampled,
        cu_num_logits,
        pos,
        idx_mapping,
        expanded_idx_mapping,
        expanded_local_pos,
        temperature,
        seed,
        num_speculative_steps=2,
        use_block_verification=True,
    )
    torch.npu.synchronize()

    sampled_cpu = sampled.cpu()
    num_sampled_cpu = num_sampled.cpu()
    assert num_sampled_cpu.tolist() == [3]
    assert sampled_cpu[0, :2].tolist() == [5, 9]
    assert 0 <= sampled_cpu[0, 2].item() < vocab


def test_mrv1_path_does_not_import_upstream_gpu_package():
    """反向用例：omni 的 MRv1 轨对上游 gpu 包零新增 import。"""
    code = (
        "import sys\n"
        "def gpu_mods():\n"
        "    return {m for m in sys.modules\n"
        "            if m == 'vllm.v1.worker.gpu'\n"
        "            or m.startswith('vllm.v1.worker.gpu.')}\n"
        # 上游基线：V1 轨（gpu_worker → gpu_model_runner）自己就会 import gpu 包
        "import vllm.v1.worker.gpu_worker  # noqa: F401\n"
        "baseline = gpu_mods()\n"
        "import omni_npu.worker.npu_worker  # noqa: F401\n"
        "new = sorted(gpu_mods() - baseline)\n"
        "assert not new, f'omni 为 MRv1 路径多 import 了上游 gpu 包: {new}'\n"
    )
    env = {
        k: v
        for k, v in os.environ.items()
        if k != "VLLM_USE_V2_MODEL_RUNNER"
    }
    proc = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, (
        f"反向用例失败：\nstdout: {proc.stdout[-2000:]}\nstderr: {proc.stderr[-2000:]}"
    )


def _make_kv_cache_runner(monkeypatch, produced):
    """给拷贝体搭最小可跑环境：只桩掉重的，纯函数留真的。"""
    from types import SimpleNamespace as NS

    import vllm.v1.worker.gpu.model_runner as up

    from omni_npu.worker.npu.model_runner import NPUModelRunnerV2

    seen = {}
    monkeypatch.setattr(up, "init_attn_backend", lambda *a: (
        [["group"]], NS(min_cg_support="s", min_cg_attn_backend="b"), [128]))
    monkeypatch.setattr(up, "BlockTables", lambda **kw: "block-tables")
    monkeypatch.setattr(up, "initialize_mamba_ssu_backend", lambda *a: None)
    monkeypatch.setattr(up, "ModelCudaGraphManager", lambda *a, **kw: "cg-manager")
    monkeypatch.setattr(up, "check_attention_cp_compatibility", lambda cfg: None)
    monkeypatch.setattr(up, "init_kv_cache", lambda *a: produced)
    monkeypatch.setattr(up, "get_kv_connector", lambda cfg, d: seen.setdefault(
        "connector_arg", d))

    runner = object.__new__(NPUModelRunnerV2)
    runner.max_model_len = 1024
    runner.is_encoder_decoder = False
    runner.dcp_size, runner.dcp_rank, runner.cp_interleave = 1, 0, False
    runner.max_num_reqs, runner.max_num_tokens = 4, 64
    runner.decode_query_len, runner.lora_capture_cases = 1, None
    runner.speculator, runner.model_state = None, None
    runner.device = "npu"
    runner.cache_config = NS(enable_prefix_caching=False, cache_dtype="auto")
    runner.parallel_config = NS(tensor_parallel_size=1)
    runner.vllm_config = NS(mamba_config=None)
    runner.compilation_config = NS(
        static_forward_context={},
        resolve_cudagraph_mode_and_sizes=lambda *a, **kw: "cg-mode",
    )
    kv_cache_config = NS(kv_cache_groups=[NS(kv_cache_spec=NS(block_size=128))])
    return runner, kv_cache_config, seen


def test_capture_model_leaves_accelerator_apis_to_the_process_patch(monkeypatch):
    """capture_model 不再包 torch.accelerator —— !2496 已在进程级重定向。

    MRv1 在同一个变更里也删掉了自己那对 mock.patch。这里钉的是"不再有作用域"：
    捕获期间看到的对象与外面完全一致，说明没有人再临时换进换出。
    """
    import torch
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    from omni_npu.worker.npu import aclgraph_utils
    from omni_npu.worker.npu.model_runner import NPUModelRunnerV2

    # 进程级补丁的效果：这就是 TorchAcceleratorMemoryPatch 装上后的样子
    def redirected_empty_cache():
        return None

    def redirected_get_memory_info():
        return (3, 4)

    monkeypatch.setattr(torch.accelerator, "empty_cache", redirected_empty_cache)
    monkeypatch.setattr(
        torch.accelerator, "get_memory_info", redirected_get_memory_info,
        raising=False,
    )

    seen = {}

    def upstream_capture(self):
        seen["empty_cache"] = torch.accelerator.empty_cache
        seen["get_memory_info"] = torch.accelerator.get_memory_info
        return 17

    monkeypatch.setattr(GPUModelRunner, "capture_model", upstream_capture)
    monkeypatch.setattr(aclgraph_utils, "ensure_graph_params", lambda cfg: None)
    monkeypatch.setattr(aclgraph_utils, "assert_no_acl_tasks_registered", lambda: None)

    runner = object.__new__(NPUModelRunnerV2)
    runner.vllm_config = SimpleNamespace(compilation_config=object())
    runner.cudagraph_manager = None
    runner.speculator = None

    assert runner.capture_model() == 17
    # 捕获期间与捕获之外是同一个对象：没有临时替换
    assert seen["empty_cache"] is redirected_empty_cache
    assert seen["get_memory_info"] is redirected_get_memory_info
    assert torch.accelerator.empty_cache is redirected_empty_cache
    assert torch.accelerator.get_memory_info is redirected_get_memory_info


def test_capture_model_does_not_reference_accelerator_itself():
    """源码级复核：capture_model 里不该再出现 torch.accelerator 或 mock.patch。"""
    import ast
    import inspect
    import textwrap

    from omni_npu.worker.npu.model_runner import NPUModelRunnerV2

    src = textwrap.dedent(inspect.getsource(NPUModelRunnerV2.capture_model))
    fn = ast.parse(src).body[0]
    # docstring 里就写着 torch.accelerator（解释为什么不再包），只查代码
    if (fn.body and isinstance(fn.body[0], ast.Expr)
            and isinstance(fn.body[0].value, ast.Constant)):
        fn.body.pop(0)
    code = ast.unparse(fn)
    assert "torch.accelerator" not in code
    assert "unittest.mock" not in code

    # 仍然只调一次上游实现
    calls = []
    for node in ast.walk(fn):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "capture_model"):
            calls.append(node)
    assert len(calls) == 1


def test_constructor_installs_graph_hooks_after_cuda_aliases(monkeypatch):
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    from omni_npu.worker.npu import model_runner as runner_module
    from omni_npu.worker.npu.model_runner import NPUModelRunnerV2

    events = []
    monkeypatch.setattr(
        runner_module,
        "install_torch_cuda_aliases",
        lambda: events.append("aliases"),
    )
    from omni_npu.compilation import npugraph_ex_config
    monkeypatch.setattr(
        npugraph_ex_config,
        "init_aclgraph_config",
        lambda config: events.append("acl-config"),
    )
    monkeypatch.setattr(
        GPUModelRunner,
        "__init__",
        lambda self, config, device: events.append("upstream-init"),
    )

    NPUModelRunnerV2(object(), object())

    # The aliases must precede init_aclgraph_config and the upstream __init__,
    # which touch torch.cuda. Nothing else is installed here any more: both
    # capture-time signals ship as plugin patches (patch_mrv2_capture_mode.py).
    assert events == ["aliases", "acl-config", "upstream-init"]


def test_constructor_initializes_aclgraph_config_without_additional_config(
    clean_cuda_aliases, monkeypatch
):
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    from omni_npu.compilation import npugraph_ex_config
    from omni_npu.worker.npu import model_runner as runner_module

    seen = []
    monkeypatch.setattr(runner_module, "install_torch_cuda_aliases", lambda: None)
    monkeypatch.setattr(npugraph_ex_config, "init_aclgraph_config", seen.append)
    monkeypatch.setattr(GPUModelRunner, "__init__", lambda self, cfg, dev: None)

    config = SimpleNamespace(additional_config=None)
    runner_module.NPUModelRunnerV2(config, object())

    assert seen == [config]


def test_prepare_dummy_attn_zeroes_block_tables(monkeypatch):
    """dummy forward 必须落到 null block —— MoME 按 block table 寻址，不吃 slot 中和。

    同时钉住"张量对象本身不变"：cudagraph 捕获依赖这些 tensor 的地址。
    """
    import torch
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    from omni_npu.worker.npu.model_runner import NPUModelRunnerV2

    tables = [
        torch.tensor([[6, 5], [4, 3]], dtype=torch.int32),
        torch.tensor([[2, 1]], dtype=torch.int32),
    ]
    slots = object()
    monkeypatch.setattr(
        GPUModelRunner,
        "prepare_dummy_attn",
        lambda self, input_batch: (tables, slots),
    )
    runner = object.__new__(NPUModelRunnerV2)

    got_tables, got_slots = runner.prepare_dummy_attn(object())

    assert got_slots is slots
    assert [t is orig for t, orig in zip(got_tables, tables)] == [True, True]
    assert all(int(t.sum()) == 0 for t in got_tables)
