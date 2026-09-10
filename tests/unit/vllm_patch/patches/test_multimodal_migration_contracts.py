# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""CPU contracts for the migration; no NPU, model weights or vLLM install.

Exercise real patch modules against small interface doubles. The backbone's
forward and shared RoPE lookup are extracted intact to avoid importing NPU ops;
these are method-level tests, not compiled-model or device integration tests.
"""

import argparse
import ast
import functools
import importlib.util
import logging
import sys
import time
import types
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

ROOT = Path(__file__).resolve().parents[4]
PATCH_ROOT = ROOT / "omni/vllm_patches/patches"


def _stub(monkeypatch, name, **attrs):
    module = types.ModuleType(name)
    module.__path__ = []
    module.__dict__.update(attrs)
    monkeypatch.setitem(sys.modules, name, module)
    if "." in name:
        parent, attr = name.rsplit(".", 1)
        if parent not in sys.modules:
            _stub(monkeypatch, parent)
        monkeypatch.setattr(sys.modules[parent], attr, module, raising=False)
    return module


def _load(monkeypatch, name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def _core(monkeypatch):
    registered = {}
    _stub(monkeypatch, "vllm.logger", init_logger=logging.getLogger)
    _stub(monkeypatch, "omni_npu.vllm_patches",
          PatchManager=SimpleNamespace(register=registered.__setitem__))
    _load(monkeypatch, "omni_npu.vllm_patches.core",
          ROOT / "omni/vllm_patches/core.py")
    return registered


@pytest.mark.parametrize("patch_dirs", [
    "low_latency",
    "high_throughout",
    "pangu_v2_base",
    "pangu_v2_moe,openpangu_v1_vl",
    "pangu_v2_hybrid,pangu_v2_moe,openpangu_v1_vl",
])
def test_base_kv_patch_preserves_upstream_memory_check_and_estimator(monkeypatch, patch_dirs):
    registered = _core(monkeypatch)
    _stub(monkeypatch, "omni_npu.envs", OMNI_VLLM_PATCHES_DIR=patch_dirs)
    _stub(monkeypatch, "vllm.utils.math_utils", cdiv=lambda a, b: (a + b - 1) // b)
    original_check = MagicMock()
    original_estimate = MagicMock()
    target = _stub(
        monkeypatch, "vllm.v1.core.kv_cache_utils", logger=MagicMock(),
        create_kv_cache_group_specs=MagicMock(),
        _check_enough_kv_cache_memory=original_check,
        _max_memory_usage_bytes_from_groups=original_estimate,
    )
    _stub(monkeypatch, "vllm.v1.kv_cache_interface", KVCacheSpec=object)
    mod = _load(monkeypatch, "migration_kv_utils",
                PATCH_ROOT / "models/pangu_v2_base/patch_kv_cache_utils.py")
    assert set(registered) == {"OverrideGroupSizePatch"}
    registered["OverrideGroupSizePatch"].apply()
    assert target._get_kv_cache_groups_uniform_page_size is (
        mod._get_kv_cache_groups_uniform_page_size_patched
    )
    assert target._max_memory_usage_bytes_from_groups is original_estimate
    assert target._check_enough_kv_cache_memory is original_check


def _function(path, name, namespace, class_name=None):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = tree.body
    if class_name:
        nodes = next(n.body for n in nodes
                     if isinstance(n, ast.ClassDef) and n.name == class_name)
    node = next(n for n in nodes
                if isinstance(n, ast.FunctionDef) and n.name == name)
    module = ast.Module(body=[ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    ), node], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[name]


def _passthrough_mhc_head(value):
    return value, None, None, None, None


def _passthrough_layer(value, *_args):
    return value, None, None, None, None


@pytest.mark.parametrize("expanded", [False, True])
@pytest.mark.parametrize("first_rank", [False, True])
def test_backbone_mhc_preserves_token_axis(expanded, first_rank):
    tokens, streams, hidden = 5, 4, 8
    x = torch.arange(tokens * hidden, dtype=torch.float32).reshape(tokens, hidden)
    expected = x[:, None, :].repeat(1, streams, 1)
    input_embeds = expected.flatten(1) if expanded else x
    cos, sin = object(), object()
    rotary = SimpleNamespace(get_cos_sin=MagicMock(return_value=(cos, sin)))
    layer = MagicMock()
    layer.self_attn = SimpleNamespace(rotary_emb=rotary)
    layer.mhc_head.side_effect = _passthrough_mhc_head
    layer.side_effect = _passthrough_layer
    model = SimpleNamespace(use_mhc=True, need_tp_padding=False, hidden_size=hidden,
                            mhc_num_stream=streams, start_layer=0, end_layer=1,
                            layers=[layer], cos_cached=None, sin_cached=None)
    namespace = dict(torch=torch, high_throughout=lambda: False,
                     get_pp_group=lambda: SimpleNamespace(
                         is_first_rank=first_rank, is_last_rank=True))
    forward = _function(ROOT / "omni/v1/models/pangu/pangu_v2_moe.py",
                        "forward", namespace, "OpenPanguV2Model")
    positions = torch.arange(tokens).repeat(3, 1)
    output = forward(model, None, positions,
                     {"hidden_states": input_embeds, "residual": None}, input_embeds)
    torch.testing.assert_close(output, expected, atol=0, rtol=0)
    rotary.get_cos_sin.assert_called_once_with(positions)
    assert layer.call_args.args[4:6] == (cos, sin)


def test_text_rope_values_are_bit_identical():
    lookup = _function(ROOT / "omni/layers/rotary_embedding/common.py",
                       "get_cos_sin", {"torch": torch})
    cache = torch.arange(128).reshape(16, 8)
    positions = torch.tensor([0, 3, 7, 15])
    cos, sin = lookup(cache, -cache, positions)
    assert torch.equal(cos.reshape(-1, 8), cache.index_select(0, positions))
    assert torch.equal(sin.reshape(-1, 8), (-cache).index_select(0, positions))


def _initialize_backbone_rotary_caches(model):
    """Execute the real constructor fragment, not the NPU model constructor."""
    path = ROOT / "omni/v1/models/pangu/pangu_v2_moe.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    model_class = next(node for node in tree.body
                       if isinstance(node, ast.ClassDef) and node.name == "OpenPanguV2Model")
    init = next(node for node in model_class.body
                if isinstance(node, ast.FunctionDef) and node.name == "__init__")

    def assignment_index(name):
        for index, node in enumerate(init.body):
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if not isinstance(target, ast.Attribute):
                    continue
                if not isinstance(target.value, ast.Name):
                    continue
                if target.value.id == "self" and target.attr == name:
                    return index
        raise AssertionError(f"Missing self.{name} assignment in OpenPanguV2Model.__init__")

    start = assignment_index("cos_cached")
    end = assignment_index("attn_layer_name")
    fragment = ast.Module(body=init.body[start:end], type_ignores=[])
    exec(compile(fragment, str(path), "exec"), {"self": model})


@pytest.mark.parametrize("cache_names", [(), ("cos_cached",), ("sin_cached",),
                                        ("cos_cached", "sin_cached")])
def test_backbone_initializes_cache_aliases_only_when_both_exist(cache_names):
    rotary = SimpleNamespace(cos_sin_cache=torch.ones(16, 16))
    for name in cache_names:
        setattr(rotary, name, torch.zeros(16, 8))
    model = SimpleNamespace(start_layer=1, layers=[
        None, SimpleNamespace(self_attn=SimpleNamespace(rotary_emb=rotary)),
    ])

    _initialize_backbone_rotary_caches(model)

    if len(cache_names) == 2:
        assert model.cos_cached is rotary.cos_cached
        assert model.sin_cached is rotary.sin_cached
    else:
        assert model.cos_cached is None
        assert model.sin_cached is None
    for name in cache_names:
        assert getattr(rotary, name).shape == (16, 8)
    assert rotary.cos_sin_cache.shape == (16, 16)


@pytest.mark.parametrize("high_throughput", [False, True])
@pytest.mark.parametrize("cache_names, none_cache", [
    ((), None), (("cos_cached",), None), (("sin_cached",), None),
    (("cos_cached", "sin_cached"), None),
    (("cos_cached", "sin_cached"), "cos_cached"),
    (("cos_cached", "sin_cached"), "sin_cached"),
])
def test_backbone_rotary_lookup_preserves_legacy_and_high_throughput_paths(
    high_throughput, cache_names, none_cache
):
    tokens, hidden = 3, 8
    x = torch.zeros(tokens, hidden)
    cache = torch.arange(16 * hidden).reshape(16, hidden)
    has_both_caches = len(cache_names) == 2 and none_cache is None
    positions = (torch.tensor([0, 3, 7]) if has_both_caches
                 else torch.tensor([[0, 3, 7], [0, 4, 8], [0, 5, 9]]))
    api_cos = torch.full((tokens, 1, 1, hidden), 1000)
    api_sin = -api_cos
    rotary = SimpleNamespace(get_cos_sin=MagicMock(return_value=(api_cos, api_sin)))
    for name in cache_names:
        setattr(rotary, name, cache if name == "cos_cached" else -cache)
    if none_cache is not None:
        setattr(rotary, none_cache, None)
    layer = MagicMock()
    layer.self_attn = SimpleNamespace(rotary_emb=rotary)
    layer.mhc_head.side_effect = _passthrough_mhc_head
    layer.side_effect = _passthrough_layer
    model = SimpleNamespace(use_mhc=False, need_tp_padding=False,
                            start_layer=0, end_layer=1, layers=[layer])
    _initialize_backbone_rotary_caches(model)
    forward = _function(
        ROOT / "omni/v1/models/pangu/pangu_v2_moe.py", "forward",
        dict(torch=torch, high_throughout=lambda: high_throughput,
             get_pp_group=lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True)),
        "OpenPanguV2Model",
    )

    output = forward(model, None, positions, None, x)

    assert output is x
    cos, sin = layer.call_args.args[4:6]
    if has_both_caches and not high_throughput:
        rotary.get_cos_sin.assert_not_called()
        assert cos.shape == sin.shape == (tokens, hidden)
        assert torch.equal(cos, cache.index_select(0, positions))
        assert torch.equal(sin, (-cache).index_select(0, positions))
    else:
        rotary.get_cos_sin.assert_called_once_with(positions)
        assert cos is api_cos
        assert sin is api_sin


@pytest.mark.parametrize("mm_first", [False, True])
@pytest.mark.parametrize("mm_config", [None, '{"connectors":{}}'])
def test_engine_args_wrappers_compose_and_mm_is_opt_in(monkeypatch, mm_first, mm_config):
    registered = _core(monkeypatch)
    calls = []

    class VllmConfig:
        additional_config = None

    upstream_config = VllmConfig()

    class EngineArgs:
        def __init__(self, upstream):
            self.upstream = upstream

        @staticmethod
        def add_cli_args(parser):
            parser.add_argument("--upstream", default="kept")
            calls.append("cli")
            return parser

        @classmethod
        def from_cli_args(cls, args):
            instance = cls(args.upstream)
            calls.append("from_cli")
            return instance

        def create_engine_config(self, usage_context=None, headless=False):
            calls.append((usage_context, headless))
            return upstream_config

    _stub(monkeypatch, "vllm", EngineArgs=EngineArgs)
    _stub(monkeypatch, "vllm.config", VllmConfig=VllmConfig)
    _stub(monkeypatch, "vllm.utils.argparse_utils",
          FlexibleArgumentParser=argparse.ArgumentParser)
    parser = EngineArgs.add_cli_args(argparse.ArgumentParser())
    assert not hasattr(parser.parse_args([]), "mm_feature_transfer_config")
    # Model another EngineArgs wrapper without the removed common args patch.
    # Test both sides of the MM wrapper.

    def extra_wrapper():
        upstream = EngineArgs.add_cli_args

        def wrapped(parser):
            parser = upstream(parser)
            parser.add_argument("--later", default="preserved")
            return parser
        EngineArgs.add_cli_args = staticmethod(wrapped)

    if not mm_first:
        extra_wrapper()
    mm = _load(monkeypatch, "migration_mm_args",
               PATCH_ROOT / "models/openpangu_v1_vl/patch_mm_feature_transfer_args.py")
    mm.MMFeatureTransferArgsPatch.apply()
    if mm_first:
        extra_wrapper()
    configurations = []
    _stub(monkeypatch, "omni_npu.connector.mm_feature_transfer", register=configurations.append)
    argv = ["--mm-feature-transfer-config", mm_config] if mm_config else []
    args = EngineArgs.add_cli_args(argparse.ArgumentParser()).parse_args(argv)
    instance = EngineArgs.from_cli_args(args)
    config = instance.create_engine_config("usage", True)
    assert instance.upstream == "kept" and args.later == "preserved"
    assert configurations == ([mm_config] if mm_config else [])
    assert config is upstream_config
    assert config.__dict__ == {}
    assert not hasattr(instance, "routed_experts_serialization_mode")
    assert calls == ["cli", "cli", "from_cli", ("usage", True)]
    assert "MMFeatureTransferArgsPatch" in registered


@pytest.mark.parametrize("symbolic", [False, True])
def test_compile_fallback_retains_real_args_and_reuses_exact_range(monkeypatch, symbolic):
    @dataclass(frozen=True)
    class Range:
        start: int
        end: int

    class PiecewiseBackend:
        def __init__(self, symbolic, manager):
            self.sym_shape_indices = [1] if symbolic else []
            self.range_entries = {}
            self.compile_ranges = [Range(1, 96)]
            self._find_range_for_shape = MagicMock(return_value=None)
            self.graph = object()
            self._log_compile_start = MagicMock()
            self.compilation_config = object()
            self.piecewise_compile_index = 2
            self.total_piecewise_compiles = 3
            self.is_last_graph = True
            self.vllm_backend = SimpleNamespace(
                compiler_manager=manager, inductor_config={}, is_encoder=False
            )

        def __call__(self, *args):
            return "upstream"

    _stub(monkeypatch, "vllm.compilation.piecewise_backend", PiecewiseBackend=PiecewiseBackend,
          RangeEntry=lambda **kwargs: SimpleNamespace(compiled=False, **kwargs))
    _stub(monkeypatch, "vllm.config.utils", Range=Range)
    patch = _function(ROOT / "omni/compilation/decorators.py", "_patch_piecewise_backend",
                      dict(torch=torch, functools=functools, logger=logging.getLogger(__name__)))
    patch()
    runnable = MagicMock(return_value="compiled")
    manager = SimpleNamespace(compile=MagicMock(return_value=runnable), save_to_file=MagicMock())
    instance = PiecewiseBackend(symbolic, manager)
    # In the symbolic case the MRoPE axis 3 must not replace token count 97.
    args = (torch.zeros(3, 97), 97, torch.zeros(97, 8)) if symbolic else (torch.zeros(97, 8),)
    assert instance(*args) == "compiled"
    assert instance(*args) == "compiled"
    manager.compile.assert_called_once()
    compiled_args = manager.compile.call_args.args[1]
    assert all(given is original for given, original in zip(compiled_args, args))
    assert manager.compile.call_args.kwargs["compile_range"] == Range(97, 97)
    manager.save_to_file.assert_called_once()
    if symbolic:
        instance._find_range_for_shape.return_value = object()
        assert instance(*args) == "upstream"


def test_multimodal_embedding_merge_preserves_values_and_empty_input(monkeypatch):
    _core(monkeypatch)

    def flatten(items):
        if isinstance(items, torch.Tensor):
            return items.reshape(-1, items.shape[-1])
        return torch.cat([flatten(item) for item in items])
    _stub(monkeypatch, "vllm.model_executor.models.utils", _flatten_embeddings=flatten,
          _embedding_count_expression=lambda items: "nested embeddings")
    _stub(monkeypatch, "vllm.multimodal", NestedTensors=object)
    mod = _load(monkeypatch, "migration_mm_merge",
                PATCH_ROOT / "models/openpangu_v1_vl/patch_multimodal_embeddings.py")
    merge = mod.NPU_MergeMultimodalEmbeddingsPatch._merge_multimodal_embeddings
    x = torch.arange(20, dtype=torch.float32).reshape(5, 4)
    mask = torch.tensor([False, True, False, True, False])
    assert merge(x, [], mask) is x
    embeddings = [torch.full((1, 4), 20.0), [torch.full((1, 4), 30.0)]]
    expected = x.clone().masked_scatter_(mask[:, None], flatten(embeddings))
    assert merge(x, embeddings, mask) is x
    assert torch.equal(x, expected)
    for count in (0, 1, 3):
        unchanged = x.clone()
        with pytest.raises(ValueError, match="multimodal tokens"):
            merge(x, [torch.zeros(count, 4)], mask)
        assert torch.equal(x, unchanged)


@pytest.mark.parametrize("scenario", ["valid", "open_failed", "bad_fps", "bad_frame", "constructor_failed"])
def test_video_preserves_valid_frames_and_cleans_up_failures(monkeypatch, scenario):
    paths = []
    cap = SimpleNamespace(
        isOpened=lambda: scenario != "open_failed",
        get=lambda key: {1: 3, 2: 0 if scenario == "bad_fps" else 2, 3: 2, 4: 2}[key],
        release=MagicMock(),
    )

    def capture(path):
        paths.append(path)
        if scenario == "constructor_failed":
            raise RuntimeError("capture constructor failed")
        return cap

    _stub(monkeypatch, "cv2", VideoCapture=capture, CAP_PROP_FRAME_COUNT=1,
          CAP_PROP_FPS=2, CAP_PROP_FRAME_WIDTH=3, CAP_PROP_FRAME_HEIGHT=4)
    _stub(monkeypatch, "numpy", ndarray=torch.Tensor, uint8=torch.uint8, empty=torch.empty)
    _stub(monkeypatch, "numpy.typing", NDArray=torch.Tensor)
    _stub(monkeypatch, "vllm.multimodal.video",
          VIDEO_LOADER_REGISTRY=SimpleNamespace(register=lambda name: lambda cls: cls),
          VideoLoader=type("VideoLoader", (), {}),
          OpenCVVideoBackendMixin=type("OpenCVVideoBackendMixin", (), {}))
    mod = _load(monkeypatch, "migration_video", PATCH_ROOT / "models/openpangu_v1_vl/patch_video.py")
    backend = mod.NPUOpenCVDynamicVideoBackend
    monkeypatch.setattr(backend, "_resolve_sampled_frames", lambda **kwargs: ([0, 2], [0.0, 1.0], 1.0))

    def decode(index, *_args):
        if index == 2 and scenario == "bad_frame":
            return None
        return torch.full((2, 2, 3), index, dtype=torch.uint8)
    monkeypatch.setattr(backend, "decode_single_frame", decode)

    if scenario == "valid":
        frames, metadata = backend.load_bytes(b"test-video")
        assert frames.shape == (2, 2, 2, 3)
        assert torch.equal(frames[0], torch.zeros(2, 2, 3, dtype=torch.uint8))
        assert torch.equal(frames[1], torch.full((2, 2, 3), 2, dtype=torch.uint8))
        assert metadata["frames_indices"] == [0, 2]
        assert metadata["sample_frame_timestamps"] == [0.0, 1.0]
    else:
        with pytest.raises((ValueError, RuntimeError)):
            backend.load_bytes(b"test-video")
    if scenario != "constructor_failed":
        cap.release.assert_called_once()
    assert len(paths) == 1 and not Path(paths[0]).exists()
    assert mod.get_extracted_frame_indices(5, 4, 2) == [0, 2, 4]


def _eagle_dummy_contract():
    """Run the real dummy method with CPU buffers and explicit interface doubles."""
    contexts = MagicMock(side_effect=lambda **_kwargs: nullcontext())
    method = _function(
        PATCH_ROOT / "common/patch_eagle.py", "dummy_run",
        dict(torch=torch, CUDAGraphMode=SimpleNamespace(NONE=0),
             PADDING_SLOT_ID=-1, NULL_BLOCK_ID=0,
             set_forward_context=contexts,
             get_forward_context=lambda: SimpleNamespace(capturing=False)),
        "EagleProposerPatch",
    )
    events = []

    def record_logits(**_kwargs):
        events.append("logits")

    model = MagicMock(side_effect=lambda **_kwargs: (
        events.append("forward") or torch.zeros(2, 2)
    ))
    model.compute_logits.side_effect = record_logits
    proposer = SimpleNamespace(
        runner=SimpleNamespace(
            _omni_spec_decode_common_attn_metadata=None,
            batch_execution_and_padding_state=(1, SimpleNamespace(num_tokens=2), None),
            dp_parallel_lmhead=True, local_parallel_lmhead=False,
        ),
        attn_layer_names=["layer0"], num_speculative_tokens=2, n_predict=1,
        supports_mm_inputs=False, method="mtp", vllm_config=object(),
        input_ids=torch.arange(4), inputs_embeds=torch.zeros(4, 2),
        hidden_states=torch.zeros(4, 2), model=model,
        _get_positions=lambda num_tokens: torch.arange(num_tokens),
        model_returns_tuple=lambda: False,
        build_per_group_and_layer_attn_metadata=MagicMock(
            return_value=(None, {"layer0": "built"})
        ),
    )
    return method, proposer, contexts, events


@pytest.mark.parametrize("metadata_kind", ["none", "explicit", "stashed"])
@pytest.mark.parametrize("multimodal", [False, True])
@pytest.mark.parametrize("positional", [False, True])
def test_eagle_dummy_api_preserves_metadata_and_kv_padding(
    metadata_kind, multimodal, positional,
):
    method, proposer, contexts, events = _eagle_dummy_contract()
    proposer.supports_mm_inputs = multimodal
    metadata = None
    expected_metadata = None
    slots = torch.tensor([7, 8], dtype=torch.int32)
    blocks = torch.tensor([[1, 2]], dtype=torch.int32)
    if metadata_kind == "explicit":
        metadata = {"layer0": object(), "unrelated": object()}
        expected_metadata = {"layer0": metadata["layer0"]}
    elif metadata_kind == "stashed":
        stashed = SimpleNamespace(slot_mapping=slots, block_table_tensor=blocks)
        proposer.runner._omni_spec_decode_common_attn_metadata = stashed
        expected_metadata = {"layer0": "built"}

    if positional:
        method(proposer, metadata, 2)
    else:
        method(proposer, num_tokens=2, attn_metadata=metadata)

    assert events == ["forward", "logits", "forward", "logits"]
    assert proposer.runner.batch_execution_and_padding_state is None
    assert all(call.kwargs["attn_metadata"] == expected_metadata
               for call in contexts.call_args_list)
    for call in proposer.model.call_args_list:
        assert (call.kwargs["input_ids"] is None) == multimodal
        assert (call.kwargs["inputs_embeds"] is not None) == multimodal
    # Building dummy metadata must not write live KV slots.
    assert torch.equal(slots, torch.tensor([7, 8], dtype=torch.int32))
    assert torch.equal(blocks, torch.tensor([[1, 2]], dtype=torch.int32))
    if metadata_kind == "stashed":
        assert stashed.slot_mapping is not slots
        assert stashed.block_table_tensor is not blocks
        assert torch.equal(stashed.slot_mapping, torch.full_like(slots, -1))
        assert torch.equal(stashed.block_table_tensor, torch.full_like(blocks, 0))
        proposer.build_per_group_and_layer_attn_metadata.assert_called_once()
    else:
        proposer.build_per_group_and_layer_attn_metadata.assert_not_called()


@pytest.mark.parametrize("n_predict", [1, 2])
@pytest.mark.parametrize("capturing", [False, True])
@pytest.mark.parametrize("profile", [False, True])
def test_eagle_dummy_keeps_mtp_logits_collective_order(n_predict, capturing, profile):
    method, proposer, _, events = _eagle_dummy_contract()
    proposer.n_predict = n_predict
    proposer.num_speculative_tokens = 3
    method(proposer, num_tokens=2, is_graph_capturing=capturing, is_profile=profile)
    steps = min(3, n_predict) if capturing else 3
    assert events == (["forward"] if profile else ["forward", "logits"]) * steps
    if n_predict > 1:
        assert [call.kwargs["spec_step_idx"] for call in proposer.model.call_args_list] == list(range(steps))
        if not profile:
            logits_steps = [
                call.kwargs["spec_step_idx"] for call in proposer.model.compute_logits.call_args_list
            ]
            assert logits_steps == list(range(steps))


def test_eagle_dummy_still_requires_runner_padding_state():
    method, proposer, _, _ = _eagle_dummy_contract()
    proposer.runner.batch_execution_and_padding_state = None
    with pytest.raises(ValueError, match="_determine_batch_execution_and_padding"):
        method(proposer, num_tokens=2)


def test_runner_dummy_call_binds_to_eagle_api():
    method, proposer, contexts, _ = _eagle_dummy_contract()
    tree = ast.parse((ROOT / "omni/worker/npu_model_runner.py").read_text(encoding="utf-8"))
    runner = next(node for node in tree.body
                  if isinstance(node, ast.ClassDef) and node.name == "NPUModelRunner")
    dummy = next(node for node in runner.body
                 if isinstance(node, ast.FunctionDef) and node.name == "_dummy_run")
    calls = []
    for node in ast.walk(dummy):
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "self.drafter.dummy_run":
            calls.append(node)
    assert len(calls) == 1
    call = calls[0]
    assert [ast.unparse(arg) for arg in call.args] == ["attn_metadata", "num_tokens"]
    assert {kw.arg: ast.unparse(kw.value) for kw in call.keywords} == {
        "use_cudagraphs": "use_cudagraphs",
        "is_graph_capturing": "is_graph_capturing",
        "is_profile": "is_profile",
    }
    metadata = {"layer0": object()}
    # Execute the actual runner call expression against the actual patched
    # dummy method; the rest of the device runner is intentionally not executed.
    eval(compile(ast.Expression(call), "<runner dummy call>", "eval"), dict(
        self=SimpleNamespace(drafter=SimpleNamespace(dummy_run=functools.partial(method, proposer))),
        num_tokens=2, use_cudagraphs=True, is_graph_capturing=False,
        attn_metadata=metadata, is_profile=False,
    ))
    assert all(c.kwargs["attn_metadata"] == metadata for c in contexts.call_args_list)


def test_acl_replay_without_metadata_raises_instead_of_skipping_updates():
    get_params = MagicMock(side_effect=AssertionError("must not look up graph tasks"))
    namespace = dict(get_graph_params=get_params)
    update = _function(ROOT / "omni/compilation/acl_graph.py",
                       "_update_graph_tasks", namespace, "ACLGraphWrapper")
    descriptor = MagicMock(num_tokens=2)
    context = SimpleNamespace(attn_metadata=None, batch_descriptor=descriptor,
                              cudagraph_runtime_mode=1)
    namespace.update(time=time, logger=logging.getLogger(__name__), torch=torch,
                     get_forward_context=lambda: context, CUDAGraphMode=SimpleNamespace(NONE=0))
    replay = _function(ROOT / "omni/compilation/acl_graph.py",
                       "__call__", namespace, "ACLGraphWrapper")
    entry = SimpleNamespace(aclgraph=MagicMock(), output=object())
    wrapper = SimpleNamespace(runtime_mode=1, is_debugging_mode=False,
                              update_stream=object(),
                              concrete_aclgraph_entries={descriptor: entry})
    wrapper._update_graph_tasks = functools.partial(update, wrapper)
    with pytest.raises(RuntimeError, match="attn_metadata is empty"):
        replay(wrapper)
    entry.aclgraph.replay.assert_called_once()
    get_params.assert_not_called()


@pytest.mark.parametrize("metadata", [{}, {"attn": object()}])
def test_acl_non_none_metadata_still_updates_tasks(metadata):
    dynamic = MagicMock(return_value={"scale": 2})
    op = MagicMock()
    workspace_fn = MagicMock()
    workspace = object()
    task = SimpleNamespace(
        op_desc=SimpleNamespace(compute_dynamic_kwargs=dynamic, op_out_fn=op,
                                workspace_fn=workspace_fn),
        captured_kwargs={"input_layout": "BSND", "actual_seq_lengths": [2]},
        out_tensors=[object()], handle=object(), event=MagicMock(),
    )
    params = SimpleNamespace(task_entries={2: {"attn": task}},
                             workspaces={2: {workspace_fn: workspace}})
    get_params = MagicMock(return_value=params)
    npu = SimpleNamespace(stream=MagicMock(return_value=nullcontext()),
                          graph_task_update_begin=MagicMock(),
                          graph_task_update_end=MagicMock())
    method = _function(ROOT / "omni/compilation/acl_graph.py", "_update_graph_tasks",
                       dict(torch=SimpleNamespace(npu=npu), get_graph_params=get_params),
                       "ACLGraphWrapper")
    wrapper = SimpleNamespace(attn_layer_names=["attn"], vllm_config=object())
    context = SimpleNamespace(attn_metadata=metadata,
                              batch_descriptor=SimpleNamespace(num_tokens=2))
    stream = object()
    method(wrapper, stream, context)
    get_params.assert_called_once()
    dynamic.assert_called_once_with(context, "attn", wrapper.vllm_config)
    npu.graph_task_update_begin.assert_called_once_with(stream, task.handle)
    op.assert_called_once_with(input_layout="BSND", actual_seq_lengths=None,
                               scale=2, workspace=workspace, out=task.out_tensors)
    npu.graph_task_update_end.assert_called_once_with(stream)
    task.event.record.assert_called_once_with(stream)
    assert task.captured_kwargs["actual_seq_lengths"] == [2]
