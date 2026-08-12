# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: MIT
"""Tests for the MoME ``num_prompt_tokens`` channel (vLLM 0.25.1 adaptation).

Background.  MoME decides which running-cache block holds the conv state it has
to continue from by comparing ``num_computed_tokens`` against the request's
prompt length:

    computed >  prompt  ->  previous schedule was MTP, index by *physical*
                            high-water mark (computed - accepted + num_spec + 1)
    computed <= prompt  ->  previous schedule was a prefill, index by computed

Get that wrong and the block index is off by one whole block -- no exception, no
hang, just a wrong conv state.  Up to 0.14.0 the prompt lengths reached the
builder through ``patch_num_prompt_tokens.py``, which copied the whole of
``GPUModelRunner._build_attention_metadata`` to add one line.  On 0.25.1 the
runner binds them to the builder out of band instead, which is what this file
covers.

Runs with or without vLLM installed.  In a container with real vllm the real
module is imported; on a bare work machine the vllm/omni imports of ``mome.py``
are stubbed and the file is loaded straight from disk.  Either way::

    pytest tests/unit/attention/backends/test_mome_num_prompt_tokens.py -v
    python  tests/unit/attention/backends/test_mome_num_prompt_tokens.py
"""

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

# <repo>/tests/unit/attention/backends/ -> parents[4] is <repo>
REPO_ROOT = Path(__file__).resolve().parents[4]
MOME_PATH = REPO_ROOT / "omni/attention/backends/mome.py"


# --------------------------------------------------------------------------
# module under test: real import when vllm is around, stubbed load otherwise
# --------------------------------------------------------------------------

_ABSENT = object()
_SAVED_MODULES: dict[str, object] = {}


def _install_stubs() -> None:
    """Register the fake modules ``mome.py`` imports at module scope.

    Each name is overridden unconditionally and its previous value remembered,
    then put back in ``tearDownModule``.  Overriding rather than filling gaps
    matters: sibling test files in this directory import the real ``omni_npu``
    (and fail on the vllm import while doing so), which can leave partially
    initialised modules in ``sys.modules`` -- honouring those would drag the
    real, half-built package into this test.  Restoring afterwards keeps the
    fakes from leaking into whatever runs next in the same session.
    """

    def put(name: str, **attrs) -> types.ModuleType:
        if name not in _SAVED_MODULES:
            _SAVED_MODULES[name] = sys.modules.get(name, _ABSENT)
        mod = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(mod, key, value)
        sys.modules[name] = mod
        return mod

    class _GDNAttentionMetadataBuilder:
        """Stand-in for the upstream base class.

        Only needs to exist: every attribute the builder touches is set by hand
        on the instances below, because ``__init__`` is never run.
        """

        def _init_reorder_batch_threshold(self, *args, **kwargs):
            pass

    def _split_decodes_and_prefills(cm, decode_threshold=1, **kwargs):
        # Mirrors the real function's early return for a pure decode batch
        # (max_query_len <= decode_threshold), which is the shape these tests
        # feed it. Anything else would diverge from the container run.
        assert cm.max_query_len <= decode_threshold, (
            "stub only models the pure-decode early return; give the fake "
            "CommonAttentionMetadata query_lens <= decode_threshold"
        )
        return cm.num_reqs, 0, cm.num_actual_tokens, 0

    def _cdiv(a, b):
        if isinstance(a, torch.Tensor):
            return torch.div(a + b - 1, b, rounding_mode="floor")
        return -(-a // b)

    def _register_attention_backend(_name):
        return lambda cls: cls

    put("vllm")
    put("vllm.config", VllmConfig=object)
    put("vllm.distributed", get_tp_group=lambda: SimpleNamespace(world_size=1, rank_in_group=0))
    put("vllm.logger", init_logger=lambda _name: SimpleNamespace(
        debug=lambda *a, **k: None,
        info=lambda *a, **k: None,
        warning=lambda *a, **k: None,
    ))
    put("vllm.utils")
    put("vllm.utils.math_utils", cdiv=_cdiv)
    put("vllm.v1")
    put("vllm.v1.attention")
    put(
        "vllm.v1.attention.backend",
        AttentionBackend=object,
        AttentionCGSupport=SimpleNamespace(ALWAYS="ALWAYS"),
        CommonAttentionMetadata=object,
    )
    put("vllm.v1.attention.backends")
    put("vllm.v1.attention.backends.gdn_attn",
        GDNAttentionMetadataBuilder=_GDNAttentionMetadataBuilder)
    put("vllm.v1.attention.backends.utils",
        PAD_SLOT_ID=-1, split_decodes_and_prefills=_split_decodes_and_prefills)
    put("vllm.v1.kv_cache_interface", AttentionSpec=object, MomeSpec=object)

    put("omni_npu", envs=None)
    put("omni_npu.envs", OMNI_REUSE_PREFILLED_TOKENS=False)
    sys.modules["omni_npu"].envs = sys.modules["omni_npu.envs"]
    put("omni_npu.attention")
    put("omni_npu.attention.backends")
    put("omni_npu.attention.backends.attention", NPUAttentionBackendImpl=object)
    put(
        "omni_npu.attention.backends.utils",
        _maybe_padded_raw_tensor_to_strided_caches=lambda *a, **k: None,
        register_attention_backend=_register_attention_backend,
    )
    put("omni_npu.model_config")
    put("omni_npu.model_config.config_loader")
    put(
        "omni_npu.model_config.config_loader.loader",
        model_extra_config=SimpleNamespace(
            parall_config=SimpleNamespace(enable_flashcomm2=False)
        ),
    )


def _load_mome():
    """Prefer the real module; fall back to loading it against stubs.

    The decision is made by *trying* the real import rather than by probing for
    vllm: sibling test files install their own vllm fakes in ``sys.modules``,
    so ``import vllm`` succeeding proves nothing about whether the real
    ``omni_npu.attention.backends.mome`` can be imported.
    """
    try:
        import omni_npu.attention.backends.mome as real_mome

        return real_mome
    except Exception:  # noqa: BLE001 - any import failure means "use stubs"
        pass

    _install_stubs()
    spec = importlib.util.spec_from_file_location("_mome_under_test", MOME_PATH)
    module = importlib.util.module_from_spec(spec)
    _SAVED_MODULES.setdefault(spec.name, sys.modules.get(spec.name, _ABSENT))
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


mome = None
Builder = None


def setUpModule():
    """Load the module under test here, *not* at import time.

    pytest imports every test module during collection and only then starts
    running them.  Installing the vllm/omni stubs at import time would leave
    them in ``sys.modules`` for the whole collection phase, so every test file
    collected after this one would import fakes -- which silently changes their
    results (it did: +3 failed / +1 error / -2 skipped across
    tests/unit/vllm_patch/).  Deferring to setUpModule keeps the stubs alive
    only between setUpModule and tearDownModule, by which point collection is
    over.
    """
    global mome, Builder
    mome = _load_mome()
    Builder = mome.NPUMomeAttentionMetadataBuilder


def tearDownModule():
    for name, previous in _SAVED_MODULES.items():
        if previous is _ABSENT:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous
    _SAVED_MODULES.clear()


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------

MOME_BLOCK_SIZE = 4
MAX_NUM_REQS = 8


def make_builder(*, num_spec=1, is_decode_node=False, enable_prefix_caching=True):
    """A builder with ``__init__`` bypassed, carrying only what ``build`` reads."""
    b = Builder.__new__(Builder)
    b.vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(enable_prefix_caching=enable_prefix_caching),
        model_config=SimpleNamespace(max_model_len=1024),
        scheduler_config=SimpleNamespace(max_num_seqs=MAX_NUM_REQS),
        speculative_config=None,
    )
    b.compilation_config = SimpleNamespace(
        max_cudagraph_capture_size=None,
        cudagraph_mode=SimpleNamespace(has_full_cudagraphs=lambda: False),
    )
    b.mome_block_size = MOME_BLOCK_SIZE
    b.reorder_batch_threshold = 1
    b.num_spec = num_spec
    b.fake_num_spec = num_spec
    b.use_spec_decode = num_spec > 0
    b.is_pd_disagg = is_decode_node
    b.is_decode_node = is_decode_node
    b.reuse_prefilled_tokens = False
    b.enable_flashcomm2 = False
    b.decode_cudagraph_max_bs = MAX_NUM_REQS
    b.use_full_cuda_graph = False
    b.kv_cache_spec = SimpleNamespace(block_size=MOME_BLOCK_SIZE)
    b.cache_indices_tensor = torch.zeros((MAX_NUM_REQS, 8), dtype=torch.int32)
    b.num_computed_tokens = torch.zeros(MAX_NUM_REQS, dtype=torch.int32)
    b.num_accepted_tokens = torch.zeros(MAX_NUM_REQS, dtype=torch.int32)
    b.block_idx_last_computed_token = torch.zeros(MAX_NUM_REQS, dtype=torch.int32)
    b.block_idx_first_scheduled_token = torch.zeros(MAX_NUM_REQS, dtype=torch.int32)
    b.block_idx_last_scheduled_token = torch.zeros(MAX_NUM_REQS, dtype=torch.int32)
    return b


class FakeCommonAttentionMetadata:
    """The fields ``build()`` reaches for on the path these tests exercise.

    ``query_start_loc_cpu`` is not optional even though nothing in mome.py reads
    it directly: in a container the real
    ``vllm.v1.attention.backends.utils.split_decodes_and_prefills`` runs, and it
    reads that field whenever ``max_query_len > decode_threshold``.  Leaving it
    out passes against the stubs and errors on the real module.
    """

    def __init__(self, seq_lens, query_lens):
        self.seq_lens = torch.tensor(seq_lens, dtype=torch.int32)
        self.num_reqs = len(seq_lens)
        qsl = [0]
        for q in query_lens:
            qsl.append(qsl[-1] + q)
        self.query_start_loc = torch.tensor(qsl, dtype=torch.int32)
        self.query_start_loc_cpu = self.query_start_loc
        self.num_actual_tokens = qsl[-1]
        self.max_query_len = max(query_lens)
        self.block_table_tensor = torch.zeros(
            (self.num_reqs, 8), dtype=torch.int32
        )

    def compute_num_computed_tokens(self):
        query_lens = self.query_start_loc[1:] - self.query_start_loc[:-1]
        return self.seq_lens - query_lens


class FakeAttnGroup:
    def __init__(self, builders):
        self.metadata_builders = builders


# --------------------------------------------------------------------------
# 1. the arithmetic the channel exists to protect
# --------------------------------------------------------------------------


class TestBlockIndexArithmetic(unittest.TestCase):
    """Pin the two branches of ``_compute_prefix_caching_block_indices``.

    Numbers are the two worked examples from the migration analysis, with
    ``mome_block_size = 4`` and ``num_spec = 1``.
    """

    def test_previous_schedule_was_prefill_uses_logical_water_mark(self):
        # prompt = 8 (exactly two full blocks), prefill just wrote 0..7,
        # this is the first decode step: computed == prompt.
        b = make_builder(num_spec=1)
        cm = FakeCommonAttentionMetadata(seq_lens=[10], query_lens=[2])
        self.assertEqual(cm.compute_num_computed_tokens().tolist(), [8])

        num_prompt_tokens = torch.tensor([8], dtype=torch.int32)
        num_accepted = torch.tensor([1], dtype=torch.int32)
        last_computed, _, _ = b._compute_prefix_caching_block_indices(
            cm, MOME_BLOCK_SIZE, num_accepted, num_prompt_tokens
        )
        # cdiv(8, 4) - 1 == 1, the block holding tokens 4..7
        self.assertEqual(last_computed.tolist(), [1])

        # Dropping the prompt lengths (which is what the all-zero buffer did on
        # the drafter path) pushes it into the MTP branch: cdiv(9, 4) - 1 == 2,
        # a block this step has not written yet.
        wrong, _, _ = b._compute_prefix_caching_block_indices(
            cm, MOME_BLOCK_SIZE, num_accepted, torch.tensor([0], dtype=torch.int32)
        )
        self.assertEqual(wrong.tolist(), [2])

    def test_previous_schedule_was_mtp_uses_physical_water_mark(self):
        # previous step scheduled 2 tokens (physical 11, 12), accepted 1,
        # so computed == 12 while the cache has been written up to index 12.
        b = make_builder(num_spec=1)
        cm = FakeCommonAttentionMetadata(seq_lens=[14], query_lens=[2])
        self.assertEqual(cm.compute_num_computed_tokens().tolist(), [12])

        num_prompt_tokens = torch.tensor([8], dtype=torch.int32)
        num_accepted = torch.tensor([1], dtype=torch.int32)
        last_computed, _, _ = b._compute_prefix_caching_block_indices(
            cm, MOME_BLOCK_SIZE, num_accepted, num_prompt_tokens
        )
        # physical high-water = 12 - 1 + 1 + 1 = 13 -> cdiv(13, 4) - 1 == 3
        self.assertEqual(last_computed.tolist(), [3])

        # num_prompt_tokens=None is what the main forward path gets today, and
        # it lands one block low: cdiv(12, 4) - 1 == 2.
        wrong, _, _ = b._compute_prefix_caching_block_indices(
            cm, MOME_BLOCK_SIZE, num_accepted, None
        )
        self.assertEqual(wrong.tolist(), [2])


# --------------------------------------------------------------------------
# 2. how the builder resolves num_prompt_tokens
# --------------------------------------------------------------------------


class TestNumPromptTokensResolution(unittest.TestCase):
    def setUp(self):
        self.captured = {}
        self.builder = make_builder(num_spec=1)

        def spy(cm, block_size, num_accepted_tokens, num_prompt_tokens=None):
            self.captured["num_prompt_tokens"] = num_prompt_tokens
            zeros = torch.zeros(cm.num_reqs, dtype=torch.int32)
            return zeros, zeros.clone(), zeros.clone()

        self.builder._compute_prefix_caching_block_indices = spy
        # query_len 1 == reorder_batch_threshold, i.e. a pure decode batch.  The
        # real split_decodes_and_prefills then early-returns "all decodes",
        # which is what the stub does too -- so both paths reach the same code
        # in build() and these tests mean the same thing either way.
        self.cm = FakeCommonAttentionMetadata(seq_lens=[10, 12], query_lens=[1, 1])

    def test_falls_back_to_bound_buffer_when_argument_omitted(self):
        """Main forward path: upstream never passes the argument."""
        buf = torch.arange(MAX_NUM_REQS, dtype=torch.int32)
        self.builder.runner_num_prompt_tokens = buf

        self.builder.build(0, self.cm)

        got = self.captured["num_prompt_tokens"]
        self.assertIsNotNone(got)
        # sliced to the (padded) request count carried by the metadata
        self.assertEqual(got.tolist(), [0, 1])

    def test_explicit_argument_wins_over_bound_buffer(self):
        """MTP drafter path: patch_eagle passes its own slice."""
        self.builder.runner_num_prompt_tokens = torch.full(
            (MAX_NUM_REQS,), 99, dtype=torch.int32
        )
        explicit = torch.tensor([7, 8], dtype=torch.int32)

        self.builder.build(0, self.cm, num_prompt_tokens=explicit)

        self.assertEqual(self.captured["num_prompt_tokens"].tolist(), [7, 8])

    def test_stays_none_when_nothing_is_bound(self):
        self.builder.runner_num_prompt_tokens = None

        self.builder.build(0, self.cm)

        self.assertIsNone(self.captured["num_prompt_tokens"])

    def test_fallback_can_be_disabled(self):
        self.builder.runner_num_prompt_tokens = torch.full(
            (MAX_NUM_REQS,), 99, dtype=torch.int32
        )

        self.builder.build(0, self.cm, allow_runner_num_prompt_tokens=False)

        self.assertIsNone(self.captured["num_prompt_tokens"])


# --------------------------------------------------------------------------
# 3. graph capture must not pick the buffer up
# --------------------------------------------------------------------------


class TestCudagraphCapture(unittest.TestCase):
    def test_capture_never_sees_a_stale_buffer(self):
        """``build_for_cudagraph_capture`` calls back into ``build``.

        Without the explicit opt-out there, a buffer left over from the previous
        real step would be compared against the fabricated capture seq_lens and
        the outcome baked into the graph.
        """
        captured = {}
        builder = make_builder(num_spec=1)
        builder.runner_num_prompt_tokens = torch.full(
            (MAX_NUM_REQS,), 99, dtype=torch.int32
        )

        def spy(*args, **kwargs):
            captured["num_prompt_tokens"] = kwargs.get("num_prompt_tokens")
            captured["allow"] = kwargs.get("allow_runner_num_prompt_tokens")
            return "metadata"

        builder.build = spy
        cm = FakeCommonAttentionMetadata(seq_lens=[1, 1], query_lens=[1, 1])

        builder.build_for_cudagraph_capture(cm)

        self.assertIs(captured["allow"], False)
        self.assertIsNone(captured["num_prompt_tokens"])


# --------------------------------------------------------------------------
# 4. binding across the runner's three-level attn_groups structure
# --------------------------------------------------------------------------


class TestBindNumPromptTokens(unittest.TestCase):
    def test_reaches_every_mome_builder_and_skips_the_others(self):
        mome_a, mome_b = make_builder(), make_builder()
        other = SimpleNamespace()  # a non-MoME builder, e.g. MLA
        attn_groups = [
            [FakeAttnGroup([mome_a, mome_b])],  # two ubatches, one group
            [FakeAttnGroup([other])],
        ]
        buf = torch.arange(MAX_NUM_REQS, dtype=torch.int32)

        mome.bind_num_prompt_tokens(attn_groups, buf)

        self.assertIs(mome_a.runner_num_prompt_tokens, buf)
        self.assertIs(mome_b.runner_num_prompt_tokens, buf)
        self.assertFalse(hasattr(other, "runner_num_prompt_tokens"))

    def test_unbinds_with_none(self):
        builder = make_builder()
        attn_groups = [[FakeAttnGroup([builder])]]
        mome.bind_num_prompt_tokens(attn_groups, torch.zeros(4, dtype=torch.int32))

        mome.bind_num_prompt_tokens(attn_groups, None)

        self.assertIsNone(builder.runner_num_prompt_tokens)

    def test_empty_attn_groups_is_a_no_op(self):
        # self.attn_groups is [] until initialize_kv_cache runs, and _dummy_run
        # may fire before that during profiling.
        mome.bind_num_prompt_tokens([], None)

    def test_class_default_is_none(self):
        """A builder that was never bound must not inherit someone else's tensor."""
        self.assertIsNone(Builder.runner_num_prompt_tokens)


# --------------------------------------------------------------------------
# 5. the runner side, checked statically
# --------------------------------------------------------------------------


class TestRunnerHooks(unittest.TestCase):
    """``npu_model_runner.py`` pulls in half of vLLM, so it cannot be imported
    on a machine without vllm.  Parse it instead: what matters is that the two
    hooks exist in the two methods, because the failure mode when someone drops
    one of them is silent (stale or unfilled buffer, wrong block index, no
    exception).
    """

    @classmethod
    def setUpClass(cls):
        import ast

        source = (REPO_ROOT / "omni/worker/npu_model_runner.py").read_text(
            encoding="utf-8"
        )
        tree = ast.parse(source)

        def body_without_docstring(node):
            body = node.body
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                body = body[1:]
            return "".join(ast.dump(stmt) for stmt in body)

        cls.methods = {
            node.name: body_without_docstring(node)
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        cls.source = source

    def test_prepare_inputs_refreshes_the_buffer(self):
        self.assertIn("_prepare_inputs", self.methods)
        self.assertIn(
            "_refresh_mome_num_prompt_tokens", self.methods["_prepare_inputs"]
        )

    def test_refresh_fills_binds_and_zeroes_the_padding(self):
        body = self.methods["_refresh_mome_num_prompt_tokens"]
        self.assertIn("num_prompt_tokens", body)
        self.assertIn("copy_to_gpu", body)
        self.assertIn("bind_num_prompt_tokens", body)
        # padding rows must be zeroed, not left at the previous step's values
        self.assertIn("fill", body)

    def test_refresh_is_not_gated_on_use_spec_decode(self):
        """A prefill step has no drafts but the next step still needs this."""
        self.assertNotIn(
            "use_spec_decode", self.methods["_refresh_mome_num_prompt_tokens"]
        )

    def test_dummy_run_unbinds(self):
        self.assertIn("bind_num_prompt_tokens", self.methods["_dummy_run"])

    def test_dummy_run_unbinds_before_anything_else(self):
        """It has to precede the early return for mm_encoder_only."""
        body = self.methods["_dummy_run"]
        unbind = body.index("bind_num_prompt_tokens")
        early_return = body.index("mm_encoder_only")
        self.assertLess(unbind, early_return)

    def test_the_old_patch_file_is_gone(self):
        """It overrode _build_attention_metadata by copying the whole body."""
        stale = (
            REPO_ROOT
            / "omni/vllm_patches/patches/models/pangu_v2_hybrid/patch_num_prompt_tokens.py"
        )
        self.assertFalse(stale.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
