import importlib
import os
import pathlib
import sys
import types
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

class TestFusedMoe(unittest.TestCase):

    def setUp(self):
        self._install_mocks()
        self._import_module()

    def tearDown(self):
        self._cleanup_imports()
        self._module_patcher.stop()
        torch.zeros_like = self._original_zeros_like
        torch.empty_like = self._original_empty_like
        torch.distributed.all_to_all_single = self._original_all_to_all

    # ================= Helpers =================

    def _install_mocks(self):
        mock_torch_npu = types.ModuleType("torch_npu")

        class TensorWrapper(list):
            @property
            def device(self):
                return self[0].device

            @property
            def shape(self):
                return self[0].shape

            def to(self, *args, **kwargs):
                return TensorWrapper([self[0].to(*args, **kwargs)])

            def unsqueeze(self, dim):
                return self[0].unsqueeze(dim)

        def npu_moe_gating_top_k_softmax(gating_output, k):
            batch = gating_output.shape[0]
            weights = torch.full((batch, k), 1.0)
            ids = torch.arange(k, dtype=torch.int32).unsqueeze(0).repeat(batch, 1)
            row_idx = torch.arange(batch * k, dtype=torch.int32)
            return weights, ids, row_idx

        def npu_grouped_matmul(x, weight, **kwargs):
            base = x[0][0] if isinstance(x[0], TensorWrapper) else x[0]
            return TensorWrapper([base])

        def npu_swiglu(x):
            return x

        def npu_moe_finalize_routing(out, *args, **kwargs):
            if isinstance(out, (list, tuple)):
                return out[0]
            if isinstance(out, TensorWrapper):
                return out[0]
            return out

        def npu_moe_compute_expert_tokens(expert_idx, n):
            return torch.arange(n, dtype=torch.int32)

        def npu_moe_init_routing(hidden_states, row_idx, expert_idx, active_num):
            return hidden_states, row_idx.view(-1), expert_idx.view(-1)

        def npu_moe_init_routing_v2(hidden_states, expert_idx, scale=None, active_num=None, **kwargs):
            expanded_x_idx = torch.arange(expert_idx.numel(), dtype=torch.int32)
            expert_tokens = torch.ones(expert_idx.numel(), dtype=torch.int32)
            dynamic_scale = torch.ones(expert_idx.numel())
            return hidden_states, expanded_x_idx, expert_tokens, dynamic_scale

        def npu_dequant_swiglu_quant(gate_up_proj, weight_scale, activation_scale, **kwargs):
            activation_scale = activation_scale if torch.is_tensor(activation_scale) else torch.as_tensor(
                activation_scale)
            return gate_up_proj, activation_scale

        def npu_grouped_matmul_swiglu_quant_v2(sorted_tokens, weights, scales, pertoken_scale, expert_tokens):
            return TensorWrapper([sorted_tokens]), pertoken_scale

        def npu_dynamic_quant(x):
            return x, torch.ones(x.shape[0])

        def npu_moe_re_routing(gathered_tokens, tokens_per_expert_group, per_token_scales=None):
            idxs = torch.arange(gathered_tokens.shape[0], dtype=torch.int32)
            tokens_per_local_expert = tokens_per_expert_group.view(-1)
            return gathered_tokens, per_token_scales, idxs, tokens_per_local_expert

        def npu_grouped_matmul_finalize_routing(*args, **kwargs):
            return args[0]

        def npu_moe_distribute_dispatch_v2(**kwargs):
            x = kwargs["x"]
            expand_x = torch.zeros_like(x)
            dynamic_scale = torch.ones(x.shape[0])
            expand_idx = torch.arange(x.shape[0], dtype=torch.int32)
            expert_token_nums = torch.ones(1, dtype=torch.int64)
            ep_recv_counts = torch.ones(1, dtype=torch.int64)
            tp_recv_counts = torch.ones(1, dtype=torch.int64)
            return expand_x, dynamic_scale, expand_idx, expert_token_nums, ep_recv_counts, tp_recv_counts

        def npu_moe_distribute_combine_v2(**kwargs):
            return kwargs["expand_x"]

        mock_torch_npu.npu_moe_gating_top_k_softmax = npu_moe_gating_top_k_softmax
        mock_torch_npu.npu_grouped_matmul = npu_grouped_matmul
        mock_torch_npu.npu_swiglu = npu_swiglu
        mock_torch_npu.npu_moe_finalize_routing = npu_moe_finalize_routing
        mock_torch_npu.npu_moe_compute_expert_tokens = npu_moe_compute_expert_tokens
        mock_torch_npu.npu_moe_init_routing = npu_moe_init_routing
        mock_torch_npu.npu_moe_init_routing_v2 = npu_moe_init_routing_v2
        mock_torch_npu.npu_dequant_swiglu_quant = npu_dequant_swiglu_quant
        mock_torch_npu.npu_grouped_matmul_swiglu_quant_v2 = npu_grouped_matmul_swiglu_quant_v2
        mock_torch_npu.npu_dynamic_quant = npu_dynamic_quant
        mock_torch_npu.npu_moe_re_routing = npu_moe_re_routing
        mock_torch_npu.npu_grouped_matmul_finalize_routing = npu_grouped_matmul_finalize_routing
        mock_torch_npu.npu_moe_distribute_dispatch_v2 = npu_moe_distribute_dispatch_v2
        mock_torch_npu.npu_moe_distribute_combine_v2 = npu_moe_distribute_combine_v2
        mock_torch_npu.npu_prefetch = lambda *args, **kwargs: None

        fake_ep_group = SimpleNamespace(world_size=2, rank_in_group=0, device_group=None)
        fake_world_group = SimpleNamespace(world_size=2, rank_in_group=0)
        fake_dp_group = SimpleNamespace(world_size=1)

        vllm_platforms = types.ModuleType("vllm.platforms")
        vllm_platforms.current_platform = SimpleNamespace(device_type="cpu")
        vllm_mod = types.ModuleType("vllm")

        def get_ep_group():
            return fake_ep_group

        def get_world_group():
            return fake_world_group

        def get_dp_group():
            return fake_dp_group

        vllm_distributed = types.ModuleType("vllm.distributed")
        vllm_distributed.get_ep_group = get_ep_group
        vllm_distributed.get_world_group = get_world_group
        vllm_distributed.get_dp_group = get_dp_group

        forward_context = types.ModuleType("vllm.forward_context")
        forward_context.get_forward_context = lambda: SimpleNamespace(attn_metadata=None)
        vllm_mod.platforms = vllm_platforms
        vllm_mod.distributed = vllm_distributed
        vllm_mod.forward_context = forward_context

        operator_opt_config = SimpleNamespace(
            moe_multi_stream_tune=False,
            experts_pruning=False,
            enable_gmm_swiglu_quant=False,
            cast_w2_scale_f32=False,
            gmm_nz=False,
            new_w4_op=False,
            attn_prefetch=0,
            prefill_enable_long_seq=False,
            enable_kv_rmsnorm_rope_cache=False,
            shared_experts_to_gmm=False,
            use_dcp=False
        )
        parall_config = SimpleNamespace(redundancy_shared_expert_num=0, attn_dies=0, o_proj_tp_size=1)
        task_config = SimpleNamespace(enable_omni_placement=False, enable_attn_ffn_disaggregation=False,
                                      decode_gear_list=[16])
        model_extra = SimpleNamespace(operator_opt_config=operator_opt_config, parall_config=parall_config,
                                      task_config=task_config)
        loader_mod = types.ModuleType("loader")
        loader_mod.model_extra_config = model_extra

        utils_mod = types.ModuleType("utils")

        class ConditionalTNGScope:
            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        utils_mod.ConditionalTNGScope = ConditionalTNGScope

        tng_scope = types.SimpleNamespace(npu_stream_switch=lambda *args, **kwargs: ConditionalTNGScope())
        torchair_mod = types.ModuleType("torchair")
        torchair_mod.scope = tng_scope

        self._original_zeros_like = torch.zeros_like
        self._original_empty_like = torch.empty_like
        self._original_all_to_all = torch.distributed.all_to_all_single

        def _unwrap_tensor(x):
            while isinstance(x, TensorWrapper):
                x = x[0]
            return x

        torch.zeros_like = lambda inp, *a, **kw: self._original_zeros_like(_unwrap_tensor(inp), *a, **kw)
        torch.empty_like = lambda inp, *a, **kw: self._original_empty_like(_unwrap_tensor(inp), *a, **kw)
        torch.distributed.all_to_all_single = lambda output, input, *args, **kwargs: output.copy_(
            input if isinstance(input, torch.Tensor) else list(input)[0])

        os.environ.setdefault("ROLE", "prefill")

        mock_modules = {
            "torch_npu": mock_torch_npu,
            "vllm.platforms": vllm_platforms,
            "vllm.distributed": vllm_distributed,
            "vllm.forward_context": forward_context,
            "vllm": vllm_mod,
            "omni.models.config_loader.loader": loader_mod,
            "omni.layers.utils": utils_mod,
            "torchair": torchair_mod,
        }
        self._module_patcher = mock.patch.dict(sys.modules, mock_modules)
        self._module_patcher.start()

    def _import_module(self):
        from pathlib import Path

        module_path = Path(__file__).resolve().parents[3] / "omni" / "layers" / "moe" / "fused_moe" / "fused_moe.py"
        module_name = "fused_moe_under_test"
        if module_name in sys.modules:
            del sys.modules[module_name]
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)  # type: ignore[arg-type]
        self.fused_module = module

    def _cleanup_imports(self):
        if "fused_moe_under_test" in sys.modules:
            del sys.modules["fused_moe_under_test"]

    # ================= Tests =================

    def test_fused_topk(self):
        gating = torch.randn(2, 4)
        weights, ids, rows = self.fused_module.fused_topk(gating, topk=2, renormalize=True)
        self.assertEqual(weights.shape, (2, 2))
        self.assertTrue(torch.allclose(weights.sum(dim=-1), torch.ones(2)))
        self.assertEqual(ids.shape, (2, 2))
        self.assertEqual(rows.numel(), 4)

    def test_grouped_topk_softmax(self):
        hidden = torch.randn(3, 2)
        gating = torch.tensor([[1.0, 0.0, 0.5, -0.5, 2.0, 1.0]])
        weights, ids, rows = self.fused_module.grouped_topk(
            hidden, gating, topk=2, renormalize=True, num_expert_group=2, topk_group=1
        )
        self.assertEqual(weights.shape[1], 2)
        self.assertEqual(ids.shape, weights.shape)
        self.assertEqual(rows.numel(), ids.numel())

    def test_fused_experts_allgather_ep_warmup(self):
        hidden_states = torch.ones(2, 3)
        w1 = torch.ones(1, 3, 3)
        w2 = torch.ones(1, 3, 3)
        topk_weights = torch.ones(2, 1)
        topk_ids = torch.zeros(2, 1, dtype=torch.int32)
        row_idx = torch.zeros(2, 1, dtype=torch.int32)
        output = self.fused_module.fused_experts_allgather_ep(
            hidden_states, w1, w2, topk_weights, topk_ids, row_idx, warm_up=True, n_routed_experts=1,
            local_expert_indices=[0]
        )
        self.assertEqual(output.shape, hidden_states.shape)

    def test_fused_experts_alltoall_ep_warmup(self):
        hidden_states = torch.ones(2, 2)
        w1 = torch.ones(1, 2, 2)
        w2 = torch.ones(1, 2, 2)
        topk_weights = torch.ones(2, 1)
        topk_ids = torch.zeros(2, 1, dtype=torch.int32)
        row_idx = torch.zeros(2, 1, dtype=torch.int32)

        with mock.patch.object(torch.distributed, "all_to_all_single", wraps=torch.distributed.all_to_all_single) as a2a:
            output = self.fused_module.fused_experts_alltoall_ep(
                hidden_states, w1, w2, topk_weights, topk_ids, row_idx, warm_up=True
            )
        self.assertEqual(output.shape, hidden_states.shape)
        self.assertTrue(a2a.called)

    def test_fused_experts_ep_best_alltoall(self):
        hidden_states = torch.ones(2, 2)
        w1 = torch.ones(1, 2, 2)
        w2 = torch.ones(1, 2, 2)
        topk_weights = torch.ones(2, 1)
        topk_ids = torch.zeros(2, 1, dtype=torch.int32)
        row_idx = torch.zeros(2, 1, dtype=torch.int32)

        with mock.patch.object(self.fused_module.torch_npu, "npu_grouped_matmul", side_effect=lambda x, weight, **kwargs: x[0]):
            output = self.fused_module.fused_experts_ep_best_alltoall(
                hidden_states, w1, w2, topk_weights, topk_ids, row_idx
            )
        self.assertEqual(output.shape, hidden_states.shape)

    def test_fused_experts_allgather_ep_a3_prefill(self):
        class Layer:
            def __init__(self):
                self.weight_num_bits = 8
                self.w13_weight = torch.ones(1, 3, 3)
                self.w13_weight_scale = torch.ones(1, 3)
                self.w2_weight = torch.ones(1, 3, 3)
                self.w2_weight_scale = torch.ones(1, 3)
                self.moe_layer_idx = 0
                self.planner = SimpleNamespace(record_activation=lambda *args, **kwargs: None)

        layer = Layer()
        hidden_states = torch.ones(2, 3)
        pertoken_scale = torch.ones(2)
        topk_weights = torch.ones(2, 1)
        topk_ids = torch.zeros(2, 1, dtype=torch.int32)
        output = self.fused_module.fused_experts_allgather_ep_a3(
            layer, hidden_states, pertoken_scale, topk_weights, topk_ids, n_routed_experts=1, is_prefill=True,
            max_num_deployed_expert_per_rank=1
        )
        self.assertEqual(output.shape[0], hidden_states.shape[0])

    def test_gmm_expert_weight8(self):
        class Layer:
            def __init__(self):
                self.weight_num_bits = 8
                self.w13_weight = torch.ones(1, 2, 2)
                self.w13_weight_scale = torch.ones(1, 2)
                self.w2_weight = torch.ones(1, 2, 2)
                self.w2_weight_scale = torch.ones(1, 2)

        layer = Layer()
        x = torch.ones(1, 2)
        expert_tokens = torch.ones(1, dtype=torch.int64)
        pertoken_scale = torch.ones(1)
        out = self.fused_module.gmm_expert(layer, x, expert_tokens, dynamic_scale=pertoken_scale)
        self.assertEqual(out.shape[0], x.shape[0])

    def test_moe_infer_fusion(self):
        class Layer:
            def __init__(self):
                self.w13_weight = torch.ones(1, 2, 2)
                self.w2_weight = torch.ones(1, 2, 2)
                self.weight_num_bits = 8
                self.w13_weight_scale = torch.ones(1, 2)
                self.w2_weight_scale = torch.ones(1, 2)
                self.planner = SimpleNamespace(record_activation=lambda *args, **kwargs: None)

        layer = Layer()
        x = torch.ones(2, 2)
        topk_ids = torch.zeros(2, 1, dtype=torch.int32)
        topk_weight = torch.ones(2, 1)
        hidden_states, gathered, weights, row_idx = self.fused_module.moe_infer_fusion(
            layer, x, topk_ids, topk_weight, warm_up=False, is_prefill=True, comm_group=None
        )
        self.assertEqual(hidden_states.shape, x.shape)
        self.assertEqual(gathered.shape[1], x.shape[1])
        self.assertEqual(weights.shape[0], x.shape[0])
        self.assertEqual(row_idx.shape[0], x.shape[0])

    def test_shared_expert_quant_forward(self):
        class Gate:
            def __init__(self):
                self.weight = torch.ones(1, 2, 2)
                self.weight_scale = torch.ones(2)

        class Down:
            def __init__(self):
                self.weight = torch.ones(1, 2, 2)
                self.weight_scale = torch.ones(2)

        class Layer:
            def __init__(self):
                self.gate_up_proj = Gate()
                self.down_proj = Down()

        layer = Layer()
        sorted_tokens = torch.ones(1, 2)
        expert_tokens = torch.ones(1, dtype=torch.int64)
        out = self.fused_module.shared_expert_quant_forward(
            layer, sorted_tokens, expert_tokens, torch.bfloat16, dynamic_scale=torch.ones(1)
        )
        self.assertEqual(out.shape[0], sorted_tokens.shape[0])

    def test_moe_expert_quant_forward_weight8(self):
        class Layer:
            def __init__(self):
                self.quant_mode = True
                self.weight_num_bits = 8
                self.w13_weight = torch.ones(1, 2, 2)
                self.w13_weight_scale = torch.ones(1, 2)
                self.w2_weight = torch.ones(1, 2, 2)
                self.w2_weight_scale = torch.ones(1, 2)

        layer = Layer()
        sorted_tokens = torch.ones(1, 2)
        expert_tokens = torch.ones(1, dtype=torch.int64)
        out = self.fused_module.moe_expert_quant_forward(
            layer, sorted_tokens, expert_tokens, torch.bfloat16, dynamic_scale=torch.ones(1)
        )
        self.assertEqual(out.shape[0], sorted_tokens.shape[0])

    def test_set_fake_expand_x_and_speculative(self):
        self.fused_module.fake_expand_x.clear()
        self.fused_module.set_num_speculative_tokens(1)
        self.fused_module.set_fake_expand_x(2, [2, 2])
        self.assertIn(2, self.fused_module.fake_expand_x)
        self.fused_module.set_fake_expand_x(4, [4, 2])
        self.assertIn(4, self.fused_module.fake_expand_x)

    def test_fused_experts_moe_dispatch_combine(self):
        class Layer:
            def __init__(self):
                self.tp_size = 1
                self.quant_mode = True
                self.moe_all_to_all_group_name = "g1"
                self.moe_rs_group_name = "g2"
                self.w13_weight = torch.ones(1, 2, 2)
                self.w13_weight_scale = torch.ones(1, 2)
                self.w2_weight = torch.ones(1, 2, 2)
                self.w2_weight_scale = torch.ones(1, 2)
                self.moe_layer_idx = 0
                self.planner = SimpleNamespace(record_activation=lambda *args, **kwargs: None)

        layer = Layer()
        hidden_states = torch.ones(2, 2)
        topk_weights = torch.ones(2, 1)
        topk_ids = torch.zeros(2, 1, dtype=torch.int32)

        with mock.patch.object(self.fused_module, "moe_expert_quant_forward", side_effect=lambda *a, **kw: a[1]):
            output = self.fused_module.fused_experts_moe_dispatch_combine(
                layer, hidden_states, topk_weights, topk_ids, max_num_deployed_expert=2, is_prefill=True,
                is_route_expert=True
            )
        self.assertEqual(output.shape, hidden_states.shape)

    def test_static_routing(self):
        hidden_states = torch.randn(4, 2)
        indices = self.fused_module.static_routing(hidden_states)
        self.assertEqual(len(indices), hidden_states.shape[0])

    def test_shared_expert_alltoall_ep(self):
        def fake_all_to_all_single(output, input, **kwargs):
            output.copy_(input)

        with mock.patch.object(torch.distributed, "all_to_all_single", side_effect=fake_all_to_all_single):
            hidden_states = torch.ones(2, 2)
            expert = torch.nn.Linear(2, 2, bias=False)
            torch.nn.init.constant_(expert.weight, 1.0)
            output = self.fused_module.shared_expert_alltoall_ep(hidden_states, expert, warm_up=False)
        self.assertEqual(output.shape, hidden_states.shape)

    def test_fused_experts_allgather_ep_a2(self):
        class Layer:
            def __init__(self):
                self.weight_num_bits = 8
                self.w13_weight = torch.ones(1, 2, 2)
                self.w13_weight_scale = torch.ones(1, 2)
                self.w2_weight = torch.ones(1, 2, 2)
                self.w2_weight_scale = torch.ones(1, 2)
                self.moe_layer_idx = 0
                self.planner = SimpleNamespace(record_activation=lambda *args, **kwargs: None)

        layer = Layer()
        hidden_states = torch.ones(2, 2)
        pertoken_scale = torch.ones(2)
        topk_weights = torch.ones(2, 1)
        topk_ids = torch.zeros(2, 1, dtype=torch.int32)
        smooth_scale = torch.ones(1, 1)
        output = self.fused_module.fused_experts_allgather_ep_a2(
            layer, hidden_states, pertoken_scale, topk_weights, topk_ids, n_routed_experts=1, is_prefill=True,
            max_num_deployed_expert_per_rank=1, smooth_scale=smooth_scale
        )
        self.assertEqual(output.shape[0], hidden_states.shape[0])


if __name__ == "__main__":
    unittest.main()