import pytest
import unittest
from unittest import mock
from unittest.mock import Mock, patch, MagicMock
import importlib

import torch
from torch import nn
from torch.nn import Parameter

from omni.layers.linear import *
import omni.layers.linear as omni_linear_mod


class TestLinear(unittest.TestCase):
    def setUp(self):
        return super().setUp()

    def tearDown(self):
        return super().tearDown()

# ================= AscendUnquantizedLinearMethod =================

    class DummyLayer(nn.Module):

        def __init__(self, in_features: int = 4, out_features: int = 3,
                    dtype: torch.dtype = torch.float32):
            super().__init__()
            w = torch.randn(out_features, in_features, dtype=dtype)
            self.weight = nn.Parameter(w)

    def test_ascend_unquantized_linear_method_process_weights_flag_true_casts_and_replaces_weight(self):
        layer = TestLinear.DummyLayer()
        original_weight = layer.weight
        cast_weight = torch.randn_like(original_weight.data)

        with patch.object(model_extra_config.operator_opt_config,
                           "unquant_bmm_nz", True):
            with patch("omni.layers.linear.torch_npu.npu_format_cast",
                       return_value=cast_weight) as mock_cast:
                AscendUnquantizedLinearMethod().process_weights_after_loading(layer)

        mock_cast.assert_called_once()

        (arg_weight, arg_format), kwargs = mock_cast.call_args

        self.assertEqual(arg_format, 29)
        self.assertIs(layer.weight, original_weight)
        self.assertIsInstance(layer.weight, Parameter)
        self.assertTrue(torch.equal(layer.weight.data, cast_weight))

# ========== AscendMergedColumnParallelLinear: __init__ / forward / weight_loader ==========

    # ---------- helpers ----------

    def _make_ascend_merged_column_parallel_linear_layer(self,
                    input_size=4,
                    output_sizes=None,
                    bias=True,
                    gather_output=False,
                    skip_bias_add=False,
                    tp_size=1,
                    tp_rank=0):
        if output_sizes is None:
            output_sizes = [8, 8]
        return AscendMergedColumnParallelLinear(
            input_size=input_size,
            output_sizes=output_sizes,
            bias=bias,
            gather_output=gather_output,
            skip_bias_add=skip_bias_add,
            tp_size=tp_size,
            tp_rank=tp_rank,
            params_dtype=torch.float32,
            quant_config=None,
        )

    # ---------- __init__ ----------

    def test_ascend_merged_column_parallel_linear_init_valid_divisible_output_sizes_with_bias(self):
        input_size = 4
        output_sizes = [6, 10]  # both divisible by tp_size=2
        tp_size = 2

        layer = self._make_ascend_merged_column_parallel_linear_layer(
            input_size=input_size,
            output_sizes=output_sizes,
            bias=True,
            gather_output=False,
            skip_bias_add=False,
            tp_size=tp_size,
            tp_rank=0,
        )

        self.assertEqual(layer.output_sizes, output_sizes)
        self.assertEqual(layer.tp_size, tp_size)
        # summed output size
        self.assertEqual(layer.output_size, sum(output_sizes))
        # bias param created，shape = output_size_per_partition
        self.assertIsInstance(layer.bias, Parameter)
        self.assertEqual(layer.bias.shape[0], layer.output_size_per_partition)

    def test_ascend_merged_column_parallel_linear_init_invalid_output_sizes_not_divisible_raises(self):
        # 5 % 4 != 0
        with self.assertRaisesRegex(RuntimeError, "All output_sizes must be divisible by tp_size"):
            self._make_ascend_merged_column_parallel_linear_layer(
                input_size=4,
                output_sizes=[5, 6],
                bias=True,
                gather_output=False,
                skip_bias_add=False,
                tp_size=4,
                tp_rank=0,
            )

    # ---------- forward ----------

    def test_ascend_merged_column_parallel_linear_forward_no_gather_with_skip_bias_add_returns_tensor_and_bias(self):
        layer = self._make_ascend_merged_column_parallel_linear_layer(
            input_size=4,
            output_sizes=[6, 10],
            bias=True,
            gather_output=False,
            skip_bias_add=True,
            tp_size=1,
            tp_rank=0,
        )

        x = torch.randn(2, 4)
        output, output_bias = layer(x)

        self.assertIsInstance(output, torch.Tensor)
        self.assertIs(output_bias, layer.bias)

    def test_ascend_merged_column_parallel_linear_forward_gather_output_tensor_all_gather_called(self):
        layer = self._make_ascend_merged_column_parallel_linear_layer(
            input_size=4,
            output_sizes=[8, 8],
            bias=True,
            gather_output=True,
            skip_bias_add=False,
        )

        input_ = torch.randn(2, 4)
        output_parallel = torch.randn(2, layer.output_size_per_partition)
        layer.quant_method.apply = MagicMock(return_value=output_parallel)

        with patch("omni.layers.linear.get_mlp_tp_group") as mock_get_group:
            mock_group = MagicMock()
            mock_get_group.return_value = mock_group
            gathered = torch.randn(2, layer.output_size_per_partition)
            mock_group.all_gather.return_value = gathered

            output, output_bias = layer(input_)

        mock_get_group.assert_called_once()
        mock_group.all_gather.assert_called_once()
        args, kwargs = mock_group.all_gather.call_args
        self.assertIs(args[0], output_parallel)
        self.assertIn("dim", kwargs)
        self.assertEqual(kwargs["dim"], -1)
        self.assertIs(output, gathered)
        self.assertIsNone(output_bias)

    # ---------- weight_loader: GGUF 分支 ----------

    def test_ascend_merged_column_parallel_linear_weight_loader_gguf_weight_second_shard_materializes_qweight(self):
        layer = self._make_ascend_merged_column_parallel_linear_layer(output_sizes=[6, 6], tp_size=2, tp_rank=0)

        param = Parameter(torch.empty(0))
        param.is_gguf_weight = True
        param.output_dim = 1
        param.shard_id = []
        param.shard_id_map = {}
        param.data_container = []
        param.materialize_nested = MagicMock(return_value="nested_qweight")

        loaded_weight = torch.randn(4, 6)

        layer.weight_loader(param, loaded_weight, loaded_shard_id=0)
        layer.weight_loader(param, loaded_weight, loaded_shard_id=1)

        self.assertEqual(param.shard_id, [0, 1])
        self.assertEqual(len(param.data_container), 2)
        param.materialize_nested.assert_called_once()
        self.assertEqual(layer.qweight, "nested_qweight")

    # ---------- weight_loader: fused, loaded_shard_id is None ----------

    def test_ascend_merged_column_parallel_linear_weight_loader_fused_no_output_dim_needs_scalar_to_array(self):
        layer = self._make_ascend_merged_column_parallel_linear_layer(output_sizes=[4, 4])

        param = Parameter(torch.zeros(1))
        param.needs_scalar_to_array = True
        loaded_weight = torch.tensor(5.0)

        with patch("omni.layers.linear.adjust_scalar_to_fused_array") as mock_adjust:
            # 返回 (param_data, 替换后的 loaded_weight)
            new_loaded = torch.full_like(param.data, 2.0)
            mock_adjust.return_value = (param.data, new_loaded)

            layer.weight_loader(param, loaded_weight, loaded_shard_id=None)

        mock_adjust.assert_called_once()
        args, _ = mock_adjust.call_args
        self.assertTrue(torch.equal(args[0], param.data))
        self.assertTrue(torch.equal(param.data, new_loaded))

    def test_ascend_merged_column_parallel_linear_weight_loader_fused_with_output_dim_unpacked_recursive_shards(self):
        # output_sizes = [4, 2] => 总宽度 6
        layer = self._make_ascend_merged_column_parallel_linear_layer(output_sizes=[4, 2], tp_size=1)

        param = Parameter(torch.zeros(3, 6))
        param.output_dim = 1
        loaded_weight = torch.arange(18, dtype=torch.float32).reshape(3, 6)

        calls = []
        orig_loader = layer.weight_loader

        def spy_loader(p, lw, loaded_shard_id=None):
            if loaded_shard_id is not None:
                calls.append((loaded_shard_id, lw.clone()))
                return
            return orig_loader(p, lw, loaded_shard_id)

        with patch.object(layer, "weight_loader", side_effect=spy_loader):
            layer.weight_loader(param, loaded_weight, loaded_shard_id=None)

        # 应该为两个分片：宽度 4 和 2
        self.assertEqual(len(calls), 2)
        calls.sort(key=lambda x: x[0])
        shard0_id, shard0_weight = calls[0]
        shard1_id, shard1_weight = calls[1]

        self.assertEqual(shard0_id, 0)
        self.assertEqual(shard1_id, 1)
        self.assertEqual(shard0_weight.shape, (3, 4))
        self.assertEqual(shard1_weight.shape, (3, 2))

    # ---------- weight_loader: sharded, loaded_shard_id is not None & output_dim is not None ----------

    def test_ascend_merged_column_parallel_linear_weight_loader_sharded_with_output_dim_basic_slice_by_tp_rank(self):
        # 两个分片，每个宽 4，tp_size=2 => 每个 shard_size=2
        output_sizes = [4, 4]
        tp_size = 2
        tp_rank = 1  # 取第二个 TP 分片
        layer = self._make_ascend_merged_column_parallel_linear_layer(output_sizes=output_sizes, tp_size=tp_size, tp_rank=tp_rank)

        param = Parameter(torch.zeros(3, 4))
        param.output_dim = 1

        loaded_weight = torch.arange(12, dtype=torch.float32).reshape(3, 4)

        # 选 shard_id=1，对第二段输出 4 做 TP 分片
        layer.weight_loader(param, loaded_weight, loaded_shard_id=1)

        # shard_offset = sum([4]) // 2 = 2
        # shard_size   = 4 // 2 = 2
        # tp_rank=1 => start_idx = 1*2=2 => 取 loaded_weight[:,2:4]
        expected = loaded_weight[:, 2:4]
        self.assertTrue(torch.allclose(param.data[:, 2:4], expected))

# ================= AscendRowParallelLinear =================

    # ---------- helpers ----------

    def _make_ascend_row_parallel_linear_layer(
        self,
        input_size=16,
        output_size=8,
        tp_size=2,
        tp_rank=0,
        bias=True,
        input_is_parallel=True,
        skip_bias_add=False,
        reduce_results=True,
        params_dtype=torch.float32,
    ):
        # 默认 quant_config=None，依赖 LinearBase 内部创建 UnquantizedLinearMethod
        return AscendRowParallelLinear(
            input_size=input_size,
            output_size=output_size,
            tp_size=tp_size,
            tp_rank=tp_rank,
            bias=bias,
            input_is_parallel=input_is_parallel,
            skip_bias_add=skip_bias_add,
            params_dtype=params_dtype,
            reduce_results=reduce_results,
            quant_config=None,
        )

    # -------------------- __init__ 分支 --------------------

    def test_ascend_row_parallel_linear_init_invalid_reduce_results_with_bias_without_skip_bias_add_raises(self):
        # not reduce_results 且 bias=True 且 skip_bias_add=False -> ValueError
        with self.assertRaises(ValueError):
            self._make_ascend_row_parallel_linear_layer(
                input_size=8,
                output_size=4,
                tp_size=2,
                tp_rank=0,
                bias=True,
                input_is_parallel=True,
                skip_bias_add=False,
                reduce_results=False,
            )


    # -------------------- weight_loader 分支 --------------------

    def test_ascend_row_parallel_linear_weight_loader_input_dim_not_none_no_bitsandbytes_slices_by_tp_rank(self):
        layer = self._make_ascend_row_parallel_linear_layer(input_size=8, output_size=6, tp_size=2, tp_rank=1)
        param = Parameter(torch.zeros(2, 4, dtype=torch.float32))
        # 指定 input_dim，触发 narrow
        param.input_dim = 1
        # loaded_weight 的 dim=1 是 8，按 tp_size=2 拆成两段 4
        loaded_weight = torch.stack(
            [
                torch.arange(0, 8, dtype=torch.float32),
                torch.arange(8, 16, dtype=torch.float32),
            ],
            dim=0,
        )  # [2, 8]

        layer.weight_loader(param, loaded_weight)

        expected = loaded_weight.narrow(1, 4, 4)
        self.assertTrue(torch.allclose(param.data, expected))

    def test_ascend_row_parallel_linear_weight_loader_input_dim_not_none_with_bitsandbytes_skips_narrow(self):
        layer = self._make_ascend_row_parallel_linear_layer()
        param = Parameter(torch.zeros(2, 8, dtype=torch.float32))
        param.input_dim = 1
        param.use_bitsandbytes_4bit = True  # 跳过 narrow

        loaded_weight = torch.randn(2, 8, dtype=torch.float32)

        layer.weight_loader(param, loaded_weight)

        # 因为跳过 narrow，所以完整拷贝
        self.assertTrue(torch.allclose(param.data, loaded_weight))

    def test_ascend_row_parallel_linear_weight_loader_gguf_uninitialized_materialize_with_input_dim_divided_by_tp_size(self):
        tp_size = 2
        layer = self._make_ascend_row_parallel_linear_layer(tp_size=tp_size, tp_rank=0)

        param = UninitializedParameter(requires_grad=False)
        param.is_gguf_weight = True
        param.input_dim = 1  # 对该维度按 tp_size 除
        loaded_weight = torch.randn(4, 8, dtype=torch.float32)

        layer.weight_loader(param, loaded_weight)

        # 期望：materialize 的形状在 input_dim 维度被除以 tp_size
        self.assertEqual(tuple(param.data.shape), (4, 4))
        # 由于 tp_rank=0，应拿到前一半
        expected = loaded_weight[:, :4]
        self.assertTrue(torch.allclose(param.data, expected))

    # -------------------- forward 分支 --------------------

    def test_ascend_row_parallel_linear_forward_input_is_parallel_false_slices_input_by_tp_rank(self):
        layer_mod = importlib.import_module(AscendRowParallelLinear.__module__)

        class CaptureQuantMethod:
            def __init__(self):
                self.last_input = None
                self.last_bias = None

            def create_weights(
                self,
                layer,
                input_size_per_partition,
                output_partition_sizes,
                input_size,
                output_size,
                params_dtype,
                weight_loader,
            ):
                return

            def apply(self, layer, input_parallel, bias=None):
                self.last_input = input_parallel
                self.last_bias = bias
                # 返回形状正确即可，避免真实 GEMM
                return torch.zeros(input_parallel.shape[0], layer.output_size, dtype=input_parallel.dtype)

        class FailingTpGroup:
            def all_reduce(self, tensor):
                raise AssertionError("all_reduce should not be called when reduce_results=False")

        tp_size = 2
        tp_rank = 1
        input_size = 8
        output_size = 6

        with mock.patch.object(layer_mod, "get_mlp_tp_group", return_value=FailingTpGroup()):
            # reduce_results=False 时必须 bias=False，否则你 __init__ 的约束会 ValueError
            layer = self._make_ascend_row_parallel_linear_layer(
                input_size=input_size,
                output_size=output_size,
                tp_size=tp_size,
                tp_rank=tp_rank,
                bias=False,
                input_is_parallel=False,   # ✅ 关键：覆盖这个分支
                skip_bias_add=False,
                reduce_results=False,
            )
            qm = CaptureQuantMethod()
            layer.quant_method = qm

            # 构造可验证切片的输入
            # shape [B, 8]，tp_size=2 => 每份 4，tp_rank=1 => 取后半段 [:, 4:8]
            x = torch.arange(0, 24, dtype=torch.float32).reshape(3, 8)

            out, out_bias = layer(x)

        # 1) forward 输出形状契约
        self.assertEqual(out.shape, (3, output_size))
        self.assertIsNone(out_bias)

        # 2) ✅ 核心断言：apply 收到的 input_parallel 是按 tp_rank 切出来的那一段
        expected = x.narrow(-1, 4, 4)  # start=tp_rank*(input_size/tp_size)=1*4
        self.assertIsNotNone(qm.last_input)
        self.assertEqual(tuple(qm.last_input.shape), (3, 4))
        self.assertTrue(torch.allclose(qm.last_input, expected))


    def test_ascend_row_parallel_linear_forward_input_is_parallel_true_no_reduce_tp1_no_skip_bias_rank0(self):
        # tp_size=1 时不应调用 all_reduce
        layer_mod = importlib.import_module(AscendRowParallelLinear.__module__)

        class DummyTpGroup:
            def all_reduce(self, x):
                raise AssertionError("all_reduce should not be called when tp_size == 1")

        with mock.patch.object(layer_mod, "get_mlp_tp_group", return_value=DummyTpGroup()):
            layer = self._make_ascend_row_parallel_linear_layer(
                input_size=8,
                output_size=4,
                tp_size=1,
                tp_rank=0,
                input_is_parallel=True,
                skip_bias_add=False,
                reduce_results=True,
            )

            batch_size = 3
            x = torch.randn(batch_size, 8, dtype=torch.float32)
            out, out_bias = layer(x)

        self.assertEqual(out.shape, (batch_size, 4))
        self.assertIsNone(out_bias)

    def test_ascend_row_parallel_linear_forward_bias_fused_only_on_rank0_when_not_skip_bias(self):
        layer_mod = importlib.import_module(AscendRowParallelLinear.__module__)

        class CaptureQuantMethod:
            def __init__(self):
                self.last_bias = None

            def create_weights(
                self,
                layer,
                input_size_per_partition,
                output_partition_sizes,
                input_size,
                output_size,
                params_dtype,
                weight_loader,
            ):
                return

            def apply(self, layer, input_parallel, bias=None):
                self.last_bias = bias
                return torch.zeros(input_parallel.shape[0], layer.output_size, dtype=input_parallel.dtype)

        class DummyTpGroup:
            def all_reduce(self, x):
                return x

        with mock.patch.object(layer_mod, "get_mlp_tp_group", return_value=DummyTpGroup()):
            # rank 0
            layer0 = self._make_ascend_row_parallel_linear_layer(
                tp_size=2,
                tp_rank=0,
                skip_bias_add=False,
                reduce_results=True, 
            )
            qm0 = CaptureQuantMethod()
            layer0.quant_method = qm0

            x = torch.randn(2, layer0.input_size, dtype=torch.float32)
            out0, bias_out0 = layer0(x)

            assert qm0.last_bias is layer0.bias   # 只在 rank0 融合 bias
            assert bias_out0 is None              # skip_bias_add=False -> 第二输出 None

            # rank 1
            layer1 = self._make_ascend_row_parallel_linear_layer(
                tp_size=2,
                tp_rank=1,
                skip_bias_add=False,
                reduce_results=True,   
            )
            qm1 = CaptureQuantMethod()
            layer1.quant_method = qm1

            out1, bias_out1 = layer1(x)

            assert qm1.last_bias is None          # 非 0 rank 不融合 bias
            assert bias_out1 is None

    def test_ascend_row_parallel_linear_forward_skip_bias_add_true_returns_output_and_output_bias(self):
        class DummyQuantMethod:
            def create_weights(
                self,
                layer,
                input_size_per_partition,
                output_partition_sizes,
                input_size,
                output_size,
                params_dtype,
                weight_loader,
            ):
                return

            def apply(self, layer, input_parallel, bias=None):
                # 不使用 bias，随便返回一个形状对的 tensor
                return torch.zeros(input_parallel.shape[0], layer.output_size, dtype=input_parallel.dtype)

        layer = self._make_ascend_row_parallel_linear_layer(
            tp_size=2,
            tp_rank=0,
            skip_bias_add=True,
            reduce_results=False,
        )
        dummy_qm = DummyQuantMethod()
        layer.quant_method = dummy_qm

        x = torch.randn(3, layer.input_size, dtype=torch.float32)
        out, out_bias = layer(x)

        self.assertEqual(out.shape, (3, layer.output_size))
        self.assertIs(out_bias, layer.bias)  # skip_bias_add=True -> 返回 bias 作为第二个输出

    def test_ascend_row_parallel_linear_forward_reduce_results_true_tp_gt1_calls_all_reduce(self):
        layer_mod = importlib.import_module(AscendRowParallelLinear.__module__)

        class DummyTpGroup:
            def __init__(self):
                self.called = False
                self.last_tensor = None

            def all_reduce(self, tensor):
                self.called = True
                self.last_tensor = tensor
                return tensor * 2

        class DummyQuantMethod:
            def create_weights(
                self,
                layer,
                input_size_per_partition,
                output_partition_sizes,
                input_size,
                output_size,
                params_dtype,
                weight_loader,
            ):
                # 不创建真实权重
                return

            def apply(self, layer, input_parallel, bias=None):
                # 完全忽略权重和 bias，只要返回一个 [B, output_size] 的 tensor
                return torch.zeros(input_parallel.shape[0], layer.output_size, dtype=input_parallel.dtype)

        tp_group = DummyTpGroup()

        with mock.patch.object(layer_mod, "get_mlp_tp_group", return_value=tp_group):
            layer = self._make_ascend_row_parallel_linear_layer(
                input_size=10,
                output_size=6,
                tp_size=2,
                tp_rank=0,
                reduce_results=True,
            )
            layer.quant_method = DummyQuantMethod()

            x = torch.randn(4, 10, dtype=torch.float32)
            out, out_bias = layer(x)

        assert tp_group.called
        assert out.shape == (4, 6)
        assert out_bias is None

# ================= DP2TPRowParallelLinear =================

    # ---------- forward: input_is_parallel 路径 ----------

    def test_dp2tp_row_parallel_linear_forward_input_parallel_tp1_reduce_true_rank0_bias_fused(self):
        from omni.layers.linear import DP2TPRowParallelLinear
        layer_mod = importlib.import_module(DP2TPRowParallelLinear.__module__)

        class CaptureQuantMethod:
            def __init__(self):
                self.last_bias = None

            def create_weights(
                self,
                layer,
                input_size_per_partition,
                output_partition_sizes,
                input_size,
                output_size,
                params_dtype,
                weight_loader,
            ):
                return

            def apply(self, layer, x, bias=None):
                self.last_bias = bias
                return torch.zeros(x.shape[0], layer.output_size, dtype=x.dtype)

        class FailingGroup:
            # tp_size == 1 时不应调用 reduce_scatter
            def __init__(self):
                self.device_group = None

            def reduce_scatter(self, tensor):
                raise AssertionError("reduce_scatter should not be called when tp_size == 1")

        layer = DP2TPRowParallelLinear(
            input_size=8,
            output_size=4,
            tp_size=1,
            tp_rank=0,
            bias=True,
            input_is_parallel=True,
            skip_bias_add=False,
            params_dtype=torch.float32,
            reduce_results=True,
            quant_config=None,
        )
        qm = CaptureQuantMethod()
        layer.quant_method = qm

        x = torch.randn(3, 8, dtype=torch.float32)

        with mock.patch.object(layer_mod, "get_o_proj_tp_group", return_value=FailingGroup()):
            out, out_bias = layer(x, bsz=3, q_len=1, num_heads=1, v_head_dim=1)

        # rank0 且不 skip_bias_add -> bias 融合进 GEMM
        self.assertIs(qm.last_bias, layer.bias)
        self.assertEqual(out.shape, (3, 4))
        self.assertIsNone(out_bias)


    def test_dp2tp_row_parallel_linear_forward_input_parallel_skip_bias_add_true_returns_output_and_output_bias(self):
        from omni.layers.linear import DP2TPRowParallelLinear

        class DummyQuantMethod:
            def create_weights(
                self,
                layer,
                input_size_per_partition,
                output_partition_sizes,
                input_size,
                output_size,
                params_dtype,
                weight_loader,
            ):
                return

            def apply(self, layer, x, bias=None):
                # 不使用 bias，直接返回输出
                return torch.zeros(x.shape[0], layer.output_size, dtype=x.dtype)

        layer = DP2TPRowParallelLinear(
            input_size=8,
            output_size=4,
            tp_size=2,
            tp_rank=0,
            bias=True,
            input_is_parallel=True,
            skip_bias_add=True,
            params_dtype=torch.float32,
            reduce_results=False,
            quant_config=None,
        )
        qm = DummyQuantMethod()
        layer.quant_method = qm

        x = torch.randn(5, 8, dtype=torch.float32)
        out, out_bias = layer(x, bsz=5, q_len=1, num_heads=1, v_head_dim=1)

        self.assertEqual(out.shape, (5, 4))
        # skip_bias_add=True -> 第二个输出为 bias
        self.assertIs(out_bias, layer.bias)

    # ---------- forward: input_is_parallel = False（DP→TP all_to_all 路径） ----------

    def test_dp2tp_row_parallel_linear_forward_input_not_parallel_reduce_results_true_tp_gt1_calls_reduce_scatter(self):
        from omni.layers.linear import DP2TPRowParallelLinear
        layer_mod = importlib.import_module(DP2TPRowParallelLinear.__module__)

        tp_size = 2
        input_size = 8
        bsz = 2
        q_len = 3
        num_heads = 2
        v_head_dim = 4  # 满足 reshape 约束 input_size == num_heads * v_head_dim

        class DummyQuantMethod:
            def create_weights(
                self,
                layer,
                input_size_per_partition,
                output_partition_sizes,
                input_size,
                output_size,
                params_dtype,
                weight_loader,
            ):
                return

            def apply(self, layer, x, bias=None):
                # 返回一个非零 tensor，方便区分
                return torch.ones(x.shape[0], layer.output_size, dtype=x.dtype)

        class DummyGroup:
            def __init__(self):
                self.device_group = "dummy_group"
                self.called = False
                self.last_tensor = None

            def reduce_scatter(self, tensor):
                self.called = True
                self.last_tensor = tensor
                return tensor * 2

        class DummyPlatform:
            def __init__(self):
                self.device_type = "cpu"

        tp_group = DummyGroup()

        def fake_all_to_all_single(output, inp, group=None):
            output.copy_(inp)

        def fake_prefetch(weight, inp, size):
            return

        layer = DP2TPRowParallelLinear(
            input_size=input_size,
            output_size=6,
            tp_size=tp_size,
            tp_rank=0,
            bias=False,
            input_is_parallel=False,
            skip_bias_add=False,
            params_dtype=torch.float32,
            reduce_results=True,
            quant_config=None,
        )
        layer.quant_method = DummyQuantMethod()

        input_size_per_partition = layer.input_size_per_partition
        x = torch.randn(bsz * q_len, tp_size, input_size_per_partition, dtype=torch.float32)

        with mock.patch.object(layer_mod, "current_platform", DummyPlatform()), \
            mock.patch.object(layer_mod, "get_o_proj_tp_group", return_value=tp_group), \
            mock.patch.object(layer_mod.dist, "all_to_all_single", side_effect=fake_all_to_all_single), \
            mock.patch.object(layer_mod.torch_npu, "npu_prefetch", side_effect=fake_prefetch):

            out, out_bias = layer(x, bsz=bsz, q_len=q_len, num_heads=num_heads, v_head_dim=v_head_dim)

        self.assertTrue(tp_group.called)
        self.assertIsNotNone(tp_group.last_tensor)
        self.assertEqual(out.shape, (bsz * q_len * tp_size, layer.output_size))
        self.assertIsNone(out_bias)

# ================= Tp2DpAndTpRowParallelLinear =================

    # ---------- helper ----------

    def _make_tp2dp_row_parallel_linear_layer(
        self,
        input_size=8,
        output_size=4,
        tp_size=2,
        tp_rank=0,
        bias=True,
        input_is_parallel=True,
        skip_bias_add=False,
        reduce_results=True,
        params_dtype=torch.float32,
    ):
        return Tp2DpAndTpRowParallelLinear(
            input_size=input_size,
            output_size=output_size,
            tp_size=tp_size,
            tp_rank=tp_rank,
            bias=bias,
            input_is_parallel=input_is_parallel,
            skip_bias_add=skip_bias_add,
            params_dtype=params_dtype,
            reduce_results=reduce_results,
            quant_config=None,
        )

    # ----------- __init__ / 基本属性 -------------

    def test_tp2dp_row_parallel_linear_init_basic(self):
        layer = self._make_tp2dp_row_parallel_linear_layer(
            input_size=12,
            output_size=6,
            tp_size=3,
            tp_rank=1,
            bias=True,
            input_is_parallel=True,
            skip_bias_add=False,
            reduce_results=True,
        )

        self.assertEqual(layer.input_size, 12)
        self.assertEqual(layer.output_size, 6)
        self.assertEqual(layer.tp_size, 3)
        self.assertEqual(layer.tp_rank, 1)
        self.assertTrue(layer.input_is_parallel)
        self.assertTrue(layer.reduce_results)
        self.assertFalse(layer.skip_bias_add)
        self.assertIsNotNone(layer.bias)

        # AscendRowParallelLinear 中定义的分片相关属性
        self.assertEqual(layer.input_size_per_partition, 12 // 3)
        self.assertEqual(layer.output_size_per_partition, 6)
        self.assertEqual(layer.output_partition_sizes, [6])

    # ---------- weight_loader 核心路径 ----------

    def test_tp2dp_row_parallel_linear_weight_loader_input_dim_sliced_by_dp_and_tp_rank(self):
        """
        input_dim is not None & not bitsandbytes -> 按 DP 分片后再按 tp_rank 拼接。

        world_size = 4, tp_size = 2 -> dp_size = 2
        rank_list = [[0, 2],
                    [1, 3]]

        对于 tp_rank = 1 -> 取 global ranks 1,3 的片段 concat，最终宽度等于本 rank param_data 的宽度。
        """
        from omni.layers.linear import Tp2DpAndTpRowParallelLinear
        layer_mod = importlib.import_module(Tp2DpAndTpRowParallelLinear.__module__)

        input_size = 12
        world_size = 4
        tp_size = 2
        dp_size = world_size // tp_size  # 2

        layer = self._make_tp2dp_row_parallel_linear_layer(
            input_size=input_size,
            output_size=4,
            tp_size=tp_size,
            tp_rank=1,
        )

        param = Parameter(torch.zeros(2, input_size, dtype=torch.float32))
        param.input_dim = 1  # 按 dim=1 切

        # shard_size = param_data.shape[input_dim] // dp_size = 12 // 2 = 6
        shard_size = param.data.shape[1] // dp_size  # 6

        # 正确的假设：loaded_weight 在 input_dim 上包含 world_size 个 shard
        # 因此长度应该是 world_size * shard_size = 4 * 6 = 24
        loaded_width = world_size * shard_size       # 24
        loaded_weight = torch.arange(
            2 * loaded_width, dtype=torch.float32
        ).reshape(2, loaded_width)                  # [2, 24]

        # 期望结果：对 tp_rank=1，rank_list = [[0,2],[1,3]] -> 取 global ranks 1,3
        rank_list = torch.arange(world_size).reshape(-1, tp_size).T  # [[0,2],[1,3]]
        tp_ranks_for_this = rank_list[1]                            # tensor([1, 3])

        expected_slices = []
        for r in tp_ranks_for_this:
            start = int(r.item()) * shard_size
            expected_slices.append(
                loaded_weight.narrow(1, start, shard_size)
            )
        expected = torch.cat(expected_slices, dim=1)  # [2, 12]

        with mock.patch.object(layer_mod, "get_world_group") as mock_get_group:
            mock_group = mock.Mock()
            mock_group.world_size = world_size
            mock_get_group.return_value = mock_group
            layer.weight_loader(param, loaded_weight)

        self.assertEqual(tuple(param.data.shape), (2, input_size))
        self.assertTrue(torch.allclose(param.data, expected))


    def test_tp2dp_row_parallel_linear_weight_loader_input_dim_with_bitsandbytes_skips_slicing(self):
        """use_bitsandbytes_4bit=True 时，input_dim 逻辑跳过，直接要求形状一致并复制。"""
        layer_mod = importlib.import_module(Tp2DpAndTpRowParallelLinear.__module__)

        layer = self._make_tp2dp_row_parallel_linear_layer(
            input_size=8,
            output_size=4,
            tp_size=2,
            tp_rank=0,
        )

        param = Parameter(torch.zeros(2, 8, dtype=torch.float32))
        param.input_dim = 1
        param.use_bitsandbytes_4bit = True

        loaded_weight = torch.randn(2, 8, dtype=torch.float32)

        with mock.patch.object(layer_mod, "get_world_group") as mock_get_group:
            mock_group = mock.Mock()
            mock_group.world_size = 4
            mock_get_group.return_value = mock_group
            layer.weight_loader(param, loaded_weight)

        self.assertTrue(torch.allclose(param.data, loaded_weight))

    def test_tp2dp_row_parallel_linear_weight_loader_gguf_weight_type_and_uninitialized(self):
        """覆盖 is_gguf_weight_type 和 is_gguf_weight + UninitializedParameter materialize 分支。"""
        layer_mod = importlib.import_module(Tp2DpAndTpRowParallelLinear.__module__)
        from torch.nn.parameter import UninitializedParameter

        layer = self._make_tp2dp_row_parallel_linear_layer(
            input_size=8,
            output_size=4,
            tp_size=2,
            tp_rank=0,
        )

        # 1) is_gguf_weight_type: 只设置 weight_type，不动 data
        param_type = Parameter(torch.zeros(1, dtype=torch.float32))
        param_type.is_gguf_weight_type = True
        param_type.is_gguf_weight = False
        loaded_type = torch.tensor(5, dtype=torch.int32)

        with mock.patch.object(layer_mod, "get_world_group") as mock_get_group:
            mock_group = mock.Mock()
            mock_group.world_size = 2
            mock_get_group.return_value = mock_group
            layer.weight_loader(param_type, loaded_type)

        self.assertEqual(param_type.weight_type, 5)

        # 2) is_gguf_weight & UninitializedParameter: materialize 时 input_dim 被除以 tp_size
        param_uninit = UninitializedParameter(requires_grad=False)
        param_uninit.is_gguf_weight = True
        param_uninit.input_dim = 1
        loaded_weight = torch.randn(2, 8, dtype=torch.float32)  # dim1=8 -> 8//tp_size=4

        with mock.patch.object(layer_mod, "get_world_group") as mock_get_group:
            mock_group = mock.Mock()
            mock_group.world_size = 2
            mock_get_group.return_value = mock_group
            layer.weight_loader(param_uninit, loaded_weight)

        # materialize 后 shape[1] 被除以 tp_size
        self.assertEqual(tuple(param_uninit.data.shape), (2, 4))

    # ---------- forward: input_is_parallel ----------
    def test_tp2dp_row_parallel_linear_forward_input_parallel_bias_and_reduce_scatter(self):
        """input_is_parallel=True, tp_size>1, reduce_results=True -> 调用 reduce_scatter；rank0 融合 bias。"""
        layer_mod = importlib.import_module(Tp2DpAndTpRowParallelLinear.__module__)

        class CaptureQuantMethod:
            def __init__(self):
                self.last_bias = None

            def create_weights(
                self,
                layer,
                input_size_per_partition,
                output_partition_sizes,
                input_size,
                output_size,
                params_dtype,
                weight_loader,
            ):
                return

            def apply(self, layer, x, bias=None):
                self.last_bias = bias
                return torch.zeros(x.shape[0], layer.output_size, dtype=x.dtype)

        class DummyGroup:
            def __init__(self):
                self.device_group = None
                self.called = False
                self.last_tensor = None

            def reduce_scatter(self, tensor):
                self.called = True
                self.last_tensor = tensor
                return tensor + 1  # 只要能区分被调用即可

        tp_group = DummyGroup()

        with mock.patch.object(layer_mod, "get_o_proj_tp_group", return_value=tp_group):
            layer = self._make_tp2dp_row_parallel_linear_layer(
                input_size=8,
                output_size=4,
                tp_size=2,
                tp_rank=0,
                bias=True,
                input_is_parallel=True,
                skip_bias_add=False,
                reduce_results=True,
            )
            qm = CaptureQuantMethod()
            layer.quant_method = qm

            x = torch.randn(3, 8)
            out, out_bias = layer(x)

        self.assertIs(qm.last_bias, layer.bias)
        self.assertTrue(tp_group.called)
        self.assertEqual(out.shape, (3, 4))
        # skip_bias_add=False -> 第二输出为 None
        self.assertIsNone(out_bias)

    def test_tp2dp_row_parallel_linear_forward_input_parallel_skip_bias_add_true_returns_bias(self):
        """input_is_parallel=True, skip_bias_add=True -> GEMM 不融合 bias，第二输出返回 bias。"""
        from omni.layers.linear import Tp2DpAndTpRowParallelLinear

        class DummyQuantMethod:
            def create_weights(
                self,
                layer,
                input_size_per_partition,
                output_partition_sizes,
                input_size,
                output_size,
                params_dtype,
                weight_loader,
            ):
                return

            def apply(self, layer, x, bias=None):
                # 不用 bias，直接返回零矩阵
                return torch.zeros(x.shape[0], layer.output_size, dtype=x.dtype)

        layer = Tp2DpAndTpRowParallelLinear(
            input_size=8,
            output_size=4,
            tp_size=2,
            tp_rank=0,
            bias=True,
            input_is_parallel=True,
            skip_bias_add=True,
            params_dtype=torch.float32,
            reduce_results=False,
            quant_config=None,
        )
        layer.quant_method = DummyQuantMethod()

        x = torch.randn(5, 8)
        out, out_bias = layer(x)

        self.assertEqual(out.shape, (5, 4))
        self.assertIs(out_bias, layer.bias)

    def test_tp2dp_row_parallel_linear_forward_input_parallel_reduce_results_false_no_reduce_scatter(self):
        """reduce_results=False 时，不应调用 reduce_scatter，直接返回 output_parallel。"""
        layer_mod = importlib.import_module(Tp2DpAndTpRowParallelLinear.__module__)

        class DummyQuantMethod:
            def create_weights(
                self,
                layer,
                input_size_per_partition,
                output_partition_sizes,
                input_size,
                output_size,
                params_dtype,
                weight_loader,
            ):
                return

            def apply(self, layer, x, bias=None):
                return torch.zeros(x.shape[0], layer.output_size, dtype=x.dtype)

        class FailingGroup:
            def __init__(self):
                self.device_group = None

            def reduce_scatter(self, tensor):
                raise AssertionError("reduce_scatter should not be called when reduce_results=False")

        with mock.patch.object(layer_mod, "get_o_proj_tp_group", return_value=FailingGroup()):
            layer = self._make_tp2dp_row_parallel_linear_layer(
                input_size=10,
                output_size=6,
                tp_size=2,
                tp_rank=0,
                bias=False,
                input_is_parallel=True,
                skip_bias_add=False,
                reduce_results=False,
            )
            layer.quant_method = DummyQuantMethod()

            x = torch.randn(3, 10)
            out, out_bias = layer(x)

        self.assertEqual(out.shape, (3, 6))
        self.assertIsNone(out_bias)

    # ---------- forward: input_is_parallel=False（TP→DP 再 TP 路径） ----------

    def test_tp2dp_row_parallel_linear_forward_input_not_parallel_splits_by_tp_rank(self):
        """input_is_parallel=False -> 使用 split_tensor_along_last_dim，选中 tp_rank 对应切片。"""
        layer = self._make_tp2dp_row_parallel_linear_layer(
            input_size=12,
            output_size=5,
            tp_size=3,
            tp_rank=1,
            bias=False,
            input_is_parallel=False,
            skip_bias_add=False,
            reduce_results=False,
        )

        class CaptureQuantMethod:
            def __init__(self):
                self.last_input = None
                self.last_bias = None

            def create_weights(
                self,
                layer,
                input_size_per_partition,
                output_partition_sizes,
                input_size,
                output_size,
                params_dtype,
                weight_loader,
            ):
                return

            def apply(self, layer, x, bias=None):
                self.last_input = x.clone()
                self.last_bias = bias
                return torch.zeros(x.shape[0], layer.output_size, dtype=x.dtype)

        qm = CaptureQuantMethod()
        layer.quant_method = qm

        batch_size = 2
        x = torch.arange(batch_size * 12, dtype=torch.float32).reshape(batch_size, 12)

        out, out_bias = layer(x)

        splits = split_tensor_along_last_dim(x, num_partitions=3)
        expected_parallel = splits[1].contiguous()

        self.assertTrue(torch.allclose(qm.last_input, expected_parallel))
        self.assertEqual(out.shape, (batch_size, layer.output_size))
        self.assertIsNone(out_bias)

# ================= ColumnParallelLinearQuantGather =================

    # ---------- helpers: ColumnParallelLinearQuantGather ----------

    def _make_column_parallel_linear_quant_gather_layer(
        self,
        input_size=8,
        output_size=4,
        bias=True,
    ):
        from omni.layers.linear import ColumnParallelLinearQuantGather
        from unittest.mock import patch
        with patch("vllm.model_executor.layers.linear.get_tensor_model_parallel_world_size",
                   return_value=1), \
             patch("vllm.model_executor.layers.linear.get_tensor_model_parallel_rank",
                   return_value=0):
            layer = ColumnParallelLinearQuantGather(
                input_size=input_size,
                output_size=output_size,
                bias=bias,
                quant_config=None,
                prefix="col_qg.",
            )
        return layer

    def test_column_parallel_linear_quant_gather_forward_no_gather_uses_output_parallel_and_bias_logic(self):
        import torch

        from omni.layers.linear import ColumnParallelLinearQuantGather

        class CaptureQuantMethod:
            def __init__(self):
                self.last_layer = None
                self.last_input = None
                self.last_bias = None
                self.last_inner_gather = None

            def create_weights(
                self,
                layer,
                input_size_per_partition,
                output_partition_sizes,
                input_size,
                output_size,
                params_dtype,
                weight_loader,
            ):
                return

            def apply(self, layer, x, bias=None, inner_gather=False):
                self.last_layer = layer
                self.last_input = x.clone()
                self.last_bias = bias
                self.last_inner_gather = inner_gather
                # 只要返回一个 [B, output_size] 的 tensor 即可
                return torch.zeros(x.shape[0], layer.output_size, dtype=x.dtype)

        input_size = 8
        output_size = 4
        batch = 3

        layer = self._make_column_parallel_linear_quant_gather_layer(
            input_size=input_size,
            output_size=output_size,
            bias=True,
        )
        qm = CaptureQuantMethod()
        layer.quant_method = qm
        layer.gather_output = False      # 走 non-gather 分支
        layer.skip_bias_add = False      # bias 融合进 GEMM

        x = torch.randn(batch, input_size, dtype=torch.float32)
        out, out_bias = layer(x)

        # 检查 quant_method.apply 的调用参数
        self.assertIs(qm.last_layer, layer)
        self.assertTrue(torch.allclose(qm.last_input, x))
        self.assertIs(qm.last_bias, layer.bias)          # 融合 bias
        self.assertTrue(qm.last_inner_gather)            # inner_gather=True

        # 不 gather_output 时，输出就是 output_parallel
        self.assertEqual(out.shape, (batch, output_size))
        self.assertIsNone(out_bias)


    def test_column_parallel_linear_quant_gather_forward_gather_output_true_calls_all_gather(self):
        import torch
        from unittest.mock import patch

        from omni.layers.linear import ColumnParallelLinearQuantGather

        class DummyQuantMethod:
            def __init__(self, ret):
                self.ret = ret

            def create_weights(
                self,
                layer,
                input_size_per_partition,
                output_partition_sizes,
                input_size,
                output_size,
                params_dtype,
                weight_loader,
            ):
                return

            def apply(self, layer, x, bias=None, inner_gather=False):
                # 忽略输入，直接返回预设的 output_parallel
                return self.ret

        input_size = 6
        output_size = 4
        batch = 3

        # 构造 layer（内部已经打桩过 TP world_size/rank）
        layer = self._make_column_parallel_linear_quant_gather_layer(
            input_size=input_size,
            output_size=output_size,
            bias=True,
        )

        output_parallel = torch.randn(batch, output_size, dtype=torch.float32)
        gathered = torch.randn(batch, output_size, dtype=torch.float32)

        layer.quant_method = DummyQuantMethod(output_parallel)
        layer.gather_output = True      # 打开 gather_output，触发 all_gather 分支
        layer.skip_bias_add = False

        x = torch.randn(batch, input_size, dtype=torch.float32)

        with patch("omni.layers.linear.tensor_model_parallel_all_gather") as mock_all_gather:
            mock_all_gather.return_value = gathered

            out, out_bias = layer(x)

        # 确认 all_gather 被调用且传入了 output_parallel
        mock_all_gather.assert_called_once()
        args, kwargs = mock_all_gather.call_args
        self.assertIs(args[0], output_parallel)

        # 输出应为 all_gather 的结果
        self.assertTrue(torch.allclose(out, gathered))
        self.assertIsNone(out_bias)


    def test_column_parallel_linear_quant_gather_forward_skip_bias_add_true_returns_output_and_output_bias(self):
        import torch

        from omni.layers.linear import ColumnParallelLinearQuantGather

        class CaptureQuantMethod:
            def __init__(self):
                self.last_bias = None
                self.last_inner_gather = None

            def create_weights(
                self,
                layer,
                input_size_per_partition,
                output_partition_sizes,
                input_size,
                output_size,
                params_dtype,
                weight_loader,
            ):
                return

            def apply(self, layer, x, bias=None, inner_gather=False):
                self.last_bias = bias
                self.last_inner_gather = inner_gather
                return torch.zeros(x.shape[0], layer.output_size, dtype=x.dtype)

        input_size = 8
        output_size = 4
        batch = 5

        layer = self._make_column_parallel_linear_quant_gather_layer(
            input_size=input_size,
            output_size=output_size,
            bias=True,
        )

        qm = CaptureQuantMethod()
        layer.quant_method = qm
        layer.skip_bias_add = True      # 不在 GEMM 里加 bias
        layer.gather_output = False

        x = torch.randn(batch, input_size, dtype=torch.float32)
        out, out_bias = layer(x)

        # skip_bias_add=True -> GEMM 不应融合 bias
        self.assertIsNone(qm.last_bias)
        self.assertTrue(qm.last_inner_gather)

        # 第二个输出应为 layer.bias
        self.assertEqual(out.shape, (batch, output_size))
        self.assertIs(out_bias, layer.bias)

# ================= RowParallelLinear =================

    def test_row_parallel_linear_forward_input_parallel_tp1_no_reduce_bias_fused_rank0(self):
        import vllm.model_executor.layers.linear as vllm_linear
        tp_size = 1
        tp_rank = 0
        input_size = 8
        output_size = 4
        batch_size = 3

        class CaptureQuantMethod:
            def __init__(self):
                self.last_layer = None
                self.last_input = None
                self.last_bias = None

            def apply(self, layer, x, bias=None):
                self.last_layer = layer
                self.last_input = x.clone()
                self.last_bias = bias
                return torch.zeros(x.shape[0], layer.output_size, dtype=x.dtype)

        # all_reduce 如果被调用就抛异常，确保 tp_size==1 时不会走到这一步
        layer_mod = importlib.import_module(RowParallelLinear.__module__)

        def failing_all_reduce(tensor):
            raise AssertionError(
                "tensor_model_parallel_all_reduce should not be called when tp_size == 1"
            )

        with mock.patch.object(
            vllm_linear, "get_tensor_model_parallel_world_size", return_value=tp_size
        ), mock.patch.object(
            vllm_linear, "get_tensor_model_parallel_rank", return_value=tp_rank
        ), mock.patch.object(
            layer_mod, "tensor_model_parallel_all_reduce", side_effect=failing_all_reduce
        ):
            layer = RowParallelLinear(
                input_size=input_size,
                output_size=output_size,
                bias=True,
                input_is_parallel=True,
                skip_bias_add=False,
                params_dtype=torch.float32,
                reduce_results=True,
                quant_config=None,
                prefix="row.",
            )

            qm = CaptureQuantMethod()
            layer.quant_method = qm

            x = torch.randn(batch_size, input_size, dtype=torch.float32)
            out, out_bias = layer(x)

        # 验证 quant_method.apply 收到的 input 与 bias
        self.assertTrue(torch.allclose(qm.last_input, x))
        self.assertIs(qm.last_bias, layer.bias)
        # 输出形状与第二个输出
        self.assertEqual(out.shape, (batch_size, output_size))
        self.assertIsNone(out_bias)

    def test_row_parallel_linear_forward_input_not_parallel_splits_and_selects_tp_rank_slice(self):
        """input_is_parallel=False 时，应按 tp_size 切分最后一维并取 tp_rank 那份。"""
        import vllm.model_executor.layers.linear as vllm_linear
        tp_size = 3
        tp_rank = 1
        input_size = 12  # 可整除 tp_size
        output_size = 5
        batch_size = 2

        class CaptureQuantMethod:
            def __init__(self):
                self.last_input = None
                self.last_bias = None

            def apply(self, layer, x, bias=None):
                self.last_input = x.clone()
                self.last_bias = bias
                return torch.zeros(x.shape[0], layer.output_size, dtype=x.dtype)

        layer_mod = importlib.import_module(RowParallelLinear.__module__)

        # reduce_results=False，保证不会调用 all_reduce
        def failing_all_reduce(tensor):
            raise AssertionError(
                "tensor_model_parallel_all_reduce should not be called when reduce_results=False"
            )

        with mock.patch.object(
            vllm_linear, "get_tensor_model_parallel_world_size", return_value=tp_size
        ), mock.patch.object(
            vllm_linear, "get_tensor_model_parallel_rank", return_value=tp_rank
        ), mock.patch.object(
            layer_mod, "tensor_model_parallel_all_reduce", side_effect=failing_all_reduce
        ):
            layer = RowParallelLinear(
                input_size=input_size,
                output_size=output_size,
                bias=False,
                input_is_parallel=False,
                skip_bias_add=False,
                params_dtype=torch.float32,
                reduce_results=False,
                quant_config=None,
                prefix="row.",
            )
            qm = CaptureQuantMethod()
            layer.quant_method = qm

            x = torch.arange(batch_size * input_size, dtype=torch.float32).reshape(
                batch_size, input_size
            )
            out, out_bias = layer(x)

        # 期望的切分结果
        splits = split_tensor_along_last_dim(x, num_partitions=tp_size)
        expected_parallel = splits[tp_rank].contiguous()
        self.assertTrue(torch.allclose(qm.last_input, expected_parallel))
        self.assertIsNone(qm.last_bias)

        self.assertEqual(out.shape, (batch_size, output_size))
        self.assertIsNone(out_bias)

    def test_row_parallel_linear_forward_reduce_results_true_tp_gt1_calls_all_reduce(self):
        """tp_size>1 且 reduce_results=True 时，应调用 tensor_model_parallel_all_reduce。"""
        import vllm.model_executor.layers.linear as vllm_linear

        tp_size = 2
        tp_rank = 0
        input_size = 8
        output_size = 4
        batch_size = 3

        class DummyQuantMethod:
            def apply(self, layer, x, bias=None):
                # 返回全 1，方便后面看 all_reduce 的效果
                return torch.ones(x.shape[0], layer.output_size, dtype=x.dtype)

        class DummyAllReduce:
            def __init__(self):
                self.called = False
                self.last_tensor = None

            def __call__(self, tensor):
                self.called = True
                self.last_tensor = tensor.clone()
                return tensor * 2  # 输出乘 2，方便验证

        all_reduce = DummyAllReduce()
        layer_mod = importlib.import_module(RowParallelLinear.__module__)

        with mock.patch.object(
            vllm_linear, "get_tensor_model_parallel_world_size", return_value=tp_size
        ), mock.patch.object(
            vllm_linear, "get_tensor_model_parallel_rank", return_value=tp_rank
        ), mock.patch.object(
            layer_mod, "tensor_model_parallel_all_reduce", side_effect=all_reduce
        ):
            layer = RowParallelLinear(
                input_size=input_size,
                output_size=output_size,
                bias=False,
                input_is_parallel=True,
                skip_bias_add=False,
                params_dtype=torch.float32,
                reduce_results=True,
                quant_config=None,
                prefix="row.",
            )
            layer.quant_method = DummyQuantMethod()

            x = torch.randn(batch_size, input_size, dtype=torch.float32)
            out, out_bias = layer(x)

        self.assertTrue(all_reduce.called)
        self.assertEqual(all_reduce.last_tensor.shape, (batch_size, output_size))
        self.assertTrue(
            torch.allclose(out, torch.full((batch_size, output_size), 2.0))
        )
        self.assertIsNone(out_bias)

    def test_row_parallel_linear_forward_reduce_results_false_does_not_call_all_reduce(self):
        """reduce_results=False 时，即使 tp_size>1 也不应调用 all_reduce。"""
        import vllm.model_executor.layers.linear as vllm_linear

        tp_size = 2
        tp_rank = 0
        input_size = 10
        output_size = 6
        batch_size = 3

        class DummyQuantMethod:
            def apply(self, layer, x, bias=None):
                return torch.zeros(x.shape[0], layer.output_size, dtype=x.dtype)

        def failing_all_reduce(tensor):
            raise AssertionError(
                "tensor_model_parallel_all_reduce should not be called when reduce_results=False"
            )

        layer_mod = importlib.import_module(RowParallelLinear.__module__)

        with mock.patch.object(
            vllm_linear, "get_tensor_model_parallel_world_size", return_value=tp_size
        ), mock.patch.object(
            vllm_linear, "get_tensor_model_parallel_rank", return_value=tp_rank
        ), mock.patch.object(
            layer_mod, "tensor_model_parallel_all_reduce", side_effect=failing_all_reduce
        ):
            layer = RowParallelLinear(
                input_size=input_size,
                output_size=output_size,
                bias=False,
                input_is_parallel=True,
                skip_bias_add=False,
                params_dtype=torch.float32,
                reduce_results=False,
                quant_config=None,
                prefix="row.",
            )
            layer.quant_method = DummyQuantMethod()

            x = torch.randn(batch_size, input_size, dtype=torch.float32)
            out, out_bias = layer(x)

        self.assertEqual(out.shape, (batch_size, output_size))
        self.assertIsNone(out_bias)

    def test_row_parallel_linear_forward_skip_bias_add_true_returns_output_and_output_bias(self):
        """skip_bias_add=True 时，GEMM 不融合 bias，第二个输出返回 bias。"""
        import vllm.model_executor.layers.linear as vllm_linear

        tp_size = 2
        tp_rank = 0
        input_size = 8
        output_size = 4
        batch_size = 5

        class CaptureQuantMethod:
            def __init__(self):
                self.last_bias = None

            def apply(self, layer, x, bias=None):
                self.last_bias = bias
                return torch.zeros(x.shape[0], layer.output_size, dtype=x.dtype)

        with mock.patch.object(
            vllm_linear, "get_tensor_model_parallel_world_size", return_value=tp_size
        ), mock.patch.object(
            vllm_linear, "get_tensor_model_parallel_rank", return_value=tp_rank
        ):
            layer = RowParallelLinear(
                input_size=input_size,
                output_size=output_size,
                bias=True,
                input_is_parallel=True,
                skip_bias_add=True,  # 关键
                params_dtype=torch.float32,
                reduce_results=False,
                quant_config=None,
                prefix="row.",
            )
            qm = CaptureQuantMethod()
            layer.quant_method = qm

            x = torch.randn(batch_size, input_size, dtype=torch.float32)
            out, out_bias = layer(x)

        # skip_bias_add=True -> GEMM 不融合 bias
        self.assertIsNone(qm.last_bias)
        # 第二个输出为 bias 本身
        self.assertIs(out_bias, layer.bias)
        self.assertEqual(out.shape, (batch_size, output_size))

    def test_row_parallel_linear_forward_bias_fused_only_on_rank0_when_not_skip_bias_add(self):
        """不 skip_bias_add 时，仅 rank0 融合 bias，其它 rank bias=None。"""
        import vllm.model_executor.layers.linear as vllm_linear
        import omni.layers.linear as omni_linear

        tp_size = 2
        input_size = 8
        output_size = 4
        batch_size = 3

        class CaptureQuantMethod:
            def __init__(self):
                self.last_bias = None

            def apply(self, layer, x, bias=None):
                self.last_bias = bias
                return torch.zeros(x.shape[0], layer.output_size, dtype=x.dtype)

        x = torch.randn(batch_size, input_size, dtype=torch.float32)

        # mock world size & all_reduce
        with mock.patch.object(
            vllm_linear, "get_tensor_model_parallel_world_size", return_value=tp_size
        ), mock.patch.object(
            omni_linear, "tensor_model_parallel_all_reduce", side_effect=lambda t: t
        ):
            # ---------- rank 0：应融合 bias ----------
            with mock.patch.object(
                vllm_linear, "get_tensor_model_parallel_rank", return_value=0
            ):
                layer0 = RowParallelLinear(
                    input_size=input_size,
                    output_size=output_size,
                    bias=True,
                    input_is_parallel=True,
                    skip_bias_add=False,
                    params_dtype=torch.float32,
                    reduce_results=True,   # ★ 关键：设为 True，避免 __init__ 里 ValueError
                    quant_config=None,
                    prefix="row.",
                )
                qm0 = CaptureQuantMethod()
                layer0.quant_method = qm0

                out0, bias_out0 = layer0(x)

                # rank0 -> GEMM 中融合 bias，第二输出为 None
                self.assertIs(qm0.last_bias, layer0.bias)
                self.assertIsNone(bias_out0)
                self.assertEqual(out0.shape, (batch_size, output_size))

            # ---------- rank 1：不应融合 bias ----------
            with mock.patch.object(
                vllm_linear, "get_tensor_model_parallel_rank", return_value=1
            ):
                layer1 = RowParallelLinear(
                    input_size=input_size,
                    output_size=output_size,
                    bias=True,
                    input_is_parallel=True,
                    skip_bias_add=False,
                    params_dtype=torch.float32,
                    reduce_results=True,   # 同上
                    quant_config=None,
                    prefix="row.",
                )
                qm1 = CaptureQuantMethod()
                layer1.quant_method = qm1

                out1, bias_out1 = layer1(x)

                # rank>0 -> 不融合 bias，第二输出为 None
                self.assertIsNone(qm1.last_bias)
                self.assertIsNone(bias_out1)
                self.assertEqual(out1.shape, (batch_size, output_size))

# ================= RowParallelLinearWithReduceScatter =================


    def test_row_parallel_linear_with_reduce_scatter_init_bias_true_raises(self):
        """bias=True 时，__init__ 中强制检测 self.bias is not None，抛 RuntimeError。"""
        from omni.layers.linear import RowParallelLinearWithReduceScatter
        import vllm.model_executor.layers.linear as vllm_linear

        with mock.patch.object(vllm_linear, "get_tensor_model_parallel_world_size", return_value=2), \
             mock.patch.object(vllm_linear, "get_tensor_model_parallel_rank", return_value=0):
            with self.assertRaisesRegex(RuntimeError, "self.bias is not None"):
                _ = RowParallelLinearWithReduceScatter(
                    input_size=8,
                    output_size=4,
                    bias=True,               # 关键：会创建 bias，从而触发检查
                    input_is_parallel=True,
                    skip_bias_add=False,
                    params_dtype=torch.float32,
                    reduce_results=True,
                    quant_config=None,
                    prefix="rs.",
                )

    def test_row_parallel_linear_with_reduce_scatter_forward_input_parallel_tp1_no_reduce(self):
        """
        input_is_parallel=True, tp_size=1, reduce_results=True：
        不应调用 mla_tensor_model_parallel_reduce_scatter，直接返回 output_parallel。
        """
        from omni.layers.linear import RowParallelLinearWithReduceScatter
        import vllm.model_executor.layers.linear as vllm_linear
        import importlib

        class CaptureQuantMethod:
            def __init__(self):
                self.last_layer = None
                self.last_input = None
                self.last_bias = None

            def apply(self, layer, x, bias=None):
                self.last_layer = layer
                self.last_input = x.clone()
                self.last_bias = bias
                return torch.zeros(x.shape[0], layer.output_size, dtype=x.dtype)

        tp_size = 1
        batch = 3
        input_size = 8
        output_size = 4

        with mock.patch.object(vllm_linear, "get_tensor_model_parallel_world_size", return_value=tp_size), \
             mock.patch.object(vllm_linear, "get_tensor_model_parallel_rank", return_value=0):
            layer = RowParallelLinearWithReduceScatter(
                input_size=input_size,
                output_size=output_size,
                bias=False,
                input_is_parallel=True,
                skip_bias_add=False,
                params_dtype=torch.float32,
                reduce_results=True,
                quant_config=None,
                prefix="rs.",
            )
        qm = CaptureQuantMethod()
        layer.quant_method = qm

        x = torch.randn(batch, input_size, dtype=torch.float32)

        layer_mod = importlib.import_module(RowParallelLinearWithReduceScatter.__module__)

        def failing_reduce_scatter(tensor, comm_group=None):
            raise AssertionError(
                "mla_tensor_model_parallel_reduce_scatter should not be called when tp_size==1"
            )

        with mock.patch.object(layer_mod, "mla_tensor_model_parallel_reduce_scatter",
                               side_effect=failing_reduce_scatter):
            out, out_bias = layer(x)

        # 验证 quant_method.apply 的输入和 bias
        self.assertIs(qm.last_layer, layer)
        self.assertTrue(torch.allclose(qm.last_input, x))
        self.assertIsNone(qm.last_bias)

        # 输出形状正确，且没有第二个 bias 输出
        self.assertEqual(out.shape, (batch, output_size))
        self.assertIsNone(out_bias)

    def test_row_parallel_linear_with_reduce_scatter_forward_input_not_parallel_selects_tp_rank_slice(self):
        """
        input_is_parallel=False 时，应通过 split_tensor_along_last_dim 切分，
        并选择索引为 tp_rank 的那一片作为 input_parallel。
        """
        from omni.layers.linear import RowParallelLinearWithReduceScatter
        import vllm.model_executor.layers.linear as vllm_linear
        from vllm.distributed import split_tensor_along_last_dim

        class CaptureQuantMethod:
            def __init__(self):
                self.last_input = None
                self.last_bias = None

            def apply(self, layer, x, bias=None):
                self.last_input = x.clone()
                self.last_bias = bias
                return torch.zeros(x.shape[0], layer.output_size, dtype=x.dtype)

        tp_size = 3
        tp_rank = 1
        batch = 2
        input_size = 9
        output_size = 5

        with mock.patch.object(vllm_linear, "get_tensor_model_parallel_world_size", return_value=tp_size), \
             mock.patch.object(vllm_linear, "get_tensor_model_parallel_rank", return_value=tp_rank):
            layer = RowParallelLinearWithReduceScatter(
                input_size=input_size,
                output_size=output_size,
                bias=False,
                input_is_parallel=False,   # 关键：触发 split 逻辑
                skip_bias_add=False,
                params_dtype=torch.float32,
                reduce_results=False,      # 避免 reduce_scatter
                quant_config=None,
                prefix="rs.",
            )
        qm = CaptureQuantMethod()
        layer.quant_method = qm

        x = torch.arange(batch * input_size, dtype=torch.float32).reshape(batch, input_size)

        out, out_bias = layer(x)

        # 计算期望的被选中的分片
        splits = split_tensor_along_last_dim(x, num_partitions=tp_size)
        expected_parallel = splits[tp_rank].contiguous()

        self.assertTrue(torch.allclose(qm.last_input, expected_parallel))
        self.assertIsNone(qm.last_bias)
        self.assertEqual(out.shape, (batch, output_size))
        self.assertIsNone(out_bias)

    def test_row_parallel_linear_with_reduce_scatter_forward_reduce_results_true_tp_gt1_calls_mla_reduce_scatter(self):
        """
        tp_size>1 且 reduce_results=True 时，应调用 mla_tensor_model_parallel_reduce_scatter。
        """
        from omni.layers.linear import RowParallelLinearWithReduceScatter
        import vllm.model_executor.layers.linear as vllm_linear
        import importlib

        class DummyQuantMethod:
            def apply(self, layer, x, bias=None):
                # 返回一个全 1 的张量，方便检查传入 reduce_scatter 的内容
                return torch.ones(x.shape[0], layer.output_size, dtype=x.dtype)

        tp_size = 2
        tp_rank = 0
        batch = 4
        input_size = 10
        output_size = 6

        with mock.patch.object(vllm_linear, "get_tensor_model_parallel_world_size", return_value=tp_size), \
             mock.patch.object(vllm_linear, "get_tensor_model_parallel_rank", return_value=tp_rank):
            layer = RowParallelLinearWithReduceScatter(
                input_size=input_size,
                output_size=output_size,
                bias=False,
                input_is_parallel=True,
                skip_bias_add=False,
                params_dtype=torch.float32,
                reduce_results=True,      # 关键：触发 reduce_scatter 分支
                quant_config=None,
                prefix="rs.",
            )
        layer.quant_method = DummyQuantMethod()

        x = torch.randn(batch, input_size, dtype=torch.float32)

        layer_mod = importlib.import_module(RowParallelLinearWithReduceScatter.__module__)
        captured = {"called": False, "tensor": None, "comm_group": None}

        def fake_mla_reduce_scatter(tensor, comm_group=None):
            captured["called"] = True
            captured["tensor"] = tensor.clone()
            captured["comm_group"] = comm_group
            return tensor * 2

        dummy_group = object()

        with mock.patch.object(layer_mod, "mla_tensor_model_parallel_reduce_scatter",
                               side_effect=fake_mla_reduce_scatter):
            out, out_bias = layer(x, comm_group=dummy_group)

        self.assertTrue(captured["called"])
        self.assertTrue(torch.allclose(captured["tensor"],
                                       torch.ones(batch, output_size)))
        self.assertIs(captured["comm_group"], dummy_group)
        self.assertEqual(out.shape, (batch, output_size))
        # skip_bias_add=False & bias=None -> output_bias 应该为 None
        self.assertIsNone(out_bias)

    def test_row_parallel_linear_with_reduce_scatter_forward_reduce_results_false_tp_gt1_skips_mla_reduce_scatter(self):
        """
        tp_size>1 且 reduce_results=False 时，不应调用 mla_tensor_model_parallel_reduce_scatter，
        直接返回 quant_method.apply 的结果。
        """
        from omni.layers.linear import RowParallelLinearWithReduceScatter
        import vllm.model_executor.layers.linear as vllm_linear
        import importlib

        class DummyQuantMethod:
            def apply(self, layer, x, bias=None):
                return torch.zeros(x.shape[0], layer.output_size, dtype=x.dtype)

        tp_size = 2
        tp_rank = 0
        batch = 3
        input_size = 8
        output_size = 5

        with mock.patch.object(vllm_linear, "get_tensor_model_parallel_world_size", return_value=tp_size), \
             mock.patch.object(vllm_linear, "get_tensor_model_parallel_rank", return_value=tp_rank):
            layer = RowParallelLinearWithReduceScatter(
                input_size=input_size,
                output_size=output_size,
                bias=False,
                input_is_parallel=True,
                skip_bias_add=False,
                params_dtype=torch.float32,
                reduce_results=False,     # 关键：不走 reduce_scatter 分支
                quant_config=None,
                prefix="rs.",
            )
        layer.quant_method = DummyQuantMethod()

        x = torch.randn(batch, input_size, dtype=torch.float32)

        layer_mod = importlib.import_module(RowParallelLinearWithReduceScatter.__module__)

        def failing_mla_reduce_scatter(tensor, comm_group=None):
            raise AssertionError(
                "mla_tensor_model_parallel_reduce_scatter should not be called when reduce_results=False"
            )

        with mock.patch.object(layer_mod, "mla_tensor_model_parallel_reduce_scatter",
                               side_effect=failing_mla_reduce_scatter):
            out, out_bias = layer(x)

        self.assertEqual(out.shape, (batch, output_size))
        self.assertTrue(torch.all(out == 0))
        self.assertIsNone(out_bias)

# ================= MergedReplicatedLinear =================

    # ---------- helper ----------

    def _make_merged_replicated_linear_layer(
        self,
        input_size=4,
        output_sizes=None,
        bias=True,
        skip_bias_add=False,
        params_dtype=torch.float32,
        prefix="merged_repl.",
    ):
        if output_sizes is None:
            output_sizes = [4, 4]

        from omni.layers.linear import MergedReplicatedLinear
        import vllm.model_executor.layers.linear as vllm_linear

        # ReplicatedLinear 依赖 TP world_size/rank，这里用 patch 模拟单卡环境
        with mock.patch.object(
            vllm_linear, "get_tensor_model_parallel_world_size", return_value=1
        ), mock.patch.object(
            vllm_linear, "get_tensor_model_parallel_rank", return_value=0
        ):
            layer = MergedReplicatedLinear(
                input_size=input_size,
                output_sizes=output_sizes,
                bias=bias,
                skip_bias_add=skip_bias_add,
                params_dtype=params_dtype,
                quant_config=None,
                prefix=prefix,
            )
        return layer

    # ---------- weight_loader: GGUF weight_type ----------

    def test_merged_replicated_linear_weight_loader_gguf_weight_type_sets_data_and_type(self):
        layer = self._make_merged_replicated_linear_layer(output_sizes=[4, 4])

        param = Parameter(torch.zeros(3))
        param.is_gguf_weight_type = True
        param.is_gguf_weight = False
        param.shard_weight_type = {}

        loaded_weight = torch.tensor(7.0)

        loaded_shard_id = 1
        layer.weight_loader(param, loaded_weight, loaded_shard_id=loaded_shard_id)

        self.assertAlmostEqual(param.data[loaded_shard_id].item(), 7.0)
        self.assertIn(loaded_shard_id, param.shard_weight_type)
        self.assertEqual(param.shard_weight_type[loaded_shard_id], 7.0)

    # ---------- weight_loader: GGUF 权重分片 & materialize_nested ----------

    def test_merged_replicated_linear_weight_loader_gguf_weight_second_shard_materializes_qweight(self):
        layer = self._make_merged_replicated_linear_layer(output_sizes=[6, 6])

        param = Parameter(torch.empty(0))
        param.is_gguf_weight = True
        param.is_gguf_weight_type = False
        param.output_dim = 1
        param.shard_id = []
        param.shard_id_map = {}
        param.data_container = []
        param.materialize_nested = MagicMock(return_value="nested_qweight")

        loaded_weight = torch.randn(4, 6)

        layer.weight_loader(param, loaded_weight, loaded_shard_id=0)
        layer.weight_loader(param, loaded_weight, loaded_shard_id=1)

        self.assertEqual(param.shard_id, [0, 1])
        self.assertEqual(param.shard_id_map, {0: 0, 1: 1})
        self.assertEqual(len(param.data_container), 2)
        param.materialize_nested.assert_called_once()
        self.assertEqual(layer.qweight, "nested_qweight")

    # ---------- weight_loader: fused, loaded_shard_id is None & output_dim is None ----------

    def test_merged_replicated_linear_weight_loader_fused_no_output_dim_simple_copy(self):
        layer = self._make_merged_replicated_linear_layer(output_sizes=[4, 4])

        param = Parameter(torch.zeros(2, 3))
        loaded_weight = torch.ones(2, 3)

        layer.weight_loader(param, loaded_weight, loaded_shard_id=None)

        self.assertTrue(torch.allclose(param.data, loaded_weight))


    def test_merged_replicated_linear_weight_loader_fused_no_output_dim_shape_mismatch_raises(self):
        layer = self._make_merged_replicated_linear_layer(output_sizes=[4, 4])

        param = Parameter(torch.zeros(2, 3))
        loaded_weight = torch.zeros(4, 5)

        with self.assertRaisesRegex(RuntimeError, "param_data.shape != loaded_weight.shape"):
            layer.weight_loader(param, loaded_weight, loaded_shard_id=None)

    # ---------- weight_loader: fused, loaded_shard_id is None & output_dim is not None ----------

    def test_merged_replicated_linear_weight_loader_fused_with_output_dim_unpacked_recursive_shards(self):
        layer = self._make_merged_replicated_linear_layer(output_sizes=[4, 2])

        param = Parameter(torch.zeros(3, 6))
        param.output_dim = 1

        loaded_weight = torch.arange(18, dtype=torch.float32).reshape(3, 6)

        calls = []
        orig_loader = layer.weight_loader

        def spy_loader(p, lw, loaded_shard_id=None):
            if loaded_shard_id is not None:
                calls.append((loaded_shard_id, lw.clone()))
                return
            return orig_loader(p, lw, loaded_shard_id)

        with patch.object(layer, "weight_loader", side_effect=spy_loader):
            layer.weight_loader(param, loaded_weight, loaded_shard_id=None)

        # 应该拆成两个 shard：宽度 4 和 2
        self.assertEqual(len(calls), 2)
        calls.sort(key=lambda x: x[0])
        shard0_id, shard0_weight = calls[0]
        shard1_id, shard1_weight = calls[1]

        self.assertEqual(shard0_id, 0)
        self.assertEqual(shard1_id, 1)
        self.assertEqual(tuple(shard0_weight.shape), (3, 4))
        self.assertEqual(tuple(shard1_weight.shape), (3, 2))
        self.assertTrue(torch.allclose(shard0_weight, loaded_weight[:, :4]))
        self.assertTrue(torch.allclose(shard1_weight, loaded_weight[:, 4:]))

    def test_merged_replicated_linear_weight_loader_fused_with_output_dim_packed_and_marlin_adjust(self):
        output_sizes = [4, 4]
        layer = self._make_merged_replicated_linear_layer(output_sizes=output_sizes)

        param = Parameter(torch.zeros(3, 4))
        param.output_dim = 1
        param.packed_dim = 1
        param.pack_factor = 2

        loaded_weight = torch.randn(3, 4)

        with patch("omni.layers.linear.adjust_marlin_shard",
                   side_effect=lambda p, size, offset: (size, offset)) as mock_adjust:
            orig_loader = layer.weight_loader

            def spy_loader(p, lw, loaded_shard_id=None):
                if loaded_shard_id is not None:
                    # 子分片的递归不再深入
                    return
                return orig_loader(p, lw, loaded_shard_id)

            with patch.object(layer, "weight_loader", side_effect=spy_loader):
                layer.weight_loader(param, loaded_weight, loaded_shard_id=None)

        # 两个 shard，每个在 pack_factor=2 后 size=2, offset=0/2
        self.assertEqual(mock_adjust.call_count, len(output_sizes))
        sizes_offsets = [(c.args[1], c.args[2]) for c in mock_adjust.call_args_list]
        self.assertIn((2, 0), sizes_offsets)
        self.assertIn((2, 2), sizes_offsets)

    # ---------- weight_loader: sharded, loaded_shard_id is not None ----------

    def test_merged_replicated_linear_weight_loader_invalid_loaded_shard_id_raises(self):
        layer = self._make_merged_replicated_linear_layer(output_sizes=[4, 4])

        param = Parameter(torch.zeros(2, 2))
        loaded_weight = torch.zeros(2, 2)

        with self.assertRaisesRegex(RuntimeError, "loaded_shard_id >= len\\(self.output_sizes\\)"):
            layer.weight_loader(param, loaded_weight, loaded_shard_id=2)

    def test_merged_replicated_linear_weight_loader_sharded_with_output_dim_basic_slice_and_copy(self):
        output_sizes = [4, 4]
        layer = self._make_merged_replicated_linear_layer(output_sizes=output_sizes)

        param = Parameter(torch.zeros(2, 8))
        param.output_dim = 1

        full_weight = torch.arange(16, dtype=torch.float32).reshape(2, 8)
        # 对 shard_id=1 来说，loaded_weight 是第二段 [4:8]
        loaded_weight_shard = full_weight[:, 4:8].clone()

        layer.weight_loader(param, loaded_weight_shard, loaded_shard_id=1)

        # param 的后 4 列应被写入 loaded_weight_shard
        self.assertTrue(torch.allclose(param.data[:, 4:8], loaded_weight_shard))
        self.assertTrue(torch.all(param.data[:, :4] == 0))

    def test_merged_replicated_linear_weight_loader_sharded_with_output_dim_bitsandbytes_4bit(self):
        output_sizes = [4, 4]
        layer = self._make_merged_replicated_linear_layer(output_sizes=output_sizes)

        param = Parameter(torch.zeros(2, 8))
        param.output_dim = 1
        param.use_bitsandbytes_4bit = True

        loaded_weight = torch.arange(8, dtype=torch.float32).reshape(2, 4)

        layer.weight_loader(param, loaded_weight, loaded_shard_id=1)

        # use_bitsandbytes_4bit -> shard_size = loaded_weight.shape[1] = 4
        # shard_offset = 4 * loaded_shard_id = 4
        self.assertTrue(torch.allclose(param.data[:, 4:8], loaded_weight))
        self.assertTrue(torch.all(param.data[:, :4] == 0))

    # ---------- weight_loader: sharded, output_dim is None ----------

    def test_merged_replicated_linear_weight_loader_sharded_metadata_without_output_dim(self):
        layer = self._make_merged_replicated_linear_layer(output_sizes=[4, 4])

        param = Parameter(torch.zeros(4, 2))
        param.is_metadata = True

        loaded_weight = torch.ones(2, 2)

        # loaded_shard_id=1 -> shard_offset = 1 * shard_size(2) = 2
        layer.weight_loader(param, loaded_weight, loaded_shard_id=1)

        self.assertTrue(torch.allclose(param.data[2:4], loaded_weight))
        self.assertTrue(torch.all(param.data[:2] == 0))

    def test_merged_replicated_linear_weight_loader_sharded_needs_scalar_to_array_without_output_dim(self):
        layer = self._make_merged_replicated_linear_layer(output_sizes=[4, 4])

        param = Parameter(torch.zeros(1))
        param.needs_scalar_to_array = True

        loaded_weight = torch.tensor(2.0)

        with patch("omni.layers.linear.adjust_scalar_to_fused_array") as mock_adjust:
            new_loaded = torch.tensor([5.0])
            mock_adjust.return_value = (param.data, new_loaded)

            layer.weight_loader(param, loaded_weight, loaded_shard_id=0)

        mock_adjust.assert_called_once()
        self.assertTrue(torch.equal(param.data, new_loaded))

    def test_merged_replicated_linear_weight_loader_sharded_without_output_dim_final_shape_mismatch_raises(self):
        layer = self._make_merged_replicated_linear_layer(output_sizes=[4, 4])

        param = Parameter(torch.zeros(2, 2))
        loaded_weight = torch.zeros(3, 3)

        with patch("omni.layers.linear.logger") as mock_logger:
            with self.assertRaisesRegex(RuntimeError, "param_data.shape != loaded_weight.shape"):
                layer.weight_loader(param, loaded_weight, loaded_shard_id=0)

            mock_logger.warning.assert_called_once()

# ================= RowParallelLinearCross =================

    def test_row_parallel_linear_cross_init_not_reduce_with_bias_raises(self):
        """
        当 reduce_results=False 且 bias=True 且 skip_bias_add=False 时应报错，
        防止在未 reduce 的场景下多次加 bias。
        """
        from omni.layers.linear import RowParallelLinearCross

        with self.assertRaises(ValueError):
            RowParallelLinearCross(
                input_size=12,   # 保证能被 tp_size 整除
                output_size=4,
                bias=True,
                tp_size=2,
                tp_rank=0,
                input_is_parallel=True,
                skip_bias_add=False,
                params_dtype=torch.float32,
                reduce_results=False,
                quant_config=None,
                prefix="rpcross.",
            )

    # ---------- weight_loader 分支 ----------

    def test_row_parallel_linear_cross_weight_loader_input_dim_sliced_by_tp_rank(self):
        """
        input_dim 不为 None 且不使用 bitsandbytes 时，
        应按 tp_rank 沿 input_dim 维切片。
        """
        from omni.layers.linear import RowParallelLinearCross
        from torch.nn.parameter import Parameter

        input_size = 8
        tp_size = 2
        tp_rank = 1  # 取第二段

        layer = RowParallelLinearCross(
            input_size=input_size,
            output_size=4,
            bias=False,
            tp_size=tp_size,
            tp_rank=tp_rank,
            input_is_parallel=True,
            skip_bias_add=False,
            params_dtype=torch.float32,
            reduce_results=True,
            quant_config=None,
            prefix="rpcross.",
        )

        # param_data 只存放本 rank 的 shard，宽度 = input_size / tp_size = 4
        param = Parameter(torch.zeros(3, 4, dtype=torch.float32))
        param.input_dim = 1

        # loaded_weight 包含所有 rank 的数据： dim=1 为 8
        loaded_weight = torch.arange(3 * input_size,
                                     dtype=torch.float32).reshape(3, input_size)

        layer.weight_loader(param, loaded_weight)

        # 对于 tp_rank=1，期望拿到后半部分 [:, 4:8]
        expected = loaded_weight[:, 4:8]
        self.assertTrue(torch.allclose(param.data, expected))



    def test_row_parallel_linear_cross_weight_loader_gguf_uninitialized_materialize_with_input_dim(self):
        """
        is_gguf_weight=True 且 param 为 UninitializedParameter、带 input_dim 时：
        应按 tp_size 缩小对应维度并 materialize，随后再按 tp_rank 沿该维度切片。
        """
        from omni.layers.linear import RowParallelLinearCross
        from torch.nn.parameter import UninitializedParameter

        input_size = 8
        tp_size = 2
        tp_rank = 0  # 拿第一半

        layer = RowParallelLinearCross(
            input_size=input_size,
            output_size=4,
            bias=False,
            tp_size=tp_size,
            tp_rank=tp_rank,
            input_is_parallel=True,
            skip_bias_add=False,
            params_dtype=torch.float32,
            reduce_results=True,
            quant_config=None,
            prefix="rpcross.",
        )

        param = UninitializedParameter(requires_grad=False)
        param.is_gguf_weight = True
        param.input_dim = 1

        loaded_weight = torch.arange(24, dtype=torch.float32).reshape(3, 8)
        expected = loaded_weight[:, :4].clone()  # 期望拿到前一半

        layer.weight_loader(param, loaded_weight)

        # materialize 后形状在 input_dim 维度被除以 tp_size
        self.assertEqual(tuple(param.data.shape), (3, 4))
        self.assertTrue(torch.allclose(param.data, expected))

    # ---------- forward 分支 ----------

    def test_row_parallel_linear_cross_forward_input_parallel_bias_fused_on_rank0(self):
        """
        input_is_parallel=True，rank0 且不 skip_bias_add 时，
        bias 应融合进 GEMM（传给 quant_method.apply），第二输出为 None。
        """
        from omni.layers.linear import RowParallelLinearCross

        class CaptureQuantMethod:
            def __init__(self):
                self.last_layer = None
                self.last_input = None
                self.last_bias = None

            def create_weights(self, layer, input_size_per_partition,
                               output_partition_sizes, input_size, output_size,
                               params_dtype, weight_loader):
                return

            def apply(self, layer, x, bias=None):
                self.last_layer = layer
                self.last_input = x.clone()
                self.last_bias = bias
                return torch.zeros(x.shape[0], layer.output_size,
                                   dtype=x.dtype)

        input_size = 8
        output_size = 4

        layer = RowParallelLinearCross(
            input_size=input_size,
            output_size=output_size,
            bias=True,
            tp_size=2,
            tp_rank=0,
            input_is_parallel=True,
            skip_bias_add=False,
            params_dtype=torch.float32,
            reduce_results=True,
            quant_config=None,
            prefix="rpcross.",
        )
        qm = CaptureQuantMethod()
        layer.quant_method = qm

        x = torch.randn(5, input_size, dtype=torch.float32)
        out, out_bias = layer(x)

        self.assertTrue(torch.allclose(qm.last_input, x))
        self.assertIs(qm.last_bias, layer.bias)
        self.assertEqual(out.shape, (5, output_size))
        self.assertIsNone(out_bias)

    def test_row_parallel_linear_cross_forward_input_not_parallel_splits_by_tp_rank(self):
        """
        input_is_parallel=False 时，应使用 split_tensor_along_last_dim 按
        TP rank 切分输入，并只将当前 rank 片段传入 quant_method.apply。
        """
        from omni.layers.linear import RowParallelLinearCross
        import importlib
        from vllm.distributed import split_tensor_along_last_dim

        class CaptureQuantMethod:
            def __init__(self):
                self.last_input = None
                self.last_bias = None

            def create_weights(self, layer, input_size_per_partition,
                               output_partition_sizes, input_size, output_size,
                               params_dtype, weight_loader):
                return

            def apply(self, layer, x, bias=None):
                self.last_input = x.clone()
                self.last_bias = bias
                return torch.zeros(x.shape[0], layer.output_size,
                                   dtype=x.dtype)

        input_size = 12
        output_size = 5
        tp_size = 3
        tp_rank = 1  # 取第 1 片

        layer = RowParallelLinearCross(
            input_size=input_size,
            output_size=output_size,
            bias=False,
            tp_size=tp_size,
            tp_rank=tp_rank,
            input_is_parallel=False,   # 关键：走 DP→TP split 路径
            skip_bias_add=False,
            params_dtype=torch.float32,
            reduce_results=True,
            quant_config=None,
            prefix="rpcross.",
        )
        qm = CaptureQuantMethod()
        layer.quant_method = qm

        batch = 2
        x = torch.arange(batch * input_size,
                         dtype=torch.float32).reshape(batch, input_size)

        layer_mod = importlib.import_module(RowParallelLinearCross.__module__)

        # forward 内部调用的是 omni.layers.linear 里的 get_tensor_model_parallel_rank
        with mock.patch.object(layer_mod, "get_tensor_model_parallel_rank",
                               return_value=tp_rank):
            out, out_bias = layer(x)

        splits = split_tensor_along_last_dim(x, num_partitions=tp_size)
        expected_parallel = splits[tp_rank].contiguous()

        self.assertTrue(torch.allclose(qm.last_input, expected_parallel))
        # bias=False -> 传入 apply 的 bias 应为 None，第二输出也为 None
        self.assertIsNone(qm.last_bias)
        self.assertIsNone(out_bias)
        self.assertEqual(out.shape, (batch, output_size))

    def test_row_parallel_linear_cross_forward_skip_bias_add_true_returns_output_and_bias(self):
        """
        skip_bias_add=True 时，不向 GEMM 传 bias，第二输出返回 bias。
        """
        from omni.layers.linear import RowParallelLinearCross

        class CaptureQuantMethod:
            def __init__(self):
                self.last_bias = None

            def create_weights(self, layer, input_size_per_partition,
                               output_partition_sizes, input_size, output_size,
                               params_dtype, weight_loader):
                return

            def apply(self, layer, x, bias=None):
                self.last_bias = bias
                return torch.zeros(x.shape[0], layer.output_size,
                                   dtype=x.dtype)

        input_size = 8
        output_size = 4
        batch = 3

        layer = RowParallelLinearCross(
            input_size=input_size,
            output_size=output_size,
            bias=True,
            tp_size=1,
            tp_rank=0,
            input_is_parallel=True,
            skip_bias_add=True,   # 关键
            params_dtype=torch.float32,
            reduce_results=True,
            quant_config=None,
            prefix="rpcross.",
        )
        qm = CaptureQuantMethod()
        layer.quant_method = qm

        x = torch.randn(batch, input_size, dtype=torch.float32)
        out, out_bias = layer(x)

        # skip_bias_add=True -> GEMM 不融合 bias
        self.assertIsNone(qm.last_bias)
        # 第二输出返回 bias
        self.assertIs(out_bias, layer.bias)
        self.assertEqual(out.shape, (batch, output_size))

# ================= FlashCommLinearMethodBase =================

    def test_flash_comm_linear_method_base_subclass_apply_basic(self):
        """
        子类实现 apply：不传可选参数时，能够正常被调用并返回 tensor，
        且保持输入形状不变。
        """
        from omni.layers.linear import FlashCommLinearMethodBase

        class DummyLayer(torch.nn.Module):
            def __init__(self, output_size: int):
                super().__init__()
                self.output_size = output_size

        class DummyFlashCommMethod(FlashCommLinearMethodBase):
            def __init__(self):
                # 不调用父类 __init__，避免依赖 LinearMethodBase 的实参签名
                pass

            def create_weights(
                self,
                layer,
                input_size_per_partition,
                output_partition_sizes,
                input_size,
                output_size,
                params_dtype,
                weight_loader,
            ):
                # UT 中不需要真实权重，空实现即可
                return

            def apply(self,
                      layer,
                      x: torch.Tensor,
                      bias: Optional[torch.Tensor] = None,
                      module_name: Optional[str] = "",
                      x_transform: Optional[str] = None) -> torch.Tensor:
                # 记录一下传入参数，后面断言用
                self.last_layer = layer
                self.last_x = x.clone()
                self.last_bias = bias
                self.last_module_name = module_name
                self.last_x_transform = x_transform
                # 简单返回 2x，保证形状不变即可
                return x * 2

        out_features = 4
        batch = 3
        layer = DummyLayer(output_size=out_features)
        method = DummyFlashCommMethod()
        x = torch.randn(batch, out_features, dtype=torch.float32)

        out = method.apply(layer, x)

        # 形状一致
        self.assertEqual(tuple(out.shape), tuple(x.shape))
        # 内容按预期（这里简单检查 2x）
        self.assertTrue(torch.allclose(out, x * 2))

        # 验证必选参数和默认可选参数传递正确
        self.assertIs(method.last_layer, layer)
        self.assertTrue(torch.allclose(method.last_x, x))
        self.assertIsNone(method.last_bias)
        self.assertEqual(method.last_module_name, "")
        self.assertIsNone(method.last_x_transform)

    def test_flash_comm_linear_method_base_subclass_apply_with_optional_args(self):
        """
        子类实现 apply：在传入 bias / module_name / x_transform 时，
        能够正常接收这些参数并参与计算（这里简单做 x + bias）。
        """
        from omni.layers.linear import FlashCommLinearMethodBase

        class DummyLayer(torch.nn.Module):
            def __init__(self, output_size: int):
                super().__init__()
                self.output_size = output_size

        class DummyFlashCommMethod(FlashCommLinearMethodBase):
            def __init__(self):
                # 同样避免依赖父类 __init__ 的参数签名
                pass

            def create_weights(
                self,
                layer,
                input_size_per_partition,
                output_partition_sizes,
                input_size,
                output_size,
                params_dtype,
                weight_loader,
            ):
                return

            def apply(self,
                      layer,
                      x: torch.Tensor,
                      bias: Optional[torch.Tensor] = None,
                      module_name: Optional[str] = "",
                      x_transform: Optional[str] = None) -> torch.Tensor:
                # 记录所有参数
                self.last_layer = layer
                self.last_x = x.clone()
                self.last_bias = bias
                self.last_module_name = module_name
                self.last_x_transform = x_transform

                # 简单逻辑：如果有 bias 就加上（测试广播 & 参与计算即可）
                if bias is not None:
                    return x + bias
                return x

        out_features = 5
        batch = 2
        layer = DummyLayer(output_size=out_features)
        method = DummyFlashCommMethod()

        x = torch.zeros(batch, out_features, dtype=torch.float32)
        bias = torch.ones(out_features, dtype=torch.float32)
        module_name = "o_proj"
        x_transform = "transpose"

        out = method.apply(
            layer,
            x,
            bias=bias,
            module_name=module_name,
            x_transform=x_transform,
        )

        # 输出 = x + bias（广播到 batch 维度）
        expected = x + bias
        self.assertTrue(torch.allclose(out, expected))

        # 参数传递检查
        self.assertIs(method.last_layer, layer)
        self.assertTrue(torch.allclose(method.last_x, x))
        self.assertIs(method.last_bias, bias)
        self.assertEqual(method.last_module_name, module_name)
        self.assertEqual(method.last_x_transform, x_transform)
        # 形状保持一致
        self.assertEqual(tuple(out.shape), (batch, out_features))

# ================= UnquantizedFlashCommLinearMethod =================

    def test_unquantized_flash_comm_linear_method_create_weights_basic(self):
        from torch.nn.parameter import Parameter

        class DummyLayer(torch.nn.Module):
            def __init__(self):
                super().__init__()

        layer = DummyLayer()
        method = UnquantizedFlashCommLinearMethod()

        input_size_per_partition = 4
        output_partition_sizes = [3, 5]
        input_size = 8
        output_size = sum(output_partition_sizes)
        params_dtype = torch.float32

        # 传入一个额外的属性，验证 extra_weight_attrs 也被设置到了 weight 上
        method.create_weights(
            layer=layer,
            input_size_per_partition=input_size_per_partition,
            output_partition_sizes=output_partition_sizes,
            input_size=input_size,
            output_size=output_size,
            params_dtype=params_dtype,
            extra_flag="test_flag",
        )

        # 验证 weight 被注册
        self.assertTrue(hasattr(layer, "weight"))
        self.assertIsInstance(layer.weight, Parameter)

        # 形状: [sum(output_partition_sizes), input_size_per_partition]
        self.assertEqual(
            tuple(layer.weight.shape),
            (sum(output_partition_sizes), input_size_per_partition),
        )

        # set_weight_attrs 设置的基本属性
        self.assertEqual(getattr(layer.weight, "input_dim", None), 1)
        self.assertEqual(getattr(layer.weight, "output_dim", None), 0)
        # 额外属性
        self.assertEqual(getattr(layer.weight, "extra_flag", None), "test_flag")

    def test_unquantized_flash_comm_linear_method_process_weights_after_loading_transposes_and_marks_weight(self):
        from torch.nn.parameter import Parameter

        layer_mod = importlib.import_module(UnquantizedFlashCommLinearMethod.__module__)

        class DummyLayer(torch.nn.Module):
            def __init__(self, w):
                super().__init__()
                self.weight = Parameter(w)

        in_features = 4
        out_features = 6
        # 初始按照 create_weights 之后的形状: [out, in]
        w = torch.arange(out_features * in_features, dtype=torch.float32).reshape(
            out_features, in_features
        )
        layer = DummyLayer(w.clone())

        method = UnquantizedFlashCommLinearMethod()

        captured = {}

        def fake_npu_format_cast(t, fmt):
            captured["input"] = t.clone()
            captured["format"] = fmt
            # 模拟 format_cast，不改变数据
            return t

        # 打补丁 torch_npu.npu_format_cast，避免真实 NPU 依赖
        with mock.patch.object(layer_mod, "torch_npu") as mock_npu:
            mock_npu.npu_format_cast.side_effect = fake_npu_format_cast
            method.process_weights_after_loading(layer)

        # 验证传入 npu_format_cast 的是转置后的 weight
        self.assertIn("input", captured)
        self.assertTrue(torch.allclose(captured["input"], w.t().contiguous()))
        self.assertEqual(captured["format"], 29)

        # 处理后权重形状应为 [in, out]
        self.assertEqual(tuple(layer.weight.shape), (in_features, out_features))
        # 并打上 is_weight_transposed 标记
        self.assertTrue(getattr(layer.weight, "is_weight_transposed"))

    def test_unquantized_flash_comm_linear_method_apply_with_bias_2d(self):
        from torch.nn.parameter import Parameter

        in_features = 6
        out_features = 4
        weight = torch.randn(in_features, out_features, dtype=torch.float32)

        class DummyLayer(torch.nn.Module):
            def __init__(self, w):
                super().__init__()
                self.weight = Parameter(w)

        layer = DummyLayer(weight.clone())
        method = UnquantizedFlashCommLinearMethod()

        batch = 3
        x = torch.randn(batch, in_features, dtype=torch.float32)
        bias = torch.randn(out_features, dtype=torch.float32)

        out = method.apply(layer, x, bias=bias)

        # 2D 情况下应走 torch.addmm 分支
        expected = torch.addmm(bias, x, weight)
        self.assertTrue(torch.allclose(out, expected, atol=1e-5))

    def test_unquantized_flash_comm_linear_method_apply_with_bias_3d(self):
        from torch.nn.parameter import Parameter


        in_features = 5
        out_features = 7
        weight = torch.randn(in_features, out_features, dtype=torch.float32)

        class DummyLayer(torch.nn.Module):
            def __init__(self, w):
                super().__init__()
                self.weight = Parameter(w)

        layer = DummyLayer(weight.clone())
        method = UnquantizedFlashCommLinearMethod()

        batch = 2
        seq = 4
        x = torch.randn(batch, seq, in_features, dtype=torch.float32)
        bias = torch.randn(out_features, dtype=torch.float32)

        out = method.apply(layer, x, bias=bias)

        # 3D 情况下应走 torch.matmul(x, weight) + bias 分支
        expected = torch.matmul(x, weight) + bias
        self.assertTrue(torch.allclose(out, expected, atol=1e-5))
        self.assertEqual(tuple(out.shape), (batch, seq, out_features))

    def test_unquantized_flash_comm_linear_method_apply_with_x_transform_ag(self):
        """
        x_transform == 'AG' 时应调用 get_tp_group().all_gather(x, dim=0)。
        """
        from torch.nn.parameter import Parameter

        in_features = 4
        out_features = 3
        weight = torch.randn(in_features, out_features, dtype=torch.float32)

        class DummyLayer(torch.nn.Module):
            def __init__(self, w):
                super().__init__()
                self.weight = Parameter(w)

        class DummyGroup:
            def __init__(self):
                self.called = False
                self.last_x = None
                self.last_dim = None

            def all_gather(self, x, dim=0):
                self.called = True
                self.last_x = x
                self.last_dim = dim
                # 简单返回原 tensor
                return x

        layer = DummyLayer(weight.clone())
        method = UnquantizedFlashCommLinearMethod()
        group = DummyGroup()

        batch = 3
        x = torch.randn(batch, in_features, dtype=torch.float32)

        with mock.patch.object(omni_linear_mod, "get_tp_group", return_value=group):
            out = method.apply(layer, x, bias=None, x_transform="AG")

        # 验证 all_gather 被调用且 dim=0
        self.assertTrue(group.called)
        self.assertIs(group.last_x, x)
        self.assertEqual(group.last_dim, 0)

        # 输出应等价于普通 matmul
        expected = torch.matmul(x, weight)
        self.assertTrue(torch.allclose(out, expected, atol=1e-5))

    def test_unquantized_flash_comm_linear_method_apply_with_x_transform_a2a(self):
        """
        x_transform == 'A2A' 时应调用 get_tp_group().all_to_all(x)。
        """
        from torch.nn.parameter import Parameter

        in_features = 4
        out_features = 2
        weight = torch.randn(in_features, out_features, dtype=torch.float32)

        class DummyLayer(torch.nn.Module):
            def __init__(self, w):
                super().__init__()
                self.weight = Parameter(w)

        class DummyGroup:
            def __init__(self):
                self.called = False
                self.last_x = None

            def all_to_all(self, x):
                self.called = True
                self.last_x = x
                # 简单返回原 tensor
                return x

        layer = DummyLayer(weight.clone())
        method = UnquantizedFlashCommLinearMethod()
        group = DummyGroup()

        batch = 5
        x = torch.randn(batch, in_features, dtype=torch.float32)

        with mock.patch.object(omni_linear_mod, "get_tp_group", return_value=group):
            out = method.apply(layer, x, bias=None, x_transform="A2A")

        # 验证 all_to_all 被调用
        self.assertTrue(group.called)
        self.assertIs(group.last_x, x)

        expected = torch.matmul(x, weight)
        self.assertTrue(torch.allclose(out, expected, atol=1e-5))

# ================= FlashCommLinearBase =================

    def test_flash_comm_linear_base_init_basic_uses_defaults_and_unquantized_method(self):
        """quant_config=None & params_dtype=None -> 使用默认 dtype 和 UnquantizedFlashCommLinearMethod。"""

        layer_mod = importlib.import_module(FlashCommLinearBase.__module__)

        def fake_linearbase_init(self, input_size, output_size,
                                 skip_bias_add, params_dtype,
                                 quant_config, prefix):
            # 最小实现：只存一下入参，避免依赖真实 LinearBase
            self.input_size = input_size
            self.output_size = output_size
            self.skip_bias_add = skip_bias_add
            self.params_dtype = params_dtype
            self.quant_config = quant_config
            self.prefix = prefix
            self.quant_method = None

        input_size = 16
        output_size = 32
        tp_size = 2
        tp_rank = 1

        with mock.patch.object(layer_mod.LinearBase, "__init__", fake_linearbase_init):
            layer = FlashCommLinearBase(
                input_size=input_size,
                output_size=output_size,
                tp_size=tp_size,
                tp_rank=tp_rank,
                skip_bias_add=False,
                params_dtype=None,      # 触发默认 dtype 分支
                quant_config=None,      # 触发 UnquantizedFlashCommLinearMethod 分支
                prefix="flash.",
            )

        # 属性检查
        self.assertEqual(layer.input_size, input_size)
        self.assertEqual(layer.output_size, output_size)
        self.assertEqual(layer.tp_size, tp_size)
        self.assertEqual(layer.tp_rank, tp_rank)
        self.assertFalse(layer.skip_bias_add)

        # 默认 dtype = torch.get_default_dtype()
        self.assertEqual(layer.params_dtype, torch.get_default_dtype())
        # 默认 quant_method 类型
        self.assertIsInstance(layer.quant_method, UnquantizedFlashCommLinearMethod)

    def test_flash_comm_linear_base_init_params_dtype_is_respected_when_not_none(self):
        """传入显式 params_dtype 时，应使用该 dtype，而不是默认 dtype。"""

        layer_mod = importlib.import_module(FlashCommLinearBase.__module__)

        def fake_linearbase_init(self, input_size, output_size,
                                 skip_bias_add, params_dtype,
                                 quant_config, prefix):
            self.input_size = input_size
            self.output_size = output_size
            self.skip_bias_add = skip_bias_add
            self.params_dtype = params_dtype
            self.quant_config = quant_config
            self.prefix = prefix
            self.quant_method = None

        with mock.patch.object(layer_mod.LinearBase, "__init__", fake_linearbase_init):
            layer = FlashCommLinearBase(
                input_size=8,
                output_size=4,
                tp_size=1,
                tp_rank=0,
                skip_bias_add=True,
                params_dtype=torch.float16,   # 显式 dtype
                quant_config=None,
                prefix="flash.dtype.",
            )

        self.assertEqual(layer.input_size, 8)
        self.assertEqual(layer.output_size, 4)
        self.assertTrue(layer.skip_bias_add)
        # 应为 float16 而不是默认 dtype
        self.assertEqual(layer.params_dtype, torch.float16)

    def test_flash_comm_linear_base_init_quant_config_ignore_contains_prefix_uses_unquantized_method(self):
        """
        quant_config.ignore 非空且包含 prefix 时，走 UnquantizedFlashCommLinearMethod 分支，
        且不应调用 quant_config.get_quant_method。
        """

        layer_mod = importlib.import_module(FlashCommLinearBase.__module__)

        def fake_linearbase_init(self, input_size, output_size,
                                 skip_bias_add, params_dtype,
                                 quant_config, prefix):
            self.input_size = input_size
            self.output_size = output_size
            self.skip_bias_add = skip_bias_add
            self.params_dtype = params_dtype
            self.quant_config = quant_config
            self.prefix = prefix
            self.quant_method = None

        class DummyQuantConfig:
            def __init__(self):
                # ignore 列表包含 prefix
                self.ignore = ["flash.ignore.", "other"]

            def get_quant_method(self, layer, prefix=""):
                raise AssertionError("get_quant_method should not be called when prefix is in ignore")

        quant_config = DummyQuantConfig()

        with mock.patch.object(layer_mod.LinearBase, "__init__", fake_linearbase_init):
            layer = FlashCommLinearBase(
                input_size=8,
                output_size=16,
                tp_size=2,
                tp_rank=1,
                skip_bias_add=False,
                params_dtype=torch.float32,
                quant_config=quant_config,
                prefix="flash.ignore.",   # 命中 ignore
            )

        self.assertIsInstance(layer.quant_method, UnquantizedFlashCommLinearMethod)

    def test_flash_comm_linear_base_init_quant_config_active_uses_custom_quant_method(self):
        """
        quant_config 不为 None 且 ignore 为空/不命中 prefix -> 使用 quant_config.get_quant_method 返回的对象。
        """

        layer_mod = importlib.import_module(FlashCommLinearBase.__module__)

        def fake_linearbase_init(self, input_size, output_size,
                                 skip_bias_add, params_dtype,
                                 quant_config, prefix):
            self.input_size = input_size
            self.output_size = output_size
            self.skip_bias_add = skip_bias_add
            self.params_dtype = params_dtype
            self.quant_config = quant_config
            self.prefix = prefix
            self.quant_method = None

        class DummyQuantMethod:
            pass

        class DummyQuantConfig:
            def __init__(self):
                # ignore 设置为 None -> 不触发 ignore 分支
                self.ignore = None

            def get_quant_method(self, layer, prefix=""):
                return DummyQuantMethod()

        quant_config = DummyQuantConfig()

        with mock.patch.object(layer_mod.LinearBase, "__init__", fake_linearbase_init):
            layer = FlashCommLinearBase(
                input_size=10,
                output_size=20,
                tp_size=4,
                tp_rank=2,
                skip_bias_add=True,
                params_dtype=torch.float32,
                quant_config=quant_config,      # 启用自定义 quant_method 分支
                prefix="flash.active.",
            )

        self.assertIsInstance(layer.quant_method, DummyQuantMethod)

# ================= RowParallelFlashCommLinear =================

    def test_row_parallel_flash_comm_linear_init_with_bias_sets_partitions_and_bias_attrs(self):
        """带 bias 的基本构造：分片维度、bias 参数及其属性正确。"""
        from omni.layers.linear import RowParallelFlashCommLinear

        input_size = 8
        output_size = 4
        tp_size = 2

        layer = RowParallelFlashCommLinear(
            input_size=input_size,
            output_size=output_size,
            tp_size=tp_size,
            tp_rank=0,
            bias=True,
            skip_bias_add=False,
            params_dtype=torch.float32,
            quant_config=None,
            prefix="flash_row.",
        )

        self.assertEqual(layer.input_size, input_size)
        self.assertEqual(layer.output_size, output_size)
        self.assertEqual(layer.tp_size, tp_size)
        self.assertEqual(layer.tp_rank, 0)
        self.assertEqual(layer.input_size_per_partition, input_size // tp_size)

        # bias 参数存在且 shape 正确
        self.assertIsInstance(layer.bias, Parameter)
        self.assertEqual(layer.bias.shape[0], output_size)

        # set_weight_attrs 写入的属性
        self.assertEqual(getattr(layer.bias, "output_dim", None), 0)

        bias_wl = getattr(layer.bias, "weight_loader", None)
        # 应该挂了一个 bound method，指向 layer.weight_loader
        self.assertIsNotNone(bias_wl)
        self.assertTrue(callable(bias_wl))
        # 绑定的实例应为当前 layer
        self.assertIs(getattr(bias_wl, "__self__", None), layer)
        # 绑定的函数应为 RowParallelFlashCommLinear.weight_loader
        self.assertIs(
            getattr(bias_wl, "__func__", None),
            layer.__class__.weight_loader,
        )


    def test_row_parallel_flash_comm_linear_init_without_bias_registers_no_bias(self):
        """不带 bias 时，bias 注册为 None。"""
        from omni.layers.linear import RowParallelFlashCommLinear

        layer = RowParallelFlashCommLinear(
            input_size=8,
            output_size=4,
            tp_size=2,
            tp_rank=1,
            bias=False,
            skip_bias_add=False,
            params_dtype=torch.float32,
            quant_config=None,
            prefix="flash_row.",
        )

        self.assertTrue(hasattr(layer, "bias"))
        self.assertIsNone(layer.bias)

    def test_row_parallel_flash_comm_linear_weight_loader_input_dim_slices_by_tp_rank_and_squeezes_if_not_2_dims(self):
        """
        weight_loader: 有 input_dim 且 is_2_dims=False 时，
        按 tp_rank 在 input_dim 上 narrow 分片，并在必要时 squeeze。
        """
        from omni.layers.linear import RowParallelFlashCommLinear

        # tp_size=2, 取 rank=1 片
        layer = RowParallelFlashCommLinear(
            input_size=8,
            output_size=4,
            tp_size=2,
            tp_rank=1,
            bias=False,
            skip_bias_add=False,
            params_dtype=torch.float32,
            quant_config=None,
            prefix="flash_row.",
        )

        # param_data: [2, 3] -> shard_size = 3
        param = Parameter(torch.zeros(2, 3, dtype=torch.float32))
        param.input_dim = 1          # 对 dim=1 切片
        # 不设置 is_2_dims，默认 False -> 走 squeeze 分支（但这里不会改变形状）
        loaded_weight = torch.arange(2 * 6, dtype=torch.float32).reshape(2, 6)

        layer.weight_loader(param, loaded_weight)

        # 期望取 loaded_weight[:, 3:6]
        expected = loaded_weight.narrow(1, 3, 3)
        self.assertTrue(torch.allclose(param.data, expected))

    def test_row_parallel_flash_comm_linear_weight_loader_transposed_2dims_calls_format_cast(self):
        """
        weight_loader: is_weight_transposed=True 且 is_2_dims=True 时，
        会先转置 param，再 copy，并最终调用 torch_npu.npu_format_cast。
        """
        from omni.layers.linear import RowParallelFlashCommLinear
        import importlib

        layer_mod = importlib.import_module(RowParallelFlashCommLinear.__module__)

        layer = RowParallelFlashCommLinear(
            input_size=8,
            output_size=4,
            tp_size=1,
            tp_rank=0,
            bias=False,
            skip_bias_add=False,
            params_dtype=torch.float32,
            quant_config=None,
            prefix="flash_row.",
        )

        # 初始 param_data: [3, 4]
        param = Parameter(torch.zeros(3, 4, dtype=torch.float32))
        param.is_weight_transposed = True
        param.is_2_dims = True

        # loaded_weight 需要和转置后的 param_data 形状匹配 -> [4, 3]
        loaded_weight = torch.randn(4, 3, dtype=torch.float32)

        with mock.patch.object(
            layer_mod.torch_npu, "npu_format_cast",
            side_effect=lambda t, fmt: t,
        ) as mock_cast:
            layer.weight_loader(param, loaded_weight)

        # 逻辑上最后 param.data 应为 loaded_weight.t()
        self.assertTrue(torch.allclose(param.data, loaded_weight.t()))
        mock_cast.assert_called()

    def test_row_parallel_flash_comm_linear_forward_tp1_bias_fused_and_output_bias_none(self):
        """
        tp_size=1 且 skip_bias_add=False、rank=0 时：
        bias 应融合进 GEMM（传给 quant_method），第二输出为 None。
        """
        from omni.layers.linear import RowParallelFlashCommLinear

        class CaptureQuantMethod:
            def __init__(self):
                self.last_layer = None
                self.last_input = None
                self.last_bias = None
                self.last_module_name = None
                self.last_x_transform = None

            def create_weights(
                self,
                layer,
                input_size_per_partition,
                output_partition_sizes,
                input_size,
                output_size,
                params_dtype,
                weight_loader,
            ):
                return

            def apply(self,
                      layer,
                      x,
                      bias=None,
                      module_name="",
                      x_transform=None,
                      is_prefill=True):
                self.last_layer = layer
                self.last_input = x.clone()
                self.last_bias = bias
                self.last_module_name = module_name
                self.last_x_transform = x_transform
                return torch.zeros(x.shape[0], layer.output_size, dtype=x.dtype)

        input_size = 8
        output_size = 4
        batch = 3

        layer = RowParallelFlashCommLinear(
            input_size=input_size,
            output_size=output_size,
            tp_size=1,
            tp_rank=0,
            bias=True,
            skip_bias_add=False,
            params_dtype=torch.float32,
            quant_config=None,
            prefix="flash_row.",
        )
        qm = CaptureQuantMethod()
        layer.quant_method = qm  # 覆盖默认 quant_method，便于捕获调用

        x = torch.randn(batch, input_size, dtype=torch.float32)
        out, out_bias = layer(x, reduce_type="AR")

        self.assertEqual(out.shape, (batch, output_size))
        self.assertIsNone(out_bias)

        self.assertIs(qm.last_layer, layer)
        self.assertTrue(torch.allclose(qm.last_input, x))
        self.assertIs(qm.last_bias, layer.bias)
        self.assertEqual(qm.last_module_name, "flash_row.")
        self.assertIsNone(qm.last_x_transform)

    def test_row_parallel_flash_comm_linear_forward_tp_gt1_reduce_type_ar_calls_all_reduce(self):
        """
        tp_size>1 且 reduce_type='AR' 时，应调用 tensor_model_parallel_all_reduce。
        """
        from omni.layers.linear import RowParallelFlashCommLinear
        import importlib

        layer_mod = importlib.import_module(RowParallelFlashCommLinear.__module__)

        class DummyQuantMethod:
            def create_weights(
                self,
                layer,
                input_size_per_partition,
                output_partition_sizes,
                input_size,
                output_size,
                params_dtype,
                weight_loader,
            ):
                return

            def apply(self,
                      layer,
                      x,
                      bias=None,
                      module_name="",
                      x_transform=None,
                      is_prefill=True):
                return torch.ones(x.shape[0], layer.output_size, dtype=x.dtype)

        tp_size = 2
        input_size = 8
        output_size = 4
        batch = 2

        layer = RowParallelFlashCommLinear(
            input_size=input_size,
            output_size=output_size,
            tp_size=tp_size,
            tp_rank=0,
            bias=False,
            skip_bias_add=False,
            params_dtype=torch.float32,
            quant_config=None,
            prefix="flash_row.",
        )
        layer.quant_method = DummyQuantMethod()

        x = torch.randn(batch, input_size, dtype=torch.float32)

        called = {"ar": False}

        def fake_all_reduce(t):
            called["ar"] = True
            return t * 2

        def failing_reduce_scatter(t):
            raise AssertionError("reduce_scatter should not be called for reduce_type='AR'")

        with mock.patch.object(layer_mod, "tensor_model_parallel_all_reduce",
                               side_effect=fake_all_reduce), \
             mock.patch.object(layer_mod, "tensor_model_parallel_reduce_scatter",
                               side_effect=failing_reduce_scatter):
            out, out_bias = layer(x, reduce_type="AR")

        self.assertTrue(called["ar"])
        self.assertEqual(out.shape, (batch, output_size))
        self.assertIsNone(out_bias)

    def test_row_parallel_flash_comm_linear_forward_tp_gt1_reduce_type_rs_calls_reduce_scatter(self):
        """
        tp_size>1 且 reduce_type='RS' 时，应调用 tensor_model_parallel_reduce_scatter。
        """
        from omni.layers.linear import RowParallelFlashCommLinear
        import importlib

        layer_mod = importlib.import_module(RowParallelFlashCommLinear.__module__)

        class DummyQuantMethod:
            def create_weights(
                self,
                layer,
                input_size_per_partition,
                output_partition_sizes,
                input_size,
                output_size,
                params_dtype,
                weight_loader,
            ):
                return

            def apply(self,
                      layer,
                      x,
                      bias=None,
                      module_name="",
                      x_transform=None,
                      is_prefill=True):
                return torch.ones(x.shape[0], layer.output_size, dtype=x.dtype)

        tp_size = 2
        input_size = 8
        output_size = 4
        batch = 2

        layer = RowParallelFlashCommLinear(
            input_size=input_size,
            output_size=output_size,
            tp_size=tp_size,
            tp_rank=0,
            bias=False,
            skip_bias_add=False,
            params_dtype=torch.float32,
            quant_config=None,
            prefix="flash_row.",
        )
        layer.quant_method = DummyQuantMethod()

        x = torch.randn(batch, input_size, dtype=torch.float32)

        called = {"rs": False}

        def failing_all_reduce(t):
            raise AssertionError("all_reduce should not be called for reduce_type='RS'")

        def fake_reduce_scatter(t):
            called["rs"] = True
            return t * 3

        with mock.patch.object(layer_mod, "tensor_model_parallel_all_reduce",
                               side_effect=failing_all_reduce), \
             mock.patch.object(layer_mod, "tensor_model_parallel_reduce_scatter",
                               side_effect=fake_reduce_scatter):
            out, out_bias = layer(x, reduce_type="RS")

        self.assertTrue(called["rs"])
        self.assertEqual(out.shape, (batch, output_size))
        self.assertIsNone(out_bias)

    def test_row_parallel_flash_comm_linear_forward_tp_gt1_reduce_type_unknown_no_collective(self):
        """
        tp_size>1 且 reduce_type 既不是 'AR' 也不是 'RS' 时，不调用任何 collective，
        直接返回 quant_method.apply 的 output_parallel。
        """
        from omni.layers.linear import RowParallelFlashCommLinear
        import importlib

        layer_mod = importlib.import_module(RowParallelFlashCommLinear.__module__)

        class DummyQuantMethod:
            def __init__(self):
                self.ret = None

            def create_weights(
                self,
                layer,
                input_size_per_partition,
                output_partition_sizes,
                input_size,
                output_size,
                params_dtype,
                weight_loader,
            ):
                return

            def apply(self,
                      layer,
                      x,
                      bias=None,
                      module_name="",
                      x_transform=None,
                      is_prefill=True):
                self.ret = torch.ones(x.shape[0], layer.output_size, dtype=x.dtype)
                return self.ret

        layer = RowParallelFlashCommLinear(
            input_size=8,
            output_size=4,
            tp_size=2,
            tp_rank=0,
            bias=False,
            skip_bias_add=False,
            params_dtype=torch.float32,
            quant_config=None,
            prefix="flash_row.",
        )
        qm = DummyQuantMethod()
        layer.quant_method = qm

        x = torch.randn(2, 8, dtype=torch.float32)

        def failing_all_reduce(t):
            raise AssertionError("all_reduce should not be called")

        def failing_reduce_scatter(t):
            raise AssertionError("reduce_scatter should not be called")

        with mock.patch.object(layer_mod, "tensor_model_parallel_all_reduce",
                               side_effect=failing_all_reduce), \
             mock.patch.object(layer_mod, "tensor_model_parallel_reduce_scatter",
                               side_effect=failing_reduce_scatter):
            out, out_bias = layer(x, reduce_type="NONE")

        # 返回的应该就是 output_parallel 本身
        self.assertIs(out, qm.ret)
        self.assertIsNone(out_bias)

    def test_row_parallel_flash_comm_linear_forward_skip_bias_add_true_not_fuse_bias_and_return_bias(self):
        """
        skip_bias_add=True 时，GEMM 不融合 bias（传给 quant_method 的 bias=None），
        第二输出返回 bias。
        """
        from omni.layers.linear import RowParallelFlashCommLinear

        class CaptureQuantMethod:
            def __init__(self):
                self.last_bias = None

            def create_weights(
                self,
                layer,
                input_size_per_partition,
                output_partition_sizes,
                input_size,
                output_size,
                params_dtype,
                weight_loader,
            ):
                return

            def apply(self,
                      layer,
                      x,
                      bias=None,
                      module_name="",
                      x_transform=None,
                      is_prefill=True):
                self.last_bias = bias
                return torch.zeros(x.shape[0], layer.output_size, dtype=x.dtype)

        layer = RowParallelFlashCommLinear(
            input_size=8,
            output_size=4,
            tp_size=1,
            tp_rank=0,
            bias=True,
            skip_bias_add=True,
            params_dtype=torch.float32,
            quant_config=None,
            prefix="flash_row.",
        )
        qm = CaptureQuantMethod()
        layer.quant_method = qm

        x = torch.randn(3, 8, dtype=torch.float32)
        out, out_bias = layer(x, reduce_type="AR")

        self.assertEqual(out.shape, (3, 4))
        # skip_bias_add=True -> GEMM 不融合 bias
        self.assertIsNone(qm.last_bias)
        # 第二输出为 bias
        self.assertIs(out_bias, layer.bias)

    def test_row_parallel_flash_comm_linear_forward_with_next_layer_triggers_npu_prefetch(self):
        """
        forward 传入 next_layer 时，应根据 model_extra_config.operator_opt_config.attn_prefetch
        计算 prefetch_size，并对每个 next_layer 调用 torch_npu.npu_prefetch。
        """
        from omni.layers.linear import RowParallelFlashCommLinear
        import importlib

        layer_mod = importlib.import_module(RowParallelFlashCommLinear.__module__)

        class DummyQuantMethod:
            def create_weights(
                self,
                layer,
                input_size_per_partition,
                output_partition_sizes,
                input_size,
                output_size,
                params_dtype,
                weight_loader,
            ):
                return

            def apply(self,
                      layer,
                      x,
                      bias=None,
                      module_name="",
                      x_transform=None,
                      is_prefill=True):
                return torch.zeros(x.shape[0], layer.output_size, dtype=x.dtype)

        class DummyOptConfig:
            def __init__(self, attn_prefetch):
                self.attn_prefetch = attn_prefetch

        class DummyModelExtraConfig:
            def __init__(self, attn_prefetch):
                self.operator_opt_config = DummyOptConfig(attn_prefetch)

        layer = RowParallelFlashCommLinear(
            input_size=8,
            output_size=4,
            tp_size=1,   # tp_size=1，避免 collective 干扰
            tp_rank=0,
            bias=False,
            skip_bias_add=False,
            params_dtype=torch.float32,
            quant_config=None,
            prefix="flash_row.",
        )
        layer.quant_method = DummyQuantMethod()

        class DummyNextLayer:
            def __init__(self):
                self.weight = torch.empty(4, 4)

        next_layers = [DummyNextLayer(), DummyNextLayer()]

        attn_prefetch = 2
        dummy_config = DummyModelExtraConfig(attn_prefetch=attn_prefetch)
        expected_prefetch_size = attn_prefetch * 1024 * 1024

        x = torch.randn(2, 8, dtype=torch.float32)

        with mock.patch.object(layer_mod, "model_extra_config",
                               dummy_config, create=True), \
             mock.patch.object(layer_mod.torch_npu, "npu_prefetch") as mock_prefetch:
            out, out_bias = layer(x, reduce_type="AR", next_layer=next_layers)

        self.assertEqual(out.shape, (2, 4))
        self.assertIsNone(out_bias)
        self.assertEqual(mock_prefetch.call_count, len(next_layers))

        # 第三个参数应为预期的 prefetch_size
        for call in mock_prefetch.call_args_list:
            args, kwargs = call
            self.assertEqual(args[2], expected_prefetch_size)

# ================= ColumnParallelFlashCommLinear =================

    def test_column_parallel_flash_comm_linear_init_basic(self):
        """基础构造：tp_size>1 时分片维度、bias 等属性是否正确。"""
        from omni.layers.linear import ColumnParallelFlashCommLinear

        input_size = 8
        output_size = 6
        tp_size = 2

        layer = ColumnParallelFlashCommLinear(
            input_size=input_size,
            output_size=output_size,
            tp_size=tp_size,
            tp_rank=1,
            bias=True,
            skip_bias_add=False,
            params_dtype=torch.float32,
            quant_config=None,
            prefix="flash_col.basic.",
        )

        self.assertEqual(layer.input_size, input_size)
        self.assertEqual(layer.output_size, output_size)
        self.assertEqual(layer.tp_size, tp_size)
        self.assertEqual(layer.tp_rank, 1)

        self.assertEqual(layer.output_size_per_partition, output_size // tp_size)
        self.assertEqual(layer.output_partition_sizes,
                         [output_size // tp_size])

        self.assertIsNotNone(layer.bias)
        self.assertEqual(layer.bias.shape[0], layer.output_size_per_partition)
        # quant_method 来自 FlashCommLinearBase，默认一定非 None
        self.assertIsNotNone(layer.quant_method)

    def test_column_parallel_flash_comm_linear_init_without_bias_registers_none(self):
        """bias=False 时应注册 bias=None。"""
        from omni.layers.linear import ColumnParallelFlashCommLinear

        layer = ColumnParallelFlashCommLinear(
            input_size=8,
            output_size=4,
            tp_size=2,
            tp_rank=0,
            bias=False,
            skip_bias_add=False,
            params_dtype=torch.float32,
            quant_config=None,
            prefix="flash_col.nobias.",
        )

        self.assertTrue(hasattr(layer, "bias"))
        self.assertIsNone(layer.bias)

    def test_column_parallel_flash_comm_linear_init_with_output_sizes_overrides_partition_sizes(self):
        """
        当实例上预先存在 self.output_sizes 属性时，应使用每个 shard 的
        output_sizes // tp_size 作为 output_partition_sizes。
        """
        from omni.layers.linear import ColumnParallelFlashCommLinear

        class DummyColumnParallelFlashCommLinear(ColumnParallelFlashCommLinear):
            def __init__(self, input_size, output_sizes, tp_size=2, tp_rank=0,
                         **kwargs):
                # 模拟 QKV / MergedColumn 情况：提前挂上 output_sizes
                self.output_sizes = output_sizes
                super().__init__(
                    input_size=input_size,
                    output_size=sum(output_sizes),
                    tp_size=tp_size,
                    tp_rank=tp_rank,
                    output_sizes=output_sizes,
                    **kwargs,
                )

        output_sizes = [4, 8]
        tp_size = 2
        layer = DummyColumnParallelFlashCommLinear(
            input_size=16,
            output_sizes=output_sizes,
            tp_size=tp_size,
            tp_rank=0,
            bias=False,
            skip_bias_add=False,
            params_dtype=torch.float32,
            quant_config=None,
            prefix="flash_col.merge.",
        )

        # 总的 per-partition 宽度还是 sum(output_sizes) // tp_size
        self.assertEqual(layer.output_size_per_partition,
                         sum(output_sizes) // tp_size)
        # 但 output_partition_sizes 应按每个 shard 分别除以 tp_size
        self.assertEqual(
            layer.output_partition_sizes,
            [os // tp_size for os in output_sizes],
        )

    def test_column_parallel_flash_comm_linear_weight_loader_with_output_dim_slices_by_tp_rank(self):
        """
        weight_loader: 当 param 有 output_dim 时，应按 tp_rank 在该维上进行窄切片。
        """
        from omni.layers.linear import ColumnParallelFlashCommLinear

        tp_size = 2
        tp_rank = 1  # 取第二个 shard
        input_size = 10
        output_size = 6  # 这里只是给个值，不影响 weight_loader 的逻辑

        layer = ColumnParallelFlashCommLinear(
            input_size=input_size,
            output_size=output_size,
            tp_size=tp_size,
            tp_rank=tp_rank,
            bias=False,
            skip_bias_add=False,
            params_dtype=torch.float32,
            quant_config=None,
            prefix="flash_col.wl_slice.",
        )

        # param_data: [2, 3]，在 dim=1 上做 TP 分片
        param = Parameter(torch.zeros(2, 3, dtype=torch.float32))
        param.output_dim = 1
        param.is_2_dims = True  # 避免 squeeze 操作改变形状

        # loaded_weight 在 dim=1 上长度要 >= 3 * tp_size = 6
        loaded_weight = torch.arange(2 * 6, dtype=torch.float32).reshape(2, 6)

        layer.weight_loader(param, loaded_weight)

        # tp_rank=1 -> 应取 loaded_weight[:, 3:6]
        expected = loaded_weight.narrow(1, 3, 3)
        self.assertTrue(torch.allclose(param.data, expected))

    def test_column_parallel_flash_comm_linear_weight_loader_no_output_dim_squeezes_when_not_is_2_dims(self):
        """
        weight_loader: output_dim=None 且 is_2_dims=False 时，会对 loaded_weight 做 squeeze。
        """
        from omni.layers.linear import ColumnParallelFlashCommLinear

        layer = ColumnParallelFlashCommLinear(
            input_size=4,
            output_size=4,
            tp_size=1,
            tp_rank=0,
            bias=False,
            skip_bias_add=False,
            params_dtype=torch.float32,
            quant_config=None,
            prefix="flash_col.wl_squeeze.",
        )

        # param_data: 1D 向量
        param = Parameter(torch.zeros(3, dtype=torch.float32))
        # 不设置 output_dim -> None
        # 不设置 is_2_dims -> False

        # loaded_weight: [3, 1]，squeeze 后应变成 [3]
        loaded_weight = torch.arange(3, dtype=torch.float32).view(3, 1)

        layer.weight_loader(param, loaded_weight)

        self.assertTrue(
            torch.allclose(param.data, torch.arange(3, dtype=torch.float32))
        )

    def test_column_parallel_flash_comm_linear_weight_loader_with_is_weight_transposed_roundtrip_calls_npu_format_cast(self):
        """
        weight_loader: is_weight_transposed=True 时，应在前后各做一次转置，
        并调用 torch_npu.npu_format_cast。
        """
        from omni.layers.linear import ColumnParallelFlashCommLinear
        import importlib
        from unittest.mock import patch

        layer_mod = importlib.import_module(ColumnParallelFlashCommLinear.__module__)

        layer = ColumnParallelFlashCommLinear(
            input_size=4,
            output_size=8,
            tp_size=1,
            tp_rank=0,
            bias=False,
            skip_bias_add=False,
            params_dtype=torch.float32,
            quant_config=None,
            prefix="flash_col.wl_transpose.",
        )

        # 初始 param_data: [2, 4]
        param = Parameter(torch.zeros(2, 4, dtype=torch.float32))
        param.is_weight_transposed = True
        param.output_dim = 1
        param.is_2_dims = True  # 不触发 squeeze

        # loaded_weight 与转置后的 param_data 大小兼容：[4, 2]
        loaded_weight = torch.ones(4, 2, dtype=torch.float32)

        with patch.object(
            layer_mod.torch_npu,
            "npu_format_cast",
            side_effect=lambda x, fmt: x,
        ) as mock_cast:
            layer.weight_loader(param, loaded_weight)

        # 应被调用一次，且 fmt=29
        mock_cast.assert_called()
        called_tensor, called_fmt = mock_cast.call_args[0]
        self.assertEqual(called_fmt, 29)
        # 传入 npu_format_cast 的张量应为转置后的形状 [2, 4]
        self.assertEqual(tuple(called_tensor.shape), (2, 4))

    def test_column_parallel_flash_comm_linear_forward_calls_quant_method_with_bias_and_flags(self):
        """
        forward: skip_bias_add=False 时，bias 传入 quant_method.apply，
        且 x_transform / is_prefill / module_name 正确透传。
        """
        from omni.layers.linear import ColumnParallelFlashCommLinear

        class CaptureQuantMethod:
            def __init__(self):
                self.last_layer = None
                self.last_input = None
                self.last_bias = None
                self.last_module_name = None
                self.last_x_transform = None
                self.last_is_prefill = None

            def create_weights(
                self,
                layer,
                input_size_per_partition,
                output_partition_sizes,
                input_size,
                output_size,
                params_dtype,
                weight_loader,
            ):
                return

            def apply(self, layer, x, bias=None, module_name=None,
                      x_transform=None, is_prefill=True):
                self.last_layer = layer
                self.last_input = x.clone()
                self.last_bias = bias
                self.last_module_name = module_name
                self.last_x_transform = x_transform
                self.last_is_prefill = is_prefill
                return torch.zeros(x.shape[0], layer.output_size,
                                   dtype=x.dtype)

        input_size = 8
        output_size = 4
        batch = 3

        layer = ColumnParallelFlashCommLinear(
            input_size=input_size,
            output_size=output_size,
            tp_size=1,
            tp_rank=0,
            bias=True,
            skip_bias_add=False,
            params_dtype=torch.float32,
            quant_config=None,
            prefix="flash_col.fwd.",
        )
        qm = CaptureQuantMethod()
        layer.quant_method = qm

        x = torch.randn(batch, input_size, dtype=torch.float32)
        out, out_bias = layer(x, x_transform="AG", is_prefill=False)

        self.assertTrue(torch.allclose(qm.last_input, x))
        self.assertIs(qm.last_layer, layer)
        self.assertIs(qm.last_bias, layer.bias)
        self.assertEqual(qm.last_module_name, layer.prefix)
        self.assertEqual(qm.last_x_transform, "AG")
        self.assertFalse(qm.last_is_prefill)

        self.assertEqual(out.shape, (batch, output_size))
        # skip_bias_add=False -> bias 已融合，第二个输出为 None
        self.assertIsNone(out_bias)

    def test_column_parallel_flash_comm_linear_forward_skip_bias_add_true_returns_output_and_output_bias(self):
        """
        forward: skip_bias_add=True 时，不向 quant_method 传 bias，
        而是把 bias 作为第二个返回值 output_bias。
        """
        from omni.layers.linear import ColumnParallelFlashCommLinear

        class CaptureQuantMethod:
            def __init__(self):
                self.last_bias = None
                self.last_is_prefill = None

            def create_weights(
                self,
                layer,
                input_size_per_partition,
                output_partition_sizes,
                input_size,
                output_size,
                params_dtype,
                weight_loader,
            ):
                return

            def apply(self, layer, x, bias=None, module_name=None,
                      x_transform=None, is_prefill=True):
                self.last_bias = bias
                self.last_is_prefill = is_prefill
                return torch.zeros(x.shape[0], layer.output_size,
                                   dtype=x.dtype)

        layer = ColumnParallelFlashCommLinear(
            input_size=8,
            output_size=4,
            tp_size=1,
            tp_rank=0,
            bias=True,
            skip_bias_add=True,
            params_dtype=torch.float32,
            quant_config=None,
            prefix="flash_col.fwd_skip.",
        )
        qm = CaptureQuantMethod()
        layer.quant_method = qm

        x = torch.randn(5, 8, dtype=torch.float32)
        out, out_bias = layer(x, x_transform=None, is_prefill=True)

        self.assertIsNone(qm.last_bias)        # 不融合 bias
        self.assertTrue(qm.last_is_prefill)    # 默认 True
        self.assertEqual(out.shape, (5, 4))
        self.assertIs(out_bias, layer.bias)    # bias 作为第二输出返回

    def test_column_parallel_flash_comm_linear_forward_quant_method_none_raises_assertion(self):
        """
        forward: 如果 quant_method 被意外置为 None，应触发断言。
        """
        from omni.layers.linear import ColumnParallelFlashCommLinear

        layer = ColumnParallelFlashCommLinear(
            input_size=8,
            output_size=4,
            tp_size=1,
            tp_rank=0,
            bias=False,
            skip_bias_add=False,
            params_dtype=torch.float32,
            quant_config=None,
            prefix="flash_col.no_qm.",
        )

        layer.quant_method = None  # 手动破坏，触发 forward 内部 assert

        x = torch.randn(2, 8, dtype=torch.float32)
        with self.assertRaises(AssertionError):
            _ = layer(x)

# ================= QKVParallelFlashCommLinear =================

    def test_qkv_parallel_flash_comm_linear_init_basic_no_kv_sharing(self):
        """
        tp_size < total_num_kv_heads 场景：
        - num_heads, num_kv_heads, num_kv_head_replicas 推导正确
        - output_size 与 output_sizes 一致
        """
        from omni.layers.linear import QKVParallelFlashCommLinear

        hidden_size = 16
        head_size = 4
        total_num_heads = 8
        total_num_kv_heads = 8
        tp_size = 2
        tp_rank = 0

        layer = QKVParallelFlashCommLinear(
            hidden_size=hidden_size,
            head_size=head_size,
            total_num_heads=total_num_heads,
            total_num_kv_heads=total_num_kv_heads,
            tp_size=tp_size,
            tp_rank=tp_rank,
            bias=True,
            skip_bias_add=False,
            params_dtype=torch.float32,
            quant_config=None,
            prefix="qkv.basic.",
        )

        # __init__ 中直接保存的字段
        self.assertEqual(layer.hidden_size, hidden_size)
        self.assertEqual(layer.head_size, head_size)
        self.assertEqual(layer.total_num_heads, total_num_heads)
        self.assertEqual(layer.total_num_kv_heads, total_num_kv_heads)
        self.assertEqual(layer.tp_size, tp_size)
        self.assertEqual(layer.tp_rank, tp_rank)

        # 头数与 KV 头数推导
        # tp_size < total_num_kv_heads -> num_kv_heads = total_num_kv_heads / tp_size, num_kv_head_replicas = 1
        self.assertEqual(layer.num_heads, total_num_heads // tp_size)
        self.assertEqual(layer.num_kv_heads, total_num_kv_heads // tp_size)
        self.assertEqual(layer.num_kv_head_replicas, 1)

        # output_size 与 output_sizes 一致
        expected_output_sizes = [
            layer.num_heads * head_size * tp_size,
            layer.num_kv_heads * head_size * tp_size,
            layer.num_kv_heads * head_size * tp_size,
        ]
        self.assertEqual(layer.output_sizes, expected_output_sizes)
        self.assertEqual(layer.output_size, sum(expected_output_sizes))

        # ColumnParallelFlashCommLinear 中的 per-partition 维度
        self.assertEqual(
            layer.output_size_per_partition,
            layer.output_size // tp_size,
        )
        # output_partition_sizes 应当是拆分后的 [q, k, v] 每个分片的宽度
        self.assertEqual(
            layer.output_partition_sizes,
            [s // tp_size for s in expected_output_sizes],
        )

    def test_qkv_parallel_flash_comm_linear_init_with_kv_head_replicas_when_tp_ge_total_kv_heads(self):
        """
        tp_size >= total_num_kv_heads 场景：
        - num_kv_heads == 1
        - num_kv_head_replicas == tp_size / total_num_kv_heads
        """
        from omni.layers.linear import QKVParallelFlashCommLinear

        hidden_size = 16
        head_size = 2
        total_num_heads = 4
        total_num_kv_heads = 2
        tp_size = 4
        tp_rank = 3

        layer = QKVParallelFlashCommLinear(
            hidden_size=hidden_size,
            head_size=head_size,
            total_num_heads=total_num_heads,
            total_num_kv_heads=total_num_kv_heads,
            tp_size=tp_size,
            tp_rank=tp_rank,
            bias=False,
            skip_bias_add=False,
            params_dtype=torch.float32,
            quant_config=None,
            prefix="qkv.repl.",
        )

        # 头数
        self.assertEqual(layer.num_heads, total_num_heads // tp_size)
        # tp_size >= total_num_kv_heads -> num_kv_heads=1, num_kv_head_replicas=tp_size/total_num_kv_heads
        self.assertEqual(layer.num_kv_heads, 1)
        self.assertEqual(layer.num_kv_head_replicas, tp_size // total_num_kv_heads)

        # output_sizes 三路 q/k/v 的 fused 宽度
        expected_output_sizes = [
            layer.num_heads * head_size * tp_size,
            layer.num_kv_heads * head_size * tp_size,
            layer.num_kv_heads * head_size * tp_size,
        ]
        self.assertEqual(layer.output_sizes, expected_output_sizes)

    def test_qkv_parallel_flash_comm_linear_weight_loader_fused_dispatches_q_k_v(self):
        """
        loaded_shard_id=None 时，按 Q/K/V 三段切分 fused 权重，并递归调用自身
        weight_loader(..., 'q'/'k'/'v')。
        """
        from omni.layers.linear import QKVParallelFlashCommLinear
        from torch.nn import Parameter
        from unittest import mock

        hidden_size = 8
        head_size = 1
        total_num_heads = 4      # q 维度 = 4
        total_num_kv_heads = 2   # 每个 k/v 维度 = 2
        tp_size = 2
        tp_rank = 0

        layer = QKVParallelFlashCommLinear(
            hidden_size=hidden_size,
            head_size=head_size,
            total_num_heads=total_num_heads,
            total_num_kv_heads=total_num_kv_heads,
            tp_size=tp_size,
            tp_rank=tp_rank,
            bias=False,
            skip_bias_add=False,
            params_dtype=torch.float32,
            quant_config=None,
            prefix="qkv.fused.",
        )

        # 人工构造一个 param，并设定 output_dim=0（按 dim 0 切分）
        global_dim = (total_num_heads + 2 * total_num_kv_heads) * head_size  # 4 + 2*2 = 8
        param = Parameter(torch.zeros(global_dim, dtype=torch.float32))
        param.output_dim = 0

        loaded_weight = torch.arange(global_dim, dtype=torch.float32)

        calls = []
        orig_loader = layer.weight_loader

        def spy_loader(p, lw, loaded_shard_id=None):
            # 顶层 fused 调用走原始逻辑，内部递归会再次调用 spy_loader
            if loaded_shard_id is None:
                return orig_loader(p, lw, loaded_shard_id)
            # 记录 q/k/v 分支收到的权重切片
            calls.append((loaded_shard_id, lw.clone()))
            # 不再继续深入，避免真实写入
            return None

        with mock.patch.object(layer, "weight_loader", side_effect=spy_loader):
            layer.weight_loader(param, loaded_weight, loaded_shard_id=None)

        # 应该按顺序调用 q, k, v 三个分支
        self.assertEqual([c[0] for c in calls], ["q", "k", "v"])

        q_slice = calls[0][1]
        k_slice = calls[1][1]
        v_slice = calls[2][1]

        # 维度应该分别为 4, 2, 2
        self.assertEqual(q_slice.shape[0], total_num_heads * head_size)
        self.assertEqual(k_slice.shape[0], total_num_kv_heads * head_size)
        self.assertEqual(v_slice.shape[0], total_num_kv_heads * head_size)

        # 内容对应原始 loaded_weight 的三段
        self.assertTrue(torch.equal(q_slice, loaded_weight[0:4]))
        self.assertTrue(torch.equal(k_slice, loaded_weight[4:6]))
        self.assertTrue(torch.equal(v_slice, loaded_weight[6:8]))

    def test_qkv_parallel_flash_comm_linear_weight_loader_q_branch_slices_by_tp_rank(self):
        """
        loaded_shard_id='q' 时：
        - 使用 shard_offset=0, shard_size=num_heads*head_size
        - start_idx = tp_rank * shard_size
        - param.data 对应位置写入 [start_idx:start_idx+shard_size] 的切片。
        """
        from omni.layers.linear import QKVParallelFlashCommLinear
        from torch.nn import Parameter

        hidden_size = 16
        head_size = 1
        total_num_heads = 4    # 全局 q 维度：4
        total_num_kv_heads = 4
        tp_size = 2
        tp_rank = 1            # 用第二个 TP 分片

        layer = QKVParallelFlashCommLinear(
            hidden_size=hidden_size,
            head_size=head_size,
            total_num_heads=total_num_heads,
            total_num_kv_heads=total_num_kv_heads,
            tp_size=tp_size,
            tp_rank=tp_rank,
            bias=False,
            skip_bias_add=False,
            params_dtype=torch.float32,
            quant_config=None,
            prefix="qkv.qbranch.",
        )

        # 本地 fused 维度：(num_heads + 2*num_kv_heads) * head_size
        local_dim = (layer.num_heads + 2 * layer.num_kv_heads) * head_size
        param = Parameter(torch.zeros(local_dim, dtype=torch.float32))
        param.output_dim = 0  # 按 dim 0 切分

        global_q_dim = total_num_heads * head_size  # 4
        loaded_weight = torch.arange(global_q_dim, dtype=torch.float32)

        # 调用 q 分支
        layer.weight_loader(param, loaded_weight, loaded_shard_id="q")

        # shard_size = num_heads * head_size
        shard_size = layer.num_heads * head_size
        # tp_rank=1 -> start_idx=1*shard_size
        start_idx = tp_rank * shard_size

        expected_slice = loaded_weight[start_idx:start_idx + shard_size]
        # q 段写入 param 的前 shard_size 个元素
        self.assertTrue(torch.allclose(param.data[:shard_size], expected_slice))
        # 其余位置保持为 0
        if local_dim > shard_size:
            self.assertTrue(torch.all(param.data[shard_size:] == 0))

    def test_qkv_parallel_flash_comm_linear_weight_loader_k_branch_respects_kv_head_replicas(self):
        """
        tp_size >= total_num_kv_heads 场景下，K/V 分支：
        - shard_id = tp_rank // num_kv_head_replicas
        验证 KV 头复用逻辑正确。
        """
        from omni.layers.linear import QKVParallelFlashCommLinear
        from torch.nn import Parameter
        import torch

        hidden_size = 16
        head_size = 1
        total_num_heads = 4
        total_num_kv_heads = 2   # tp_size >= total_num_kv_heads
        tp_size = 4
        tp_rank = 3              # 最高 rank，用来测试 // num_kv_head_replicas

        layer = QKVParallelFlashCommLinear(
            hidden_size=hidden_size,
            head_size=head_size,
            total_num_heads=total_num_heads,
            total_num_kv_heads=total_num_kv_heads,
            tp_size=tp_size,
            tp_rank=tp_rank,
            bias=False,
            skip_bias_add=False,
            params_dtype=torch.float32,
            quant_config=None,
            prefix="qkv.kvrep.",
        )

        # 校验一下 num_kv_heads / num_kv_head_replicas 计算
        self.assertEqual(layer.num_heads, total_num_heads // tp_size)          # 4/4=1
        self.assertEqual(layer.num_kv_heads, 1)
        self.assertEqual(layer.num_kv_head_replicas, tp_size // total_num_kv_heads)  # 4/2=2

        # 本地 fused 维度：(num_heads + 2*num_kv_heads) * head_size
        local_dim = (layer.num_heads + 2 * layer.num_kv_heads) * head_size     # (1+2*1)*1 = 3
        param = Parameter(torch.zeros(local_dim, dtype=torch.float32))
        param.output_dim = 0
        # 模拟真实权重场景：不对 loaded_weight 做 squeeze，保持维度匹配
        param.is_2_dims = True

        # 假设全局 K 权重维度为 total_num_kv_heads * head_size = 2，对应值 [0, 1]
        global_k_dim = total_num_kv_heads * head_size
        loaded_weight = torch.arange(global_k_dim, dtype=torch.float32)

        # K 分支中：
        # shard_offset = num_heads * head_size = 1
        # shard_size   = num_kv_heads * head_size = 1
        # shard_id     = tp_rank // num_kv_head_replicas = 3 // 2 = 1
        # start_idx    = shard_id * shard_size = 1
        layer.weight_loader(param, loaded_weight, loaded_shard_id="k")

        # weight_loader 内部对 param_data 做了 narrow(0, 1, 1)，copy_ 写回到 param.data[1]
        self.assertEqual(param.data.shape, torch.Size([local_dim]))
        # 预期：K 分支拿到的是 loaded_weight[1] == 1.0
        self.assertAlmostEqual(param.data[1].item(), 1.0)
        # 其它位置保持为 0
        self.assertAlmostEqual(param.data[0].item(), 0.0)
        self.assertAlmostEqual(param.data[2].item(), 0.0)

# ================= MergedColumnParallelFlashCommLinear =================

    def test_merged_column_parallel_flash_comm_linear_init_basic_partitions_and_bias(self):
        """基本构造：校验 output_sizes / 分片尺寸 / bias 属性。"""
        from omni.layers.linear import MergedColumnParallelFlashCommLinear
        from torch.nn import Parameter
        import torch

        input_size = 8
        output_sizes = [4, 8]
        tp_size = 2
        tp_rank = 1

        layer = MergedColumnParallelFlashCommLinear(
            input_size=input_size,
            output_sizes=output_sizes,
            tp_size=tp_size,
            tp_rank=tp_rank,
            bias=True,
            skip_bias_add=False,
            params_dtype=torch.float32,
            quant_config=None,
            prefix="mcol_flash.init.",
        )

        # 基本属性
        self.assertEqual(layer.input_size, input_size)
        self.assertEqual(layer.output_size, sum(output_sizes))
        self.assertEqual(layer.output_sizes, output_sizes)
        self.assertEqual(layer.tp_size, tp_size)
        self.assertEqual(layer.tp_rank, tp_rank)

        # 分片大小
        self.assertEqual(layer.output_size_per_partition,
                         layer.output_size // tp_size)
        expected_partitions = [os // tp_size for os in output_sizes]
        self.assertEqual(layer.output_partition_sizes, expected_partitions)

        # bias 形状与属性
        self.assertIsInstance(layer.bias, Parameter)
        self.assertEqual(layer.bias.shape[0], layer.output_size_per_partition)
        self.assertEqual(getattr(layer.bias, "output_dim", None), 0)

        bias_loader = getattr(layer.bias, "weight_loader", None)
        self.assertIsNotNone(bias_loader)
        # 比较绑定方法的 __func__，避免 bound-method 对象 identity 不同
        self.assertEqual(
            getattr(bias_loader, "__func__", bias_loader),
            getattr(layer.weight_loader, "__func__", layer.weight_loader),
        )

    def test_merged_column_parallel_flash_comm_linear_init_invalid_output_sizes_assert(self):
        """
        output_sizes 中存在不能被 tp_size 整除的情况时，应触发断言。
        """
        from omni.layers.linear import MergedColumnParallelFlashCommLinear
        import torch

        with self.assertRaises(AssertionError):
            MergedColumnParallelFlashCommLinear(
                input_size=8,
                output_sizes=[5, 4],  # 5 % 2 != 0
                tp_size=2,
                tp_rank=0,
                bias=False,
                skip_bias_add=False,
                params_dtype=torch.float32,
                quant_config=None,
                prefix="mcol_flash.bad.",
            )

    def test_merged_column_parallel_flash_comm_linear_weight_loader_basic_slice_per_shard_and_tp_rank(self):
        """
        weight_loader 基本切片逻辑：
        - shard_offset = sum(output_sizes[:loaded_shard_id]) // tp_size
        - shard_size   = output_sizes[loaded_shard_id] // tp_size
        - start_idx    = tp_rank * shard_size
        验证对应切片写入 param.data 指定区间。
        """
        from omni.layers.linear import MergedColumnParallelFlashCommLinear
        from torch.nn import Parameter
        import torch

        output_sizes = [4, 8]
        tp_size = 2
        tp_rank = 1  # 取第二个 TP 分片

        layer = MergedColumnParallelFlashCommLinear(
            input_size=10,
            output_sizes=output_sizes,
            tp_size=tp_size,
            tp_rank=tp_rank,
            bias=False,
            skip_bias_add=False,
            params_dtype=torch.float32,
            quant_config=None,
            prefix="mcol_flash.slice.",
        )

        # param_data 形状：总长 6，其中 [2:6] 位置会被覆盖
        param = Parameter(torch.zeros(6, dtype=torch.float32))
        param.output_dim = 0  # 在 dim 0 上做切片

        # loaded_weight 的 dim0 至少要覆盖 start_idx+shard_size
        # shard_offset = sum([4]) // 2 = 2（只影响 param 的窄切范围）
        # shard_size   = 8 // 2 = 4
        # start_idx    = tp_rank * shard_size = 1 * 4 = 4
        loaded_weight = torch.arange(0, 8, dtype=torch.float32)

        layer.weight_loader(param, loaded_weight, loaded_shard_id=1)

        # 期望 param.data[2:6] == loaded_weight[4:8]
        expected = loaded_weight.narrow(0, 4, 4)
        self.assertTrue(torch.allclose(param.data[2:], expected))
        self.assertTrue(torch.all(param.data[:2] == 0))

    def test_merged_column_parallel_flash_comm_linear_weight_loader_with_is_2_dims_false_squeezes(self):
        """
        当 is_2_dims 为 False（默认）时，会对 loaded_weight 做 squeeze，
        支持 loaded_weight 带有多余的维度。
        """
        from omni.layers.linear import MergedColumnParallelFlashCommLinear
        from torch.nn import Parameter
        import torch

        output_sizes = [4]
        tp_size = 2
        tp_rank = 0

        layer = MergedColumnParallelFlashCommLinear(
            input_size=10,
            output_sizes=output_sizes,
            tp_size=tp_size,
            tp_rank=tp_rank,
            bias=False,
            skip_bias_add=False,
            params_dtype=torch.float32,
            quant_config=None,
            prefix="mcol_flash.squeeze.",
        )

        # shard_offset = 0
        # shard_size   = 4 // 2 = 2
        param = Parameter(torch.zeros(2, dtype=torch.float32))
        param.output_dim = 0

        # 让 loaded_weight 有额外的一维，便于验证 squeeze 行为
        loaded_weight = torch.arange(0, 2, dtype=torch.float32).view(2, 1)

        layer.weight_loader(param, loaded_weight, loaded_shard_id=0)

        expected = torch.arange(0, 2, dtype=torch.float32)
        self.assertTrue(torch.allclose(param.data, expected))

    def test_merged_column_parallel_flash_comm_linear_weight_loader_with_weight_transposed_roundtrip(self):
        """
        is_weight_transposed=True 分支：
        - 前置对 param.data 做一次转置
        - 写入后再次转置并经过 npu_format_cast
        验证调用 npu_format_cast 且最终数据形状/内容合理。
        """
        from omni.layers.linear import MergedColumnParallelFlashCommLinear
        from torch.nn import Parameter
        from unittest.mock import patch
        import torch

        output_sizes = [4]
        tp_size = 1
        tp_rank = 0

        layer = MergedColumnParallelFlashCommLinear(
            input_size=10,
            output_sizes=output_sizes,
            tp_size=tp_size,
            tp_rank=tp_rank,
            bias=False,
            skip_bias_add=False,
            params_dtype=torch.float32,
            quant_config=None,
            prefix="mcol_flash.transposed.",
        )

        # 初始 param.data 形状为 (3, 4)，先在 weight_loader 里被转置成 (4, 3)
        param = Parameter(torch.zeros(3, 4, dtype=torch.float32))
        param.output_dim = 0
        param.is_weight_transposed = True

        # 与转置后的 param.data 形状匹配的 loaded_weight
        loaded_weight = torch.arange(12, dtype=torch.float32).view(4, 3)

        # patch 掉 npu_format_cast，简化为直接返回输入
        with patch("omni.layers.linear.torch_npu.npu_format_cast",
                   side_effect=lambda x, fmt: x) as mock_cast:
            layer.weight_loader(param, loaded_weight, loaded_shard_id=0)

        mock_cast.assert_called_once()

        # 最终 param.data 应被第二次转置回 (3, 4)
        self.assertEqual(tuple(param.data.shape), (3, 4))

        # 内容应等于 loaded_weight 的转置
        expected = loaded_weight.t().contiguous()
        self.assertTrue(torch.allclose(param.data, expected))

if __name__ == "__main__":
    unittest.main()
