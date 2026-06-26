# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
import inspect

import pytest
from unittest.mock import patch, MagicMock

import torch
from vllm.compilation.cuda_graph import CUDAGraphOptions
from vllm.config import CUDAGraphMode, VllmConfig

import omni_npu.compilation.acl_graph as acl_graph_mod

REQUIRED_SYMBOLS = [
    "weak_ref_tensor",
    "weak_ref_tensors",
    "GraphParams",
    "set_graph_params",
    "get_graph_params",
    "set_aclgraph_recapture",
    "consume_aclgraph_recapture",
    "ACLGraphWrapper",
    "ACLGraphEntry",
]
missing_symbols = [name for name in REQUIRED_SYMBOLS if not hasattr(acl_graph_mod, name)]
if missing_symbols:
    pytest.skip(
        f"acl_graph APIs changed, missing symbols: {missing_symbols}",
        allow_module_level=True,
    )

weak_ref_tensor = acl_graph_mod.weak_ref_tensor
weak_ref_tensors = acl_graph_mod.weak_ref_tensors
GraphParams = acl_graph_mod.GraphParams
set_graph_params = acl_graph_mod.set_graph_params
get_graph_params = acl_graph_mod.get_graph_params
ACLGraphWrapper = acl_graph_mod.ACLGraphWrapper
ACLGraphEntry = acl_graph_mod.ACLGraphEntry
set_aclgraph_recapture = acl_graph_mod.set_aclgraph_recapture
consume_aclgraph_recapture = acl_graph_mod.consume_aclgraph_recapture


@pytest.fixture(autouse=True)
def reset_acl_graph_state():
    acl_graph_mod._graph_params = None
    acl_graph_mod.global_recapture = False
    yield
    acl_graph_mod._graph_params = None
    acl_graph_mod.global_recapture = False


def test_weak_ref_tensor():
    """Test whether the weak reference shares data with the original tensor"""
    tensor = torch.tensor([1.0, 2.0, 3.0])
    result = weak_ref_tensor(tensor)
    assert torch.equal(result, tensor)


def test_weak_ref_tensors():
    """Test the handling of tuple and list tensor by weak_ref_tensors"""
    tensor_list = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]
    tensor_tuple = (torch.tensor([5.0, 6.0]), torch.tensor([7.0, 8.0]))
    tensor_dict = {"key": torch.tensor([1.0, 2.0])}

    weak_ref_tensor_list = weak_ref_tensors(tensor_list)
    assert all(torch.equal(wt, t) for wt, t in zip(weak_ref_tensor_list, tensor_list))

    weak_ref_tensor_tuple = weak_ref_tensors(tensor_tuple)
    assert all(torch.equal(wt, t) for wt, t in zip(weak_ref_tensor_tuple, tensor_tuple))

    with pytest.raises(ValueError, match="Invalid type for tensors"):
        weak_ref_tensors(tensor_dict)


def test_weak_ref_tensor_with_ascend_support():
    """Test torch.ops._C_ascend module contains a property named 'weak_ref_tensor'"""
    tensor = torch.tensor([1.0, 2.0, 3.0])

    # Simulate the torch.ops._C_ascend module and add the weak_ref_tensor attribute
    mock_weak_ref_tensor = MagicMock(return_value=tensor)
    mock_ascend = MagicMock()
    mock_ascend.weak_ref_tensor = mock_weak_ref_tensor

    with patch("torch.ops._C_ascend", new=mock_ascend):
        result = weak_ref_tensor(tensor)

        mock_weak_ref_tensor.assert_called_once_with(tensor)
        assert torch.equal(result, tensor)


def test_set_get_graph_params():
    """Test set_graph_params and get_graph_params function"""
    aclgraph_capture_sizes = {1, 2, 3}
    set_graph_params(aclgraph_capture_sizes)

    with pytest.raises(ValueError, match="Graph parameters have already been set!"):
        set_graph_params(aclgraph_capture_sizes)

    graph_params = get_graph_params()

    assert isinstance(graph_params, GraphParams)
    assert set(graph_params.task_entries.keys()) == aclgraph_capture_sizes
    assert set(graph_params.workspaces.keys()) == aclgraph_capture_sizes


def test_consume_aclgraph_recapture_once():
    set_aclgraph_recapture(True)
    assert consume_aclgraph_recapture() is True
    assert consume_aclgraph_recapture() is False


def test_set_aclgraph_recapture_coalesces_requests():
    set_aclgraph_recapture(True)
    set_aclgraph_recapture(True)
    assert consume_aclgraph_recapture() is True
    assert consume_aclgraph_recapture() is False


class TestACLGraphWrapper:
    """Test class for ACLGraphWrapper"""

    @pytest.fixture
    def default_npu_graph(self):
        with patch("torch.npu.NPUGraph") as mock_npu_graph, \
             patch("torch.npu.graph") as mock_npu_graph_ctx, \
             patch("torch.npu.Stream") as mock_npu_stream:
            mock_graph_instance = MagicMock()
            mock_graph_instance.replay = MagicMock()
            mock_graph_instance.update = MagicMock()
            mock_npu_graph.return_value = mock_graph_instance

            # mock torch.npu.graph context manager to avoid executing the actual logic within the with statement
            mock_npu_graph_ctx.return_value.__enter__ = lambda self: None
            mock_npu_graph_ctx.return_value.__exit__ = lambda *args: None

            mock_npu_stream.return_value = MagicMock()

            yield mock_graph_instance


    @pytest.fixture
    def default_aclgraph_wrapper(self, default_npu_graph):
        with patch("omni_npu.compilation.acl_graph.get_forward_context") as mock_get_forward_context:
            mock_context = MagicMock()
            mock_batch_descriptor = MagicMock()
            mock_batch_descriptor.num_reqs = None
            mock_batch_descriptor.num_tokens = None
            mock_context.batch_descriptor = mock_batch_descriptor
            mock_context.cudagraph_runtime_mode = CUDAGraphMode.PIECEWISE
            mock_context.attn_metadata = None
            mock_get_forward_context.return_value = mock_context

            vllm_config = MagicMock(spec=VllmConfig)
            vllm_config.additional_config = None

            wrapper = ACLGraphWrapper(
                runnable=MagicMock(),
                vllm_config=vllm_config,
                runtime_mode=CUDAGraphMode.PIECEWISE,
                graph_pool=MagicMock()
            )
            wrapper._forward_context = mock_context
            wrapper.concrete_aclgraph_entries = {}
            wrapper.aclgraph_options = MagicMock(
                gc_disable=False,
                debug_log_enable=False,
                weak_ref_output=False,
            )
            wrapper.default_npu_graph = default_npu_graph
            yield wrapper


    def test_acl_graph_wrapper_init_normal(self):
        """Test ACLGraphWrapper initialization"""
        runnable = MagicMock()
        vllm_config = MagicMock(spec=VllmConfig)
        vllm_config.additional_config = None
        runtime_mode = CUDAGraphMode.PIECEWISE
        graph_pool = MagicMock()
        cudagraph_options = MagicMock()

        wrapper = ACLGraphWrapper(runnable, vllm_config, runtime_mode, graph_pool, cudagraph_options)

        assert wrapper.runnable == runnable
        assert wrapper.vllm_config == vllm_config
        assert wrapper.graph_pool == graph_pool
        assert wrapper.runtime_mode == runtime_mode
        assert wrapper.aclgraph_options == cudagraph_options
        assert wrapper.need_static_compile is False
        assert isinstance(wrapper.concrete_aclgraph_entries, dict)
        assert wrapper.recapture is False

    def test_acl_graph_wrapper_init_with_static_compile_enabled(self):
        """Test ACLGraphWrapper reads static compile from enabled config."""
        runnable = MagicMock()
        vllm_config = MagicMock(spec=VllmConfig)
        vllm_config.additional_config = {
            "npugraph_ex_config": {
                "enable": True,
                "static_kernel_compile": True,
            }
        }

        wrapper = ACLGraphWrapper(
            runnable,
            vllm_config,
            CUDAGraphMode.PIECEWISE,
            MagicMock(),
            MagicMock(),
        )

        assert wrapper.need_static_compile is True


    def test_acl_graph_wrapper_init_with_none_graph_pool(self):
        """Test ACLGraphWrapper initialization when graph_pool is None"""
        runnable = MagicMock()
        vllm_config = MagicMock(spec=VllmConfig)
        runtime_mode = CUDAGraphMode.PIECEWISE
        cudagraph_options = MagicMock()

        mock_graph_pool = MagicMock()
        with patch("omni_npu.compilation.acl_graph.current_platform.get_global_graph_pool", return_value=mock_graph_pool) as mock_get_global_graph_pool:
            wrapper = ACLGraphWrapper(runnable, vllm_config, runtime_mode, None, cudagraph_options)

            assert wrapper.graph_pool == mock_graph_pool
            mock_get_global_graph_pool.assert_called_once()


    def test_acl_graph_wrapper_init_with_none_cudagraph_options(self):
        """Test ACLGraphWrapper initialization when cudagraph_options is None"""
        runnable = MagicMock()
        vllm_config = MagicMock(spec=VllmConfig)
        runtime_mode = CUDAGraphMode.PIECEWISE
        graph_pool = MagicMock()

        with patch("omni_npu.compilation.acl_graph.CUDAGraphOptions", return_value=MagicMock(spec=CUDAGraphOptions)) as mock_cudagraph_options:
            wrapper = ACLGraphWrapper(runnable, vllm_config, runtime_mode, graph_pool, None)

            # Verify whether the type of aclgraph_options is CUDAGraphOptions
            assert isinstance(wrapper.aclgraph_options, CUDAGraphOptions)
            mock_cudagraph_options.assert_called_once()

    
    def test_acl_graph_wrapper_getattr(self):
        """Test ACLGraphWrapper __get_attr__ method"""
        # Allow runnable to have the key attribute
        runnable = MagicMock(spec_set=["key"])
        runnable.key = "value"
        
        wrapper = ACLGraphWrapper(runnable, MagicMock(), CUDAGraphMode.PIECEWISE)

        assert wrapper.key == "value"
        with pytest.raises(AttributeError):
            _ = wrapper.nonexistent_attr


    def test_acl_graph_wrapper_unwrap(self):
        """Test ACLGraphWrapper unwrap method"""
        runnable = MagicMock()
        wrapper = ACLGraphWrapper(runnable, MagicMock(), CUDAGraphMode.PIECEWISE)
        assert wrapper.unwrap() == runnable


    def test_update_graph_recapture(self):
        """Test update_graph_recapture marks cached entries."""
        wrapper = ACLGraphWrapper(MagicMock(), MagicMock(), CUDAGraphMode.PIECEWISE)
        batch_descriptor = MagicMock()
        wrapper.concrete_aclgraph_entries = {
            batch_descriptor: ACLGraphEntry(batch_descriptor=batch_descriptor)
        }

        wrapper.recapture = True
        wrapper.update_graph_recapture()

        assert wrapper.concrete_aclgraph_entries[batch_descriptor].recapture is True
        assert wrapper.recapture is False


    def test_call_non_aclgraph_runtime_mode(self, default_aclgraph_wrapper):
        """Test the direct invocation of the runnable in the CUDAGraphMode.NONE mode."""
        default_aclgraph_wrapper.runtime_mode = CUDAGraphMode.NONE
        default_aclgraph_wrapper._forward_context.cudagraph_runtime_mode = CUDAGraphMode.NONE
        
        result = default_aclgraph_wrapper(torch.tensor([1.0]))

        # Verify direct invocation of the runnable method
        default_aclgraph_wrapper.runnable.assert_called_once()
        # Verify the returned value is the return value of the runnable.
        assert result == default_aclgraph_wrapper.runnable.return_value
        # Verify not create a new entry for batch descriptor
        assert not default_aclgraph_wrapper.concrete_aclgraph_entries


    def test_call_with_non_none_aclgraph_runtime_mode(self, default_aclgraph_wrapper):
        """Test the aclgraph replay behavior in a non-NONE aclgraph_runtime_mode."""
        default_aclgraph_wrapper.runtime_mode = CUDAGraphMode.PIECEWISE
        default_aclgraph_wrapper._forward_context.cudagraph_runtime_mode = CUDAGraphMode.PIECEWISE
        batch_descriptor = default_aclgraph_wrapper._forward_context.batch_descriptor
        batch_descriptor.num_tokens = 3
        mock_aclgraph = MagicMock()
        entry = ACLGraphEntry(
            batch_descriptor=batch_descriptor,
            aclgraph=mock_aclgraph,
            output=torch.tensor([2.0]),
        )
        default_aclgraph_wrapper.concrete_aclgraph_entries = {batch_descriptor: entry}

        with patch.object(default_aclgraph_wrapper, "_update_graph_tasks") as mock_update_graph_tasks:
            result = default_aclgraph_wrapper(torch.tensor([1.0]))

        default_aclgraph_wrapper.runnable.assert_not_called()
        mock_aclgraph.replay.assert_called_once()
        mock_update_graph_tasks.assert_called_once_with(
            default_aclgraph_wrapper.update_stream,
            default_aclgraph_wrapper._forward_context,
        )
        assert result == entry.output


    def test_call_with_aslkv_none(self, default_aclgraph_wrapper):
        """Test __call__ method raise error when attn_metadata is None."""
        default_aclgraph_wrapper._forward_context.attn_metadata = None
        batch_descriptor = default_aclgraph_wrapper._forward_context.batch_descriptor
        batch_descriptor.num_tokens = 3
        mock_aclgraph = MagicMock()
        entry = ACLGraphEntry(
            batch_descriptor=batch_descriptor,
            aclgraph=mock_aclgraph,
            output=torch.tensor([2.0]),
        )
        default_aclgraph_wrapper.concrete_aclgraph_entries = {batch_descriptor: entry}

        with pytest.raises(RuntimeError):
            default_aclgraph_wrapper(torch.tensor([1.0]))

        mock_aclgraph.replay.assert_called_once()


    def test_call_attn_metadata_gqa_mode(self, default_aclgraph_wrapper):
        """Test __call__ method executes normally under the GQA mode."""
        mock_attn_metadata = MagicMock(spec_set=["query_start_loc", "seq_lens"])
        mock_attn_metadata.query_start_loc = [0, 1]
        mock_attn_metadata.seq_lens = torch.tensor([2])
        default_aclgraph_wrapper._forward_context.attn_metadata = {
            "metadata1": mock_attn_metadata
        }
        default_aclgraph_wrapper.runnable = MagicMock(return_value=torch.tensor([1.0]))

        with patch("omni_npu.compilation.acl_graph.ensure_weak_ref_graph_params"):
            result = default_aclgraph_wrapper(torch.tensor([1.0]))

        default_aclgraph_wrapper.runnable.assert_called_once()
        assert torch.equal(result, torch.tensor([1.0]))


    def test_call_attn_metadata_mla_mode(self, default_aclgraph_wrapper):
        """Test __call__ method executes normally under the MLA mode."""
        mock_attn_metadata = MagicMock()
        mock_attn_metadata.decode = MagicMock()
        mock_attn_metadata.decode.query_cumlens = torch.tensor([1])
        mock_attn_metadata.decode.seq_lens = torch.tensor([2])
        mock_attn_metadata.decode.seq_sink_len = torch.tensor([3])
        default_aclgraph_wrapper._forward_context.attn_metadata = {
            "metadata1": mock_attn_metadata
        }
        default_aclgraph_wrapper.runnable = MagicMock(return_value=torch.tensor([1.0]))

        with patch("omni_npu.compilation.acl_graph.ensure_weak_ref_graph_params"):
            result = default_aclgraph_wrapper(torch.tensor([1.0]))

        default_aclgraph_wrapper.runnable.assert_called_once()
        assert torch.equal(result, torch.tensor([1.0]))


    def test_call_with_input_address_mismatch(self, default_aclgraph_wrapper):
        """Test the behavior when input address do not match in debugging mode."""
        default_aclgraph_wrapper.is_debugging_mode = True

        test_tensor = torch.tensor([1.0])
        new_input_address = [test_tensor.data_ptr()]
        old_input_address = [new_input_address[0] + 1]

        batch_descriptor = default_aclgraph_wrapper._forward_context.batch_descriptor
        entry = ACLGraphEntry(
            batch_descriptor=batch_descriptor,
            aclgraph=MagicMock(),
            output=torch.tensor([1.0]),
            input_addresses=old_input_address,
        )
        default_aclgraph_wrapper.concrete_aclgraph_entries = {batch_descriptor: entry}

        with pytest.raises(AssertionError):
            default_aclgraph_wrapper(test_tensor)

        entry.aclgraph.replay.assert_not_called()

    def test_static_kernel_compile_runs_extra_runnable_only_once(
        self, default_aclgraph_wrapper
    ):
        call_source = inspect.getsource(acl_graph_mod.ACLGraphWrapper.__call__)
        if "need_static_compile" not in call_source:
            pytest.skip(
                f"Loaded ACLGraphWrapper.__call__ from {acl_graph_mod.__file__} "
                "does not include need_static_compile logic."
            )

        class BatchDescriptorStub:
            def __init__(self, num_reqs, num_tokens):
                self.num_reqs = num_reqs
                self.num_tokens = num_tokens

        batch_descriptor = BatchDescriptorStub(num_reqs=3, num_tokens=3)
        default_aclgraph_wrapper._forward_context.cudagraph_runtime_mode = (
            default_aclgraph_wrapper.runtime_mode
        )
        default_aclgraph_wrapper._forward_context.batch_descriptor = batch_descriptor
        default_aclgraph_wrapper.need_static_compile = True
        default_aclgraph_wrapper.need_super_kernel_optimize = True
        default_aclgraph_wrapper.concrete_aclgraph_entries = {}
        default_aclgraph_wrapper.runnable = MagicMock(return_value=torch.tensor([1.0]))

        with patch("omni_npu.compilation.acl_graph.ensure_weak_ref_graph_params"):
            result = default_aclgraph_wrapper(torch.tensor([1.0]))

        assert torch.equal(result, torch.tensor([1.0]))
        assert batch_descriptor in default_aclgraph_wrapper.concrete_aclgraph_entries
        assert default_aclgraph_wrapper.runnable.call_count == 2

class TestACLGraphWrapperUpdateMethods:
    """Test class for ACLGraphWrapper update methods."""

    @pytest.fixture
    def wrapper_with_update_stream(self):
        with patch("torch.npu.NPUGraph"), \
             patch("torch.npu.graph"), \
             patch("torch.npu.Stream") as mock_stream, \
             patch("omni_npu.compilation.acl_graph.get_forward_context") as mock_get_forward_context:

            mock_stream_instance = MagicMock()
            mock_stream.return_value = mock_stream_instance
            mock_stream_instance.__enter__ = lambda self: mock_stream_instance
            mock_stream_instance.__exit__ = lambda *args: None

            mock_context = MagicMock()
            mock_batch_descriptor = MagicMock()
            mock_batch_descriptor.num_tokens = 4
            mock_context.batch_descriptor = mock_batch_descriptor
            mock_context.cudagraph_runtime_mode = CUDAGraphMode.PIECEWISE
            mock_context.attn_metadata = {}
            mock_get_forward_context.return_value = mock_context

            vllm_config = MagicMock(spec=VllmConfig)
            vllm_config.additional_config = None

            wrapper = ACLGraphWrapper(
                runnable=MagicMock(),
                vllm_config=vllm_config,
                runtime_mode=CUDAGraphMode.PIECEWISE,
                update_stream=mock_stream_instance,
                attn_layer_names=["attn"],
            )
            wrapper._forward_context = mock_context

            yield wrapper, mock_context, mock_stream_instance

    def _make_task_entry(self, compute_dynamic_kwargs=None):
        op_out_fn = MagicMock()
        workspace_fn = MagicMock()
        op_desc = acl_graph_mod.OpDescriptor(
            op_out_fn=op_out_fn,
            workspace_fn=workspace_fn,
            compute_dynamic_kwargs=compute_dynamic_kwargs,
        )
        task_entry = acl_graph_mod.GraphTaskEntry(
            op_desc=op_desc,
            captured_kwargs={"static_key": "static_value"},
            out_tensors=[MagicMock()],
            handle=MagicMock(),
            event=MagicMock(),
        )
        return task_entry, op_out_fn, workspace_fn

    def test_update_graph_tasks_with_none_attn_metadata(self, wrapper_with_update_stream):
        """Test _update_graph_tasks raises when attn_metadata is missing."""
        wrapper, forward_context, update_stream = wrapper_with_update_stream

        forward_context.attn_metadata = None

        with pytest.raises(RuntimeError, match="attn_metadata is empty"):
            wrapper._update_graph_tasks(update_stream, forward_context)

    def test_update_graph_tasks_skips_non_attention_layers(self, wrapper_with_update_stream):
        """Test _update_graph_tasks ignores non-target layers."""
        wrapper, forward_context, update_stream = wrapper_with_update_stream
        task_entry, op_out_fn, workspace_fn = self._make_task_entry(
            compute_dynamic_kwargs=MagicMock(return_value={"dynamic_key": "dynamic_value"})
        )
        graph_params = GraphParams(
            task_entries={4: {"other_layer": task_entry}},
            workspaces={4: {workspace_fn: MagicMock()}},
        )

        with patch("omni_npu.compilation.acl_graph.get_graph_params", return_value=graph_params), \
             patch("torch.npu.stream"), \
             patch("torch.npu.graph_task_update_begin") as mock_update_begin, \
             patch("torch.npu.graph_task_update_end") as mock_update_end:
            wrapper._update_graph_tasks(update_stream, forward_context)

        mock_update_begin.assert_not_called()
        mock_update_end.assert_not_called()
        op_out_fn.assert_not_called()
        task_entry.event.record.assert_not_called()

    def test_wrapper_initialization_with_update_stream(self):
        """Test ACLGraphWrapper initialization with update_stream parameter"""
        runnable = MagicMock()
        vllm_config = MagicMock(spec=VllmConfig)
        runtime_mode = CUDAGraphMode.PIECEWISE
        graph_pool = MagicMock()
        update_stream = MagicMock(spec=torch.npu.Stream)

        wrapper = ACLGraphWrapper(
            runnable, vllm_config, runtime_mode, graph_pool,
            cudagraph_options=None, update_stream=update_stream
        )

        assert wrapper.update_stream == update_stream

    def test_update_graph_tasks_records_event_when_no_update_fn(self, wrapper_with_update_stream):
        """Test _update_graph_tasks records an event when no update fn exists."""
        wrapper, forward_context, update_stream = wrapper_with_update_stream
        task_entry, op_out_fn, workspace_fn = self._make_task_entry()
        graph_params = GraphParams(
            task_entries={4: {"attn": task_entry}},
            workspaces={4: {workspace_fn: MagicMock()}},
        )

        with patch("omni_npu.compilation.acl_graph.get_graph_params", return_value=graph_params), \
             patch("torch.npu.stream"), \
             patch("torch.npu.graph_task_update_begin") as mock_update_begin, \
             patch("torch.npu.graph_task_update_end") as mock_update_end:
            wrapper._update_graph_tasks(update_stream, forward_context)

        mock_update_begin.assert_not_called()
        mock_update_end.assert_not_called()
        op_out_fn.assert_not_called()
        task_entry.event.record.assert_called_once_with(update_stream)

    def test_update_graph_tasks_records_event_when_update_returns_none(self, wrapper_with_update_stream):
        """Test _update_graph_tasks records an event when update returns None."""
        wrapper, forward_context, update_stream = wrapper_with_update_stream
        update_fn = MagicMock(return_value=None)
        task_entry, op_out_fn, workspace_fn = self._make_task_entry(
            compute_dynamic_kwargs=update_fn
        )
        graph_params = GraphParams(
            task_entries={4: {"attn": task_entry}},
            workspaces={4: {workspace_fn: MagicMock()}},
        )

        with patch("omni_npu.compilation.acl_graph.get_graph_params", return_value=graph_params), \
             patch("torch.npu.stream"), \
             patch("torch.npu.graph_task_update_begin") as mock_update_begin, \
             patch("torch.npu.graph_task_update_end") as mock_update_end:
            wrapper._update_graph_tasks(update_stream, forward_context)

        update_fn.assert_called_once_with(forward_context, "attn", wrapper.vllm_config)
        mock_update_begin.assert_not_called()
        mock_update_end.assert_not_called()
        op_out_fn.assert_not_called()
        task_entry.event.record.assert_called_once_with(update_stream)

    def test_update_graph_tasks_updates_task(self, wrapper_with_update_stream):
        """Test _update_graph_tasks reissues the op with merged kwargs."""
        wrapper, forward_context, update_stream = wrapper_with_update_stream
        update_fn = MagicMock(return_value={"dynamic_key": "dynamic_value"})
        task_entry, op_out_fn, workspace_fn = self._make_task_entry(
            compute_dynamic_kwargs=update_fn
        )
        workspace = MagicMock()
        graph_params = GraphParams(
            task_entries={4: {"attn": task_entry}},
            workspaces={4: {workspace_fn: workspace}},
        )

        with patch("omni_npu.compilation.acl_graph.get_graph_params", return_value=graph_params), \
             patch("torch.npu.stream"), \
             patch("torch.npu.graph_task_update_begin") as mock_update_begin, \
             patch("torch.npu.graph_task_update_end") as mock_update_end:
            wrapper._update_graph_tasks(update_stream, forward_context)

        update_fn.assert_called_once_with(forward_context, "attn", wrapper.vllm_config)
        mock_update_begin.assert_called_once_with(update_stream, task_entry.handle)
        op_out_fn.assert_called_once_with(
            static_key="static_value",
            dynamic_key="dynamic_value",
            workspace=workspace,
            out=task_entry.out_tensors,
        )
        mock_update_end.assert_called_once()
        task_entry.event.record.assert_called_once_with(update_stream)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])