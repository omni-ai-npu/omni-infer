# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm.config import CUDAGraphMode
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID
from vllm.v1.spec_decode.utils import PADDING_SLOT_ID

from omni_npu.vllm_patches.usefull_patch import patch_eagle as eagle_mod
from omni_npu.vllm_patches.usefull_patch.patch_eagle import EagleProposerPatch


def _fake_proposer(attn_metadata):
    fake = SimpleNamespace()
    fake.runner = SimpleNamespace(
        _omni_spec_decode_common_attn_metadata=attn_metadata,
        batch_execution_and_padding_state=(
            CUDAGraphMode.NONE,
            SimpleNamespace(num_tokens=2),
            None,
        ),
        dp_parallel_lmhead=False,
        local_parallel_lmhead=False,
    )
    fake.attn_layer_names = ["layer0"]
    fake.num_speculative_tokens = 1
    fake.n_predict = 1
    fake.supports_mm_inputs = False
    fake.method = "eagle"
    fake.vllm_config = MagicMock()
    fake.input_ids = torch.arange(4)
    fake.inputs_embeds = torch.zeros(4, 2)
    fake.hidden_states = torch.zeros(4, 2)
    fake.model = MagicMock(return_value=torch.zeros(2, 2))

    def _arange_positions(num_tokens):
        return torch.arange(num_tokens)

    fake._get_positions = _arange_positions
    fake.build_per_group_and_layer_attn_metadata = MagicMock(
        return_value=(None, {"layer0": "built"})
    )
    return fake


@pytest.mark.unit
def test_eagle_dummy_run_pads_stashed_common_metadata(monkeypatch):
    orig_slots = torch.tensor([7, 8, 9], dtype=torch.int32)
    orig_blocks = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)
    meta = SimpleNamespace(
        slot_mapping=orig_slots.clone(),
        block_table_tensor=orig_blocks.clone(),
    )
    fake = _fake_proposer(meta)
    monkeypatch.setattr(eagle_mod, "set_forward_context", lambda **_kwargs: nullcontext())
    monkeypatch.setattr(
        eagle_mod, "get_forward_context", lambda: SimpleNamespace(capturing=True)
    )

    EagleProposerPatch.dummy_run(fake, attn_metadata=None)

    assert torch.equal(orig_slots, torch.tensor([7, 8, 9], dtype=torch.int32))
    assert torch.equal(orig_blocks, torch.tensor([[1, 2], [3, 4]], dtype=torch.int32))
    assert torch.equal(
        meta.slot_mapping,
        torch.full_like(orig_slots, PADDING_SLOT_ID),
    )
    assert torch.equal(
        meta.block_table_tensor,
        torch.full_like(orig_blocks, NULL_BLOCK_ID),
    )
    fake.build_per_group_and_layer_attn_metadata.assert_called_once()
    fake.model.assert_called_once()
    assert fake.runner.batch_execution_and_padding_state is None


@pytest.mark.unit
def test_eagle_dummy_run_without_metadata_skips_padding(monkeypatch):
    fake = _fake_proposer(None)
    monkeypatch.setattr(eagle_mod, "set_forward_context", lambda **_kwargs: nullcontext())
    monkeypatch.setattr(
        eagle_mod, "get_forward_context", lambda: SimpleNamespace(capturing=True)
    )

    EagleProposerPatch.dummy_run(fake, attn_metadata=None)

    fake.build_per_group_and_layer_attn_metadata.assert_not_called()
    fake.model.assert_called_once()
