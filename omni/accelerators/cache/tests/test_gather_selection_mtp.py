import json
from pathlib import Path
from types import SimpleNamespace

import pytest


GLM_DECODE_CONFIG = (
    Path(__file__).resolve().parents[4]
    / "omni"
    / "models"
    / "configs"
    / "glm5_w8a8c16_a3_1p1d_d.json"
)


def test_glm_decode_enables_request_grouped_mtp():
    config = json.loads(GLM_DECODE_CONFIG.read_text())

    assert config["operator_optimizition_config"][
        "mtp_remove_redundant_kv"
    ] is True


def test_batch_reorder_keeps_mtp_status_and_table_on_target_slots():
    torch = pytest.importorskip("torch")
    mla = pytest.importorskip("omni.layers.attention.backend.mla")
    builder = object.__new__(mla.AscendMLAMetadataBuilder)

    layer_count = 1
    batch_size = 3
    mtp_tokens = 3
    head_count = 1
    status_width = 2
    block_count = 2

    status = torch.arange(
        layer_count
        * batch_size
        * mtp_tokens
        * head_count
        * status_width,
        dtype=torch.int32,
    ).view(
        layer_count,
        batch_size,
        mtp_tokens,
        head_count,
        status_width,
    )
    table = torch.arange(
        batch_size * mtp_tokens * block_count,
        dtype=torch.int32,
    ).view(batch_size, mtp_tokens, block_count)
    original_status = status.clone()
    original_table = table.clone()

    omni_cache = SimpleNamespace(
        selection_kv_block_status_buffer=torch.empty_like(status),
        selection_kv_block_table_buffer=torch.empty_like(table),
    )
    old_req_ids = ["A", "B", "C"]
    new_req_ids = ["B", "D", "C"]

    builder._update_status_buffered(
        omni_cache,
        status,
        old_req_ids,
        new_req_ids,
        fill_value=-1,
    )
    builder._reorder_block_table_only(
        omni_cache,
        table,
        old_req_ids,
        new_req_ids,
    )

    assert torch.equal(status[:, 0], original_status[:, 1])
    assert torch.all(status[:, 1] == -1)
    assert torch.equal(status[:, 2], original_status[:, 2])
    assert torch.equal(table[0], original_table[1])
    assert torch.equal(table[1], original_table[0])
    assert torch.equal(table[2], original_table[2])


@pytest.mark.parametrize(
    "mtp_enabled,num_speculative_tokens,decode_gears,expected",
    [
        (False, 1, [2, 4], {2: [1, 2], 4: [1, 2, 3, 4]}),
        (True, 1, [2, 4], {2: [2], 4: [2, 4]}),
        (True, 2, [3, 6], {3: [3], 6: [3, 6]}),
    ],
)
def test_indexer_groups_decode_query_lengths_by_request(
    monkeypatch,
    mtp_enabled,
    num_speculative_tokens,
    decode_gears,
    expected,
):
    torch = pytest.importorskip("torch")
    deepseek_mla = pytest.importorskip(
        "omni.layers.attention.deepseek_mla"
    )

    class DummyLinear(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

    monkeypatch.setattr(deepseek_mla, "ReplicatedLinear", DummyLinear)
    monkeypatch.setattr(
        deepseek_mla,
        "get_had_pow2",
        lambda size: torch.ones(size, size, dtype=torch.bfloat16),
    )
    monkeypatch.setattr(
        deepseek_mla,
        "current_platform",
        SimpleNamespace(device_type="cpu"),
    )
    monkeypatch.setattr(
        deepseek_mla,
        "model_extra_config",
        SimpleNamespace(
            operator_opt_config=SimpleNamespace(
                mtp_remove_redundant_kv=mtp_enabled
            ),
            task_config=SimpleNamespace(decode_gear_list=decode_gears),
        ),
    )
    monkeypatch.setattr(
        deepseek_mla,
        "get_current_vllm_config",
        lambda: SimpleNamespace(
            speculative_config=SimpleNamespace(
                num_speculative_tokens=num_speculative_tokens
            )
        ),
    )

    config = SimpleNamespace(
        hidden_size=8,
        index_n_heads=1,
        index_head_dim=128,
        qk_rope_head_dim=64,
        index_topk=2048,
        q_lora_rank=4,
    )
    indexer = deepseek_mla.Indexer(config)

    assert {
        gear: lengths.tolist()
        for gear, lengths in indexer.actual_seq_lengths.items()
    } == expected


def _npu_modules():
    torch = pytest.importorskip("torch")
    torch_npu = pytest.importorskip("torch_npu")

    if not hasattr(torch_npu, "npu_gather_selection_kv_cache"):
        pytest.importorskip("custom_ops")
    if not hasattr(torch_npu, "npu_gather_selection_kv_cache"):
        pytest.skip("npu_gather_selection_kv_cache is not registered")

    torch_npu.npu.set_device(0)
    return torch, torch_npu


def _build_inputs(torch, *, request_grouped):
    device = "npu:0"
    block_size = 128
    full_kv_len = 8192
    topk = 2048
    query_seq = 2
    selection_slot_count = query_seq * topk
    selection_block_count = selection_slot_count // block_size
    full_block_count = (full_kv_len + 1 + block_size - 1) // block_size

    full_block_table = torch.arange(
        1,
        full_block_count + 1,
        dtype=torch.int32,
        device=device,
    ).view(1, -1)
    full_key = torch.zeros(
        full_block_count + 1,
        block_size,
        1,
        512,
        dtype=torch.bfloat16,
        device=device,
    )
    full_key_rope = torch.zeros(
        full_block_count + 1,
        block_size,
        1,
        64,
        dtype=torch.bfloat16,
        device=device,
    )

    replaced_topkid = full_kv_len - 1
    physical_block = int(
        full_block_table[0, replaced_topkid // block_size].item()
    )
    physical_offset = replaced_topkid % block_size
    full_key[physical_block, physical_offset, 0].fill_(1)

    selection_key = torch.zeros(
        selection_block_count,
        block_size,
        512,
        dtype=torch.bfloat16,
        device=device,
    )
    selection_key_rope = torch.zeros(
        selection_block_count,
        block_size,
        64,
        dtype=torch.bfloat16,
        device=device,
    )
    selection_block_table = torch.arange(
        selection_block_count,
        dtype=torch.int32,
        device=device,
    )

    if request_grouped:
        full_q_actual_seq = torch.tensor(
            [query_seq], dtype=torch.int32, device=device
        )
        full_kv_actual_seq = torch.tensor(
            [full_kv_len], dtype=torch.int32, device=device
        )
        selection_block_table = selection_block_table.view(1, -1)
        selection_status = torch.full(
            (1, selection_slot_count * 2),
            -1,
            dtype=torch.int32,
            device=device,
        )
    else:
        full_q_actual_seq = torch.arange(
            1, query_seq + 1, dtype=torch.int32, device=device
        )
        full_kv_actual_seq = torch.tensor(
            [full_kv_len - 1, full_kv_len],
            dtype=torch.int32,
            device=device,
        )
        full_block_table = full_block_table.repeat(query_seq, 1)
        selection_block_table = selection_block_table.view(query_seq, -1)
        selection_status = torch.full(
            (query_seq, selection_slot_count),
            -1,
            dtype=torch.int32,
            device=device,
        )

    return {
        "full_block_table": full_block_table,
        "full_key": full_key,
        "full_key_rope": full_key_rope,
        "full_kv_actual_seq": full_kv_actual_seq,
        "full_q_actual_seq": full_q_actual_seq,
        "selection_block_table": selection_block_table,
        "selection_key": selection_key,
        "selection_key_rope": selection_key_rope,
        "selection_status": selection_status,
        "full_kv_len": full_kv_len,
        "replaced_topkid": replaced_topkid,
        "physical_block": physical_block,
        "physical_offset": physical_offset,
    }


def _run_gather(torch_npu, inputs, topk):
    torch_npu.npu_gather_selection_kv_cache(
        selection_k_rope=inputs["selection_key_rope"],
        selection_kv_cache=inputs["selection_key"],
        selection_kv_block_table=inputs["selection_block_table"],
        selection_kv_block_status=inputs["selection_status"],
        selection_topk_indices=topk,
        full_k_rope=inputs["full_key_rope"].squeeze(-2),
        full_kv_cache=inputs["full_key"].squeeze(-2),
        full_kv_block_table=inputs["full_block_table"],
        full_kv_actual_seq=inputs["full_kv_actual_seq"],
        full_q_actual_seq=inputs["full_q_actual_seq"],
        selection_topk_block_size=1,
    )


def _selection_value(inputs, table_row, slot):
    slot = int(slot.item())
    block = int(
        inputs["selection_block_table"][table_row, slot // 128].item()
    )
    return inputs["selection_key"][block, slot % 128]


def test_mtp_rejection_clears_draft_mapping_and_prevents_stale_kv_reuse():
    torch, torch_npu = _npu_modules()
    grouped = _build_inputs(torch, request_grouped=True)
    expanded = _build_inputs(torch, request_grouped=False)
    full_kv_len = grouped["full_kv_len"]
    replaced_topkid = grouped["replaced_topkid"]

    first_topk = torch.stack(
        [
            torch.arange(0, 2048, dtype=torch.int32, device="npu:0"),
            torch.arange(
                full_kv_len - 2048,
                full_kv_len,
                dtype=torch.int32,
                device="npu:0",
            ),
        ],
        dim=0,
    ).view(2, 1, 1, 2048)

    _run_gather(torch_npu, grouped, first_topk.clone())
    _run_gather(torch_npu, expanded, first_topk.clone())

    grouped_topkids = grouped["selection_status"].view(1, 4096, 2)[..., 0]
    expanded_topkids = expanded["selection_status"].view(2, 2048, 2)[..., 0]
    assert torch.count_nonzero(
        grouped_topkids == replaced_topkid
    ).item() == 0
    assert torch.count_nonzero(
        expanded_topkids[1] == replaced_topkid
    ).item() > 0

    old_value = grouped["full_key"][
        grouped["physical_block"],
        grouped["physical_offset"],
        0,
    ].clone()
    new_value = old_value.clone()
    new_value.fill_(37)
    for inputs in (grouped, expanded):
        inputs["full_key"][
            inputs["physical_block"],
            inputs["physical_offset"],
            0,
        ].copy_(new_value)

    grouped["full_kv_actual_seq"].fill_(full_kv_len + 1)
    expanded["full_kv_actual_seq"] = torch.tensor(
        [full_kv_len, full_kv_len + 1],
        dtype=torch.int32,
        device="npu:0",
    )
    second_topk = torch.stack(
        [
            torch.arange(2048, 4096, dtype=torch.int32, device="npu:0"),
            torch.cat(
                [
                    torch.arange(
                        4096,
                        4096 + 2047,
                        dtype=torch.int32,
                        device="npu:0",
                    ),
                    torch.tensor(
                        [replaced_topkid],
                        dtype=torch.int32,
                        device="npu:0",
                    ),
                ]
            ),
        ],
        dim=0,
    ).view(2, 1, 1, 2048)
    replaced_position = 2047
    grouped_slots = second_topk.clone()
    expanded_slots = second_topk.clone()

    _run_gather(torch_npu, grouped, grouped_slots)
    _run_gather(torch_npu, expanded, expanded_slots)

    grouped_value = _selection_value(
        grouped, 0, grouped_slots[1, 0, 0, replaced_position]
    )
    expanded_value = _selection_value(
        expanded, 1, expanded_slots[1, 0, 0, replaced_position]
    )

    assert torch.equal(grouped_value, new_value)
    assert torch.equal(expanded_value, old_value)
