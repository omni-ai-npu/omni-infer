# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Tests for core.deploy.finalize_model — compressed-tensors assembly."""
import json

import torch
from safetensors.torch import load_file, save_file

from jointfix.core.deploy import finalize_model
from jointfix.core.primitives import UNIVERSAL_SKIP_PATTERNS, rtn_quantize


def _make_orig(model_dir):
    model_dir.mkdir(parents=True, exist_ok=True)
    tensors = {
        "model.layers.0.self_attn.o_proj.weight": torch.randn(4, 8),  # quantized (in layer file)
        "model.embed_tokens.weight": torch.randn(16, 8),              # skip -> passthrough
        "model.norm.weight": torch.ones(8),                          # 1D -> passthrough
        "lm_head.weight": torch.randn(16, 8),                        # skip -> passthrough
    }
    save_file(tensors, str(model_dir / "model.safetensors"))         # single-file (no index)
    (model_dir / "config.json").write_text(json.dumps({"hidden_size": 8}))
    (model_dir / "tokenizer.json").write_text("FAKE TOKENIZER")      # aux file


def _make_quant(quant_dir):
    quant_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    w = torch.randn(4, 8)
    q, s = rtn_quantize(w)
    save_file({
        "model.layers.0.self_attn.o_proj.weight": q,
        "model.layers.0.self_attn.o_proj.weight_scale": s,
    }, str(quant_dir / "layer_0000.safetensors"))


def test_finalize_assembles_compressed_tensors(tmp_path):
    orig = tmp_path / "orig"
    qd = tmp_path / "quant"
    out = tmp_path / "deploy"
    _make_orig(orig)
    _make_quant(qd)

    finalize_model(str(orig), str(qd), str(out), skip_patterns=UNIVERSAL_SKIP_PATTERNS)

    # shard reuses the original filename
    shard = load_file(str(out / "model.safetensors"))
    assert shard["model.layers.0.self_attn.o_proj.weight"].dtype == torch.int8
    assert "model.layers.0.self_attn.o_proj.weight_scale" in shard
    assert shard["model.embed_tokens.weight"].dtype != torch.int8     # passthrough
    assert shard["model.norm.weight"].dtype != torch.int8             # 1D passthrough
    assert "lm_head.weight_scale" not in shard                        # skip -> not quantized

    # index maps every tensor
    index = json.loads((out / "model.safetensors.index.json").read_text())["weight_map"]
    assert index["model.layers.0.self_attn.o_proj.weight"] == "model.safetensors"
    assert index["model.layers.0.self_attn.o_proj.weight_scale"] == "model.safetensors"

    # config.json gains the quantization_config; ignore lists the skip-pattern weights
    cfg = json.loads((out / "config.json").read_text())
    qc = cfg["quantization_config"]
    assert qc["quant_method"] == "compressed-tensors"
    assert qc["quantize"] == "w8a8_dynamic"
    assert "model.embed_tokens" in qc["ignore"]
    assert "lm_head" in qc["ignore"]
    assert "model.layers.0.self_attn.o_proj" not in qc["ignore"]      # it IS quantized

    # aux file copied
    assert (out / "tokenizer.json").read_text() == "FAKE TOKENIZER"


def test_finalize_ignores_bf16_passthrough_kept_weights(tmp_path):
    """
    A quantizable-shape weight kept BF16 (e.g. --skip-shared-experts adds the
    skip at the *method* layer, invisible to the backend skip list finalize gets)
    must still land in `ignore` — otherwise vLLM treats it as quantized, looks for a
    weight_scale that isn't there, and fails to load. The deciding signal is the
    DATA: a 2D weight with no weight_scale sibling is un-quantized, period.
    """
    orig, qd = tmp_path / "orig", tmp_path / "quant"
    orig.mkdir(parents=True)
    qd.mkdir(parents=True)
    save_file({
        "model.layers.0.mlp.experts.0.down_proj.weight": torch.randn(4, 8),       # int8
        "model.layers.0.mlp.shared_experts.down_proj.weight": torch.randn(4, 8),   # BF16 kept
        "model.norm.weight": torch.ones(8),                                        # 1D
    }, str(orig / "model.safetensors"))
    (orig / "config.json").write_text(json.dumps({"hidden_size": 8}))
    q, s = rtn_quantize(torch.randn(4, 8))
    save_file({
        "model.layers.0.mlp.experts.0.down_proj.weight": q,
        "model.layers.0.mlp.experts.0.down_proj.weight_scale": s,
        "model.layers.0.mlp.shared_experts.down_proj.weight": torch.randn(4, 8).bfloat16(),
    }, str(qd / "layer_0000.safetensors"))

    # backend skip list does NOT contain mlp.shared_experts (that's a method-level skip)
    finalize_model(str(orig), str(qd), None, skip_patterns=UNIVERSAL_SKIP_PATTERNS)

    ignore = json.loads((qd / "config.json").read_text())["quantization_config"]["ignore"]
    assert "model.layers.0.mlp.shared_experts.down_proj" in ignore   # BF16 -> must be ignored
    assert "model.layers.0.mlp.experts.0.down_proj" not in ignore    # int8 -> quantized
    assert "model.norm" not in ignore                                # 1D -> not a Linear
