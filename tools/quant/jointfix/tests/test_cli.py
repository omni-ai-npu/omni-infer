# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""
CLI behavior: `quantize` assembles a deployable model in one step by default,
with `--no-finalize` to stop at the per-layer artifacts.
"""
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from jointfix.core.primitives import UNIVERSAL_SKIP_PATTERNS, rtn_quantize


def _make_orig(model_dir):
    model_dir.mkdir(parents=True, exist_ok=True)
    save_file({
        "model.layers.0.self_attn.o_proj.weight": torch.randn(4, 8),  # quantized
        "model.embed_tokens.weight": torch.randn(16, 8),              # skip -> passthrough
        "model.norm.weight": torch.ones(8),                           # 1D -> passthrough
        "lm_head.weight": torch.randn(16, 8),                         # skip -> passthrough
    }, str(model_dir / "model.safetensors"))
    (model_dir / "config.json").write_text(json.dumps({"hidden_size": 8}))


def _make_quant(quant_dir):
    """Mimic what the runner leaves in --output: per-layer int8 files."""
    quant_dir.mkdir(parents=True, exist_ok=True)
    q, s = rtn_quantize(torch.randn(4, 8))
    save_file({
        "model.layers.0.self_attn.o_proj.weight": q,
        "model.layers.0.self_attn.o_proj.weight_scale": s,
    }, str(quant_dir / "layer_0000.safetensors"))


# ── the --no-finalize flag ──────────────────────────────────────────────────
def test_quantize_finalize_is_default_on():
    from jointfix.cli import build_parser
    args = build_parser().parse_args(
        ["quantize", "--backend", "pangu", "--method", "jointfix",
         "--model", "m", "--output", "o", "--calib-data", "c"])
    assert args.no_finalize is False          # one-step by default
    assert args.calib_format == "text"        # legacy language calibration path


def test_no_finalize_flag_opts_out():
    from jointfix.cli import build_parser
    args = build_parser().parse_args(
        ["quantize", "--backend", "pangu", "--method", "jointfix",
         "--model", "m", "--output", "o", "--calib-data", "c", "--no-finalize"])
    assert args.no_finalize is True


def test_legacy_jointfix_command_keeps_text_defaults(monkeypatch):
    """The documented language-only production command must remain parse-compatible."""
    from jointfix.cli import build_parser

    argv = [
        "jointfix", "quantize", "--backend", "pangu", "--method", "jointfix",
        "--model", "bf16", "--output", "w8a8",
        "--calib-data", "examples/data/wikitext_train.parquet",
        "--n-samples", "32", "--seq-len", "1024",
        "--num-iterations", "2", "--iter-ab-tol", "0.05",
        "--num-devices", "16", "--device", "npu",
        "--objective", "output-recon", "--write-quant", "gptq",
        "--skip-shared-experts",
    ]
    monkeypatch.setattr("sys.argv", argv)
    args = build_parser().parse_args(argv[1:])

    assert args.method == "jointfix"
    assert args.calib_format == "text"
    assert args.n_samples == 32 and args.seq_len == 1024
    assert args.num_iterations == 2 and args.iter_ab_tol == 0.05
    assert args.num_devices == 16 and args.device == "npu"
    assert args.objective == "output-recon" and args.write_quant == "gptq"
    assert args.skip_shared_experts is True


def test_omni_calibration_cli_flags():
    from jointfix.cli import build_parser
    args = build_parser().parse_args(
        ["quantize", "--backend", "pangu", "--method", "jointfix",
         "--model", "m", "--output", "o", "--calib-data", "omni.jsonl",
         "--calib-format", "omni", "--mm-max-pixels", "200704"])
    assert args.calib_format == "omni"
    assert args.mm_max_pixels == 200704


def test_jointfix_mdmixq_is_registered():
    from jointfix.registry import available_methods
    assert "jointfix-mdmixq" in available_methods()


# ── the auto-finalize dispatch ──────────────────────────────────────────────
def test_finalize_deploy_assembles_in_place_into_output(tmp_path):
    """
    Enabled: assemble the deployable model in-place into --output, leaving the
    per-layer intermediates beside it (so a later re-finalize is still possible).
    """
    from jointfix.cli import _finalize_deploy
    orig, out = tmp_path / "orig", tmp_path / "out"
    _make_orig(orig)
    _make_quant(out)                                       # --output already holds layer_0000

    deploy = _finalize_deploy(str(orig), str(out), UNIVERSAL_SKIP_PATTERNS, enabled=True)

    assert Path(deploy) == out                             # in-place: deploy dir IS --output
    shard = load_file(str(out / "model.safetensors"))
    assert shard["model.layers.0.self_attn.o_proj.weight"].dtype == torch.int8
    cfg = json.loads((out / "config.json").read_text())
    assert cfg["quantization_config"]["quantize"] == "w8a8_dynamic"
    assert (out / "layer_0000.safetensors").exists()       # intermediates preserved


def test_finalize_deploy_skipped_when_disabled(tmp_path):
    """--no-finalize: do nothing, return None, leave only the per-layer artifacts."""
    from jointfix.cli import _finalize_deploy
    orig, out = tmp_path / "orig", tmp_path / "out"
    _make_orig(orig)
    _make_quant(out)

    deploy = _finalize_deploy(str(orig), str(out), UNIVERSAL_SKIP_PATTERNS, enabled=False)

    assert deploy is None
    assert not (out / "model.safetensors.index.json").exists()   # nothing assembled


# ── finalize honors --skip-shared-experts (uncalibrated / MTP weights) ──────
def test_finalize_subcommand_has_skip_shared_experts_flag():
    from jointfix.cli import build_parser
    a = build_parser().parse_args(["finalize", "--model", "m", "--quantized", "q"])
    assert a.skip_shared_experts is False                 # default off
    assert a.method == "jointfix"                         # legacy MTP behavior
    a = build_parser().parse_args(
        ["finalize", "--model", "m", "--quantized", "q", "--skip-shared-experts"])
    assert a.skip_shared_experts is True


def test_deploy_skip_patterns_extends_with_shared_experts():
    from jointfix.cli import _deploy_skip_patterns

    class _Backend:                       # minimal stub: only skip_patterns matters
        def skip_patterns(self):
            return ["embed", "lm_head"]

    class _Args:
        skip_shared_experts = False

    base = _deploy_skip_patterns(_Backend(), _Args())
    assert "mlp.shared_experts" not in base

    _Args.skip_shared_experts = True
    ext = _deploy_skip_patterns(_Backend(), _Args())
    assert "mlp.shared_experts" in ext
    assert base == ["embed", "lm_head"]                   # original list not mutated


def test_mtp_bf16_skip_is_mdmixq_only():
    from jointfix.cli import _deploy_skip_patterns

    class _Backend:
        def skip_patterns(self):
            return ["embed", "lm_head"]

        def mtp_skip_patterns(self):
            return ["model.layers.46.", "model.layers.47.", "model.layers.48."]

    class _Args:
        skip_shared_experts = True
        method = "jointfix"

    legacy = _deploy_skip_patterns(_Backend(), _Args())
    assert "mlp.shared_experts" in legacy
    assert "model.layers.46." not in legacy

    _Args.method = "jointfix-mdmixq"
    mdmixq = _deploy_skip_patterns(_Backend(), _Args())
    assert "mlp.shared_experts" in mdmixq
    assert "model.layers.46." in mdmixq
    assert "model.layers.48." in mdmixq


def test_finalize_quantizes_mtp_for_jointfix_but_keeps_it_bf16_for_mdmixq(tmp_path):
    from jointfix.cli import _deploy_skip_patterns
    from jointfix.core.deploy import finalize_model

    class _Backend:
        def skip_patterns(self):
            return list(UNIVERSAL_SKIP_PATTERNS)

        def mtp_skip_patterns(self):
            return ["model.layers.46."]

    class _Args:
        skip_shared_experts = False
        method = "jointfix"

    orig = tmp_path / "orig"
    orig.mkdir()
    mtp_weight = "model.layers.46.self_attn.o_proj.weight"
    save_file({
        mtp_weight: torch.randn(4, 8),
        "model.norm.weight": torch.ones(8),
    }, str(orig / "model.safetensors"))
    (orig / "config.json").write_text(json.dumps({"hidden_size": 8}))

    legacy = tmp_path / "legacy"
    legacy.mkdir()
    finalize_model(
        str(orig), str(legacy), None,
        skip_patterns=_deploy_skip_patterns(_Backend(), _Args()),
        rtn_uncalibrated=True,
    )
    legacy_shard = load_file(str(legacy / "model.safetensors"))
    assert legacy_shard[mtp_weight].dtype == torch.int8
    assert mtp_weight.replace(".weight", ".weight_scale") in legacy_shard

    _Args.method = "jointfix-mdmixq"
    mdmixq = tmp_path / "mdmixq"
    mdmixq.mkdir()
    finalize_model(
        str(orig), str(mdmixq), None,
        skip_patterns=_deploy_skip_patterns(_Backend(), _Args()),
        rtn_uncalibrated=True,
    )
    mdmixq_shard = load_file(str(mdmixq / "model.safetensors"))
    assert mdmixq_shard[mtp_weight].dtype != torch.int8
    assert mtp_weight.replace(".weight", ".weight_scale") not in mdmixq_shard


def test_uncalibrated_shared_expert_kept_bf16_when_skip(tmp_path):
    """
    The MTP bug: a shared-expert weight the calibration loop never visited (no
    per-layer file -> 'uncalibrated', like layers 46/47/48) must NOT be RTN-quantized
    under --skip-shared-experts. It must stay BF16 and land in ignore, like monolith.
    """
    from jointfix.cli import _deploy_skip_patterns
    from jointfix.core.deploy import finalize_model

    class _Backend:
        def skip_patterns(self):
            return list(UNIVERSAL_SKIP_PATTERNS)          # backend list has NO shared_experts

    class _Args:
        skip_shared_experts = True

    orig, qd = tmp_path / "orig", tmp_path / "quant"
    orig.mkdir(parents=True)
    qd.mkdir(parents=True)
    save_file({
        "model.layers.48.mlp.shared_experts.down_proj.weight": torch.randn(4, 8),  # MTP-like
        "model.norm.weight": torch.ones(8),
    }, str(orig / "model.safetensors"))
    (orig / "config.json").write_text(json.dumps({"hidden_size": 8}))
    # qd has NO layer_*.safetensors -> the shared expert is "uncalibrated"

    finalize_model(str(orig), str(qd), None,
                   skip_patterns=_deploy_skip_patterns(_Backend(), _Args()),
                   rtn_uncalibrated=True)

    shard = load_file(str(qd / "model.safetensors"))
    se = "model.layers.48.mlp.shared_experts.down_proj"
    assert shard[se + ".weight"].dtype != torch.int8         # BF16, not RTN-quantized
    assert (se + ".weight_scale") not in shard
    ignore = json.loads((qd / "config.json").read_text())["quantization_config"]["ignore"]
    assert se in ignore
