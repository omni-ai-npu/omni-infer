# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Merge calibrated modality artifacts into an existing language+MTP checkpoint."""
from __future__ import annotations

import json
import os
import re
import shutil
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file


def _expected_vision_weights(depth: int = 24):
    suffixes = ("attn.qkv", "attn.proj", "mlp.up_proj", "mlp.down_proj")
    return {
        f"visual.blocks.{layer}.{suffix}.weight"
        for layer in range(depth)
        for suffix in suffixes
    }


def _expected_audio_weights(depth: int = 16):
    suffixes = (
        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
        "self_attn.out_proj", "fc1", "fc2",
    )
    return {
        f"audio_tower.layers.{layer}.{suffix}.weight"
        for layer in range(depth)
        for suffix in suffixes
    }


def _copy_or_link(source: Path, target: Path, link: bool):
    if link:
        try:
            os.link(source, target)
            return
        except OSError:
            shutil.copy2(source, target)
            return
    shutil.copy2(source, target)


def _validate_multimodal_serving_config(config: dict, quantized_bases: set[str]) -> None:
    model_type = str(config.get("model_type", "")).lower()
    if "openpangu" not in model_type or not any(tag in model_type for tag in ("vl", "omni")):
        raise RuntimeError(
            "final config is not an OpenPangu VL/OMNI model_type: "
            f"{config.get('model_type')!r}"
        )
    architectures = config.get("architectures", [])
    expected_tag = "Omni" if "omni" in model_type else "VL"
    if not any(expected_tag.lower() in str(name).lower() for name in architectures):
        raise RuntimeError(
            f"final config does not select an OpenPangu {expected_tag} architecture: "
            f"{architectures!r}"
        )

    quant_config = config.get("quantization_config") or {}
    expected_fields = {
        "quant_method": "compressed-tensors",
        "quantize": "w8a8_dynamic",
        "format": "int-quantized",
        "quantization_status": "compressed",
    }
    wrong = {
        key: (quant_config.get(key), expected)
        for key, expected in expected_fields.items()
        if quant_config.get(key) != expected
    }
    if wrong:
        raise RuntimeError(f"final compressed-tensors config is invalid: {wrong}")
    group = quant_config.get("config_groups", {}).get("group_0", {})
    if group.get("targets") != ["Linear"]:
        raise RuntimeError(f"final qconfig has wrong targets: {group.get('targets')!r}")

    ignore = set(quant_config.get("ignore", []))
    stale = quantized_bases & ignore
    if stale:
        raise RuntimeError(
            "calibrated multimodal INT8 modules remain in qconfig.ignore: "
            f"{sorted(stale)}"
        )
    raw_language = sorted(name for name in ignore if name.startswith("model."))
    if raw_language:
        raise RuntimeError(
            "raw model.* ignore names are invalid for the Omni main model: "
            f"{raw_language[:8]}"
        )
    if not any(name.startswith("openpangu.language_model.") for name in ignore):
        raise RuntimeError("final qconfig has no Omni-mapped language ignore entries")
    if int(config.get("num_nextn_predict_layers", 0)) > 0:
        if not any(name.startswith("re:^model\\.layers\\.") for name in ignore):
            raise RuntimeError("final qconfig has no exact MTP regex ignore entries")


def _map_omni_runtime_ignore(config: dict) -> None:
    """Map checkpoint tensor names to the module names used by Omni serving."""
    quant_config = config.get("quantization_config")
    if not quant_config:
        raise RuntimeError("finalized text checkpoint has no quantization_config")
    num_hidden_layers = int(config["num_hidden_layers"])
    num_mtp_layers = int(config.get("num_nextn_predict_layers", 0))
    mtp_stop = num_hidden_layers + num_mtp_layers
    mapped = []
    for name in quant_config.get("ignore", []):
        match = re.match(r"^model\.layers\.(\d+)\.", name)
        if match and num_hidden_layers <= int(match.group(1)) < mtp_stop:
            target = "re:^" + re.escape(name) + "$"
        elif name == "lm_head":
            target = "openpangu.language_model.lm_head"
        elif name.startswith("model."):
            target = "openpangu.language_model." + name
        else:
            target = name
        if target not in mapped:
            mapped.append(target)
    quant_config["ignore"] = mapped


def validate_text_base_checkpoint(model_dir: str) -> None:
    """Validate deployable W8A8 LLM + BF16 MTP before modality replacement."""
    root = Path(model_dir)
    config_path = root / "config.json"
    index_path = root / "model.safetensors.index.json"
    if not config_path.is_file() or not index_path.is_file():
        raise FileNotFoundError(
            "text base must be a deployable checkpoint with config/index: "
            f"{root}"
        )
    config = json.loads(config_path.read_text())
    _validate_multimodal_serving_config(config, set())
    ignore = set(config["quantization_config"]["ignore"])
    required_bf16 = {
        "visual.merger.mlp.0", "visual.merger.mlp.2",
        "visual.vision_projection.fc1",
    }
    if int(config.get("audio_config", {}).get("encoder_layers", 0)) > 0:
        required_bf16.add("audio_tower.proj")
    if not required_bf16.issubset(ignore):
        raise RuntimeError(
            "text base does not preserve required non-backbone modules in BF16: "
            f"{sorted(required_bf16 - ignore)}"
        )

    index = json.loads(index_path.read_text())["weight_map"]
    by_shard = defaultdict(list)
    for name, shard in index.items():
        if name.startswith(("model.layers.", "visual.", "audio_tower.")):
            by_shard[shard].append(name)
    num_hidden_layers = int(config["num_hidden_layers"])
    counts = defaultdict(int)
    for shard, names in by_shard.items():
        with safe_open(str(root / shard), framework="pt") as handle:
            for name in names:
                if name.endswith(".weight_scale") and name.startswith(("visual.", "audio_tower.")):
                    raise RuntimeError(f"text base unexpectedly quantized a modality: {name}")
                if not name.endswith(".weight"):
                    continue
                tensor_slice = handle.get_slice(name)
                if len(tensor_slice.get_shape()) != 2:
                    continue
                dtype = tensor_slice.get_dtype()
                if name.startswith("visual.blocks."):
                    counts[("vision", dtype)] += 1
                elif name.startswith("audio_tower.layers."):
                    counts[("audio", dtype)] += 1
                elif name.startswith("model.layers."):
                    layer = int(name.split(".")[2])
                    kind = "language" if layer < num_hidden_layers else "mtp"
                    counts[(kind, dtype)] += 1
    vision_depth = int(config.get("vision_config", {}).get("depth", 0))
    audio_depth = int(config.get("audio_config", {}).get("encoder_layers", 0))
    expected_vision = vision_depth * 4
    expected_audio = audio_depth * 6
    if counts[("vision", "BF16")] != expected_vision:
        raise RuntimeError(f"text base modalities are not intact BF16 backbones: {dict(counts)}")
    if audio_depth and counts[("audio", "BF16")] != expected_audio:
        raise RuntimeError(f"text base audio is not an intact BF16 backbone: {dict(counts)}")
    if counts[("vision", "I8")] or counts[("audio", "I8")]:
        raise RuntimeError(f"text base contains premature modality INT8 weights: {dict(counts)}")
    if counts[("language", "I8")] == 0:
        raise RuntimeError(f"text base is missing language INT8 weights: {dict(counts)}")
    n_mtp = int(config.get("num_nextn_predict_layers", 0))
    if counts[("mtp", "I8")] != 0 or (n_mtp and counts[("mtp", "BF16")] == 0):
        raise RuntimeError(f"MTP must be entirely BF16: {dict(counts)}")
    print(
        "VALID text base: "
        f"LLM_INT8={counts[('language', 'I8')]} "
        f"MTP_BF16={counts[('mtp', 'BF16')]} "
        f"ViT=BF16({expected_vision}) Audio=BF16({expected_audio}); serving config valid",
        flush=True,
    )


def finalize_text_artifacts(
    original_model_dir: str,
    text_artifact_dir: str,
    output_dir: str,
) -> Path:
    """Finalize calibrated LLM layers, preserving MTP and modalities in BF16."""
    from jointfix.backends.pangu import PanguBackend
    from jointfix.core.deploy import finalize_model

    original = Path(original_model_dir)
    artifacts = Path(text_artifact_dir)
    output = Path(output_dir)
    config = json.loads((original / "config.json").read_text())
    depth = int(config["num_hidden_layers"])
    expected = {f"layer_{layer:04d}.safetensors" for layer in range(depth)}
    actual = {path.name for path in artifacts.glob("layer_*.safetensors")}
    if actual != expected:
        raise RuntimeError(
            "text artifacts are incomplete; "
            f"missing={sorted(expected - actual)}, unexpected={sorted(actual - expected)}"
        )
    if output.exists():
        raise FileExistsError(f"text base output must not exist: {output}")

    backend = PanguBackend(str(original))
    skip = backend.skip_patterns() + [
        "mlp.shared_experts",
        "visual.",
        "audio_tower.",
    ]
    finalize_model(
        str(original), str(artifacts), str(output),
        skip_patterns=skip, rtn_uncalibrated=True,
    )
    output_config_path = output / "config.json"
    output_config = json.loads(output_config_path.read_text())
    _map_omni_runtime_ignore(output_config)
    output_config_path.write_text(
        json.dumps(output_config, indent=2, ensure_ascii=False) + "\n"
    )
    validate_text_base_checkpoint(str(output))
    (output / "text_base_finalize_metadata.json").write_text(
        json.dumps({
            "original_checkpoint": str(original.resolve()),
            "text_artifacts": str(artifacts.resolve()),
            "language_algorithm": "jointfix_or_jointfix_mdmixq",
            "mtp_weight_format": "bfloat16",
            "multimodal_towers": "preserved_bf16_for_later_jointfix",
        }, indent=2, ensure_ascii=False) + "\n"
    )
    return output


def validate_multimodal_artifacts(artifact_dir: str, *, require_vision=True,
                                  require_audio=True, model_dir: str | None = None):
    artifacts = Path(artifact_dir)
    if not artifacts.is_dir():
        raise FileNotFoundError(f"multimodal artifact directory not found: {artifacts}")
    metadata_path = artifacts / "multimodal_quant_metadata.json"
    if not metadata_path.is_file():
        raise RuntimeError("multimodal artifacts have no v2 quantization metadata")
    metadata = json.loads(metadata_path.read_text())
    config_root = Path(model_dir or metadata.get("model", ""))
    if not (config_root / "config.json").is_file():
        raise RuntimeError(
            "cannot determine model config for dynamic tower depths; pass model_dir "
            "or keep metadata.model accessible"
        )
    model_config = json.loads((config_root / "config.json").read_text())
    vision_depth = int(model_config.get("vision_config", {}).get("depth", 0))
    audio_depth = int(model_config.get("audio_config", {}).get("encoder_layers", 0))
    if require_vision and vision_depth <= 0:
        raise RuntimeError("model config has no positive vision_config.depth")
    if require_audio and audio_depth <= 0:
        raise RuntimeError("model config has no positive audio_config.encoder_layers")
    if metadata.get("algorithm") != "jointfix_strict_block_gptq_v2":
        raise RuntimeError(
            "refusing pre-v2 multimodal artifacts without bounded smoothing and "
            f"reconstruction gates: algorithm={metadata.get('algorithm')!r}"
        )
    if metadata.get("gelu_cross_layer_smoothing") is not False:
        raise RuntimeError("v2 artifacts do not explicitly disable GELU cross-layer smoothing")
    completed = set(metadata.get("completed_modalities", []))
    required_modalities = (
        ({"vision"} if require_vision else set())
        | ({"audio"} if require_audio else set())
    )
    if not required_modalities.issubset(completed):
        raise RuntimeError(
            f"v2 metadata is missing completed modalities: {required_modalities - completed}"
        )
    vision_files = sorted(artifacts.glob("visual_*.safetensors"))
    audio_files = sorted(artifacts.glob("audio_*.safetensors"))
    expected_vision_files = {
        f"visual_layer_{layer:04d}.safetensors" for layer in range(vision_depth)
    }
    expected_audio_files = {
        f"audio_layer_{layer:04d}.safetensors" for layer in range(audio_depth)
    }
    actual_vision_files = {path.name for path in vision_files}
    actual_audio_files = {path.name for path in audio_files}
    if require_vision and actual_vision_files != expected_vision_files:
        raise RuntimeError(
            f"ViT artifacts must contain exactly {vision_depth} backbone layer files; "
            f"missing={sorted(expected_vision_files - actual_vision_files)}, "
            f"unexpected={sorted(actual_vision_files - expected_vision_files)}"
        )
    if require_audio and actual_audio_files != expected_audio_files:
        raise RuntimeError(
            f"Audio artifacts must contain exactly {audio_depth} backbone layer files; "
            f"missing={sorted(expected_audio_files - actual_audio_files)}, "
            f"unexpected={sorted(actual_audio_files - expected_audio_files)}"
        )

    replacements = {}
    for artifact in vision_files + audio_files:
        for name, tensor in load_file(str(artifact)).items():
            if name in replacements:
                raise RuntimeError(f"duplicate calibrated tensor in artifacts: {name}")
            replacements[name] = tensor
    int8_weights = {
        name for name, tensor in replacements.items()
        if name.endswith(".weight") and tensor.dtype == torch.int8
    }
    vision_weights = {name for name in int8_weights if name.startswith("visual.")}
    audio_weights = {name for name in int8_weights if name.startswith("audio_tower.")}
    expected_vision_weights = _expected_vision_weights(vision_depth)
    expected_audio_weights = _expected_audio_weights(audio_depth)
    if require_vision and vision_weights != expected_vision_weights:
        raise RuntimeError(
            "ViT INT8 tensors must be exactly the 96 backbone Block Linear weights; "
            f"missing={sorted(expected_vision_weights - vision_weights)}, "
            f"unexpected={sorted(vision_weights - expected_vision_weights)}"
        )
    if require_audio and audio_weights != expected_audio_weights:
        raise RuntimeError(
            "Audio INT8 tensors must be exactly the 96 backbone Layer Linear weights; "
            f"missing={sorted(expected_audio_weights - audio_weights)}, "
            f"unexpected={sorted(audio_weights - expected_audio_weights)}"
        )
    vision_count = len(vision_weights)
    audio_count = len(audio_weights)
    for name in int8_weights:
        scale = name.removesuffix(".weight") + ".weight_scale"
        if scale not in replacements:
            raise RuntimeError(f"calibrated INT8 weight has no scale: {name}")
        scale_tensor = replacements[scale].float()
        if not torch.isfinite(scale_tensor).all() or (scale_tensor <= 0).any():
            raise RuntimeError(f"invalid non-positive/non-finite scale: {scale}")
        if float(scale_tensor.max().item()) > 10.0:
            raise RuntimeError(
                f"catastrophic modality weight scale (>10) detected: {scale} "
                f"max={float(scale_tensor.max().item())}"
            )

    for kind, depth, required in (
        ("vision", vision_depth, require_vision),
        ("audio", audio_depth, require_audio),
    ):
        if not required:
            continue
        metrics_path = artifacts / f"{kind}_reconstruction_metrics.json"
        if not metrics_path.is_file():
            raise RuntimeError(f"missing {kind} per-layer reconstruction metrics")
        metrics = json.loads(metrics_path.read_text())
        expected_prefix = "visual.blocks" if kind == "vision" else "audio_tower.layers"
        expected_layers = {f"{expected_prefix}.{idx}" for idx in range(depth)}
        if set(metrics) != expected_layers:
            raise RuntimeError(
                f"{kind} reconstruction metrics are incomplete: "
                f"missing={sorted(expected_layers - set(metrics))}"
            )
        failed = [name for name, values in metrics.items() if values.get("passed") is not True]
        if failed:
            raise RuntimeError(f"{kind} layers failed reconstruction: {failed}")
    print(
        f"VALID multimodal artifacts: ViT={vision_count} Audio={audio_count} "
        "strict-GPTQ-v2 weights; bounded smooth; GELU smooth=off; recon=PASS; RTN=0",
        flush=True,
    )
    return replacements, int8_weights, vision_count, audio_count


def merge_multimodal_artifacts(
    base_quant_dir: str,
    artifact_dir: str,
    output_dir: str,
    *,
    require_vision: bool = True,
    require_audio: bool = True,
    link_unchanged_shards: bool = True,
) -> Path:
    """
    Create a full checkpoint without quantizing any uncovered tensor.

    ``base_quant_dir`` already contains the calibrated language/MTP checkpoint.
    Only tensors explicitly produced by the modality JointFix run are replaced;
    there is intentionally no RTN/uncovered-weight code path in this function.
    """
    base = Path(base_quant_dir)
    artifacts = Path(artifact_dir)
    output = Path(output_dir)
    if not (base / "model.safetensors.index.json").is_file():
        raise FileNotFoundError(f"invalid base quantized checkpoint: {base}")
    if output.exists():
        raise FileExistsError(f"output must not exist: {output}")
    output.mkdir(parents=True)

    replacements, int8_weights, vision_count, audio_count = validate_multimodal_artifacts(
        str(artifacts), require_vision=require_vision, require_audio=require_audio,
        model_dir=str(base),
    )

    index = json.loads((base / "model.safetensors.index.json").read_text())
    old_map = index["weight_map"]
    new_map = dict(old_map)
    by_shard = defaultdict(dict)
    for name, tensor in replacements.items():
        lookup = name
        if name.endswith(".weight_scale"):
            lookup = name.removesuffix(".weight_scale") + ".weight"
        if lookup not in old_map:
            raise KeyError(f"artifact tensor has no base checkpoint weight: {name}")
        shard = old_map[lookup]
        by_shard[shard][name] = tensor
        new_map[name] = shard

    shards = sorted(set(old_map.values()))
    for index_in_run, shard in enumerate(shards, 1):
        source = base / shard
        target = output / shard
        if shard not in by_shard:
            _copy_or_link(source, target, link_unchanged_shards)
        else:
            tensors = load_file(str(source))
            tensors.update(by_shard[shard])
            save_file(tensors, str(target))
        print(f"  [merge {index_in_run}/{len(shards)}] {shard}", flush=True)

    (output / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": index.get("metadata", {}), "weight_map": new_map}, indent=2)
    )

    for item in base.iterdir():
        if item.suffix == ".safetensors" or item.name in {
            "model.safetensors.index.json", "config.json"
        }:
            continue
        target = output / item.name
        if item.is_file():
            shutil.copy2(item, target)
        elif item.is_dir():
            shutil.copytree(item, target)

    config = json.loads((base / "config.json").read_text())
    quant_config = config.get("quantization_config")
    if not quant_config:
        raise RuntimeError("base checkpoint has no quantization_config")
    quantized_bases = {name.removesuffix(".weight") for name in int8_weights}
    quant_config["ignore"] = [
        name for name in quant_config.get("ignore", []) if name not in quantized_bases
    ]
    # These non-backbone Linear modules intentionally remain BF16; their presence
    # also proves the inherited ignore list was not over-pruned.
    required_bf16 = {
        "visual.merger.mlp.0",
        "visual.merger.mlp.2",
        "visual.vision_projection.fc1",
    }
    if require_audio:
        required_bf16.add("audio_tower.proj")
    if not required_bf16.issubset(set(quant_config["ignore"])):
        raise RuntimeError(
            "required BF16 non-backbone modules disappeared from qconfig.ignore: "
            f"{sorted(required_bf16 - set(quant_config['ignore']))}"
        )
    _validate_multimodal_serving_config(config, quantized_bases)
    (output / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n"
    )

    metadata = {
        "base_quantized_checkpoint": str(base.resolve()),
        "multimodal_artifacts": str(artifacts.resolve()),
        "algorithm": "jointfix_strict_block_gptq_v2",
        "rtn_used_for_multimodal": False,
        "gelu_cross_layer_smoothing": False,
        "per_layer_reconstruction_gate": "passed",
        "vision_int8_weights": vision_count,
        "audio_int8_weights": audio_count,
        "unchanged_shards_hardlinked": bool(link_unchanged_shards),
    }
    (output / "multimodal_merge_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n"
    )
    print(
        f"VALID merged checkpoint: ViT={vision_count} Audio={audio_count} "
        "all modality INT8 weights are calibrated strict-GPTQ; serving config valid",
        flush=True,
    )
    return output
