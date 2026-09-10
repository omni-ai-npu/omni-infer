# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Sequential error-propagating JointFix runner for Omni ViT and Audio Tower."""
from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict

import torch
from safetensors import safe_open
from safetensors.torch import load_file

from jointfix.core.checkpoint import atomic_save
from jointfix.core.devices import clear_device_cache, resolve_devices, set_npu_compile_mode
from jointfix.core.stats import StatsConfig
from jointfix.methods.jointfix import JointSearchConfig
from jointfix.multimodal.data import load_audio_calibration, load_vision_calibration
from jointfix.multimodal.models import (
    AudioLayer,
    AudioState,
    VisionBlock,
    VisionState,
    install_collectors,
    prepare_audio_state,
    prepare_vision_state,
)
from jointfix.multimodal.quant import (
    quantize_dense_transformer_layer,
)


class CheckpointReader:
    def __init__(self, model_dir: str):
        self.root = Path(model_dir)
        index = self.root / "model.safetensors.index.json"
        if not index.is_file():
            raise FileNotFoundError(f"missing sharded checkpoint index: {index}")
        self.weight_map = json.loads(index.read_text())["weight_map"]

    def load_prefix(self, prefix: str) -> Dict[str, torch.Tensor]:
        by_shard = defaultdict(list)
        for name, shard in self.weight_map.items():
            if name.startswith(prefix):
                by_shard[shard].append(name)
        result = {}
        for shard, names in by_shard.items():
            with safe_open(str(self.root / shard), framework="pt") as f:
                for name in names:
                    result[name] = f.get_tensor(name)
        if not result:
            raise KeyError(f"checkpoint contains no tensors under {prefix!r}")
        return result

    def load_exact_prefixes(self, prefixes) -> Dict[str, torch.Tensor]:
        result = {}
        for prefix in prefixes:
            result.update(self.load_prefix(prefix))
        return result


def _dequantized(tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    result = {}
    for name, value in tensors.items():
        if name.endswith(".weight_scale"):
            continue
        scale_name = name.replace(".weight", ".weight_scale")
        if value.dtype == torch.int8:
            if scale_name not in tensors:
                raise RuntimeError(f"INT8 weight has no scale: {name}")
            result[name] = (value.float() * tensors[scale_name].float()).to(torch.bfloat16)
        else:
            result[name] = value
    return result


def _remove_hooks(handles):
    for handle in handles:
        handle.remove()


def _write_traces(path: Path, traces: dict):
    serializable = {
        name: {key: value for key, value in trace.items() if not torch.is_tensor(value)}
        for name, trace in traces.items()
    }
    path.write_text(json.dumps(serializable, indent=2, ensure_ascii=False))


def _read_traces(path: Path):
    return json.loads(path.read_text()) if path.is_file() else {}


def _write_json(path: Path, value: dict):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False))


def reconstruction_metrics(reference: torch.Tensor, candidate: torch.Tensor,
                           layer_input: torch.Tensor) -> dict:
    """Return full-layer and residual-update reconstruction diagnostics."""
    if reference.shape != candidate.shape or reference.shape != layer_input.shape:
        raise ValueError(
            "reconstruction tensors must have identical shapes: "
            f"reference={reference.shape}, candidate={candidate.shape}, "
            f"input={layer_input.shape}"
        )

    def _pair(ref: torch.Tensor, got: torch.Tensor, prefix: str) -> dict:
        # Per-token cosine avoids a numerically unstable float32 reduction over
        # ~100M elements (which can otherwise report impossible values > 1).
        ref = ref.float().reshape(-1, ref.shape[-1])
        got = got.float().reshape(-1, got.shape[-1])
        diff = got - ref
        ref_rms = ref.square().mean().sqrt().clamp(min=1e-12)
        rel_rmse = diff.square().mean().sqrt() / ref_rms
        cosine = torch.nn.functional.cosine_similarity(
            ref, got, dim=1, eps=1e-12
        ).clamp(min=-1.0, max=1.0).mean()
        return {
            f"{prefix}_rel_rmse": float(rel_rmse.item()),
            f"{prefix}_cosine": float(cosine.item()),
            f"{prefix}_reference_rms": float(ref_rms.item()),
            f"{prefix}_error_absmax": float(diff.abs().max().item()),
        }

    result = _pair(reference, candidate, "output")
    result.update(_pair(reference.float() - layer_input.float(),
                        candidate.float() - layer_input.float(), "update"))
    return result


def enforce_reconstruction_gate(
    metrics: dict,
    *,
    tag: str,
    max_output_rel_rmse: float,
    min_output_cosine: float,
    max_update_rel_rmse: float,
    min_update_cosine: float,
) -> dict:
    """Fail a layer before saving when calibrated INT8 output drifts too far."""
    checks = {
        "output_rel_rmse": metrics["output_rel_rmse"] <= max_output_rel_rmse,
        "output_cosine": metrics["output_cosine"] >= min_output_cosine,
        "update_rel_rmse": metrics["update_rel_rmse"] <= max_update_rel_rmse,
        "update_cosine": metrics["update_cosine"] >= min_update_cosine,
    }
    passed = all(checks.values())
    result = {
        **metrics,
        "passed": passed,
        "thresholds": {
            "max_output_rel_rmse": max_output_rel_rmse,
            "min_output_cosine": min_output_cosine,
            "max_update_rel_rmse": max_update_rel_rmse,
            "min_update_cosine": min_update_cosine,
        },
        "checks": checks,
    }
    print(
        f"[RECON-GATE] {tag} pass={passed} "
        f"output_rel_rmse={metrics['output_rel_rmse']:.4e} "
        f"output_cos={metrics['output_cosine']:.7f} "
        f"update_rel_rmse={metrics['update_rel_rmse']:.4e} "
        f"update_cos={metrics['update_cosine']:.7f}",
        flush=True,
    )
    if not passed:
        failed = [name for name, ok in checks.items() if not ok]
        raise RuntimeError(
            f"{tag}: layer reconstruction gate failed: {failed}; metrics={result}"
        )
    return result


def _gate_layer(reference: torch.Tensor, candidate: torch.Tensor,
                layer_input: torch.Tensor, *, tag: str, gate_config: dict) -> dict:
    return enforce_reconstruction_gate(
        reconstruction_metrics(reference, candidate, layer_input),
        tag=tag,
        **gate_config,
    )


def _segment_batches(segments, token_budget):
    start = 0
    batch_start = 0
    batch_segments = []
    batch_tokens = 0
    for length in segments:
        if batch_segments and batch_tokens + length > token_budget:
            yield batch_start, start, batch_segments
            batch_start = start
            batch_segments = []
            batch_tokens = 0
        batch_segments.append(length)
        batch_tokens += length
        start += length
    if batch_segments:
        yield batch_start, start, batch_segments


def _forward_vision(layer, state: VisionState, device, token_budget):
    outputs = []
    for start, end, segments in _segment_batches(state.segments, token_budget):
        outputs.append(layer(
            state.hidden[start:end].to(device),
            cos=state.cos[start:end], sin=state.sin[start:end], segments=segments,
        ).cpu())
    return torch.cat(outputs, dim=0)


def _forward_audio(layer, state: AudioState, device, token_budget):
    outputs = []
    for start, end, segments in _segment_batches(state.segments, token_budget):
        outputs.append(layer(
            state.hidden[start:end].to(device), segments=segments,
        ).cpu())
    return torch.cat(outputs, dim=0)


def _quantize_vision(reader: CheckpointReader, model_config: dict, manifest: str,
                     output: Path, device, method_config: JointSearchConfig,
                     stats_config: StatsConfig, *, n_samples: int,
                     min_pixels: int, max_pixels: int, image_use_fast: bool,
                     forward_token_budget: int, smooth_scale_min: float,
                     smooth_scale_max: float, gate_config: dict):
    vision_config = model_config["vision_config"]
    pixels, grids = load_vision_calibration(
        str(reader.root), manifest, n_samples=n_samples,
        min_pixels=min_pixels, max_pixels=max_pixels,
        image_use_fast=image_use_fast,
    )
    frontend = reader.load_exact_prefixes(("visual.patch_embed.", "visual.layernorm_pre."))
    state = prepare_vision_state(pixels, grids, frontend, vision_config, device)
    trace_path = output / "vision_joint_search_traces.json"
    traces = _read_traces(trace_path)
    metrics_path = output / "vision_reconstruction_metrics.json"
    reconstruction = _read_traces(metrics_path)
    print(
        f"[vision] initial hidden={tuple(state.hidden.shape)} "
        f"segments={len(state.segments)} tokens={sum(state.segments)}",
        flush=True,
    )

    depth = int(vision_config["depth"])
    for layer_idx in range(depth):
        prefix = f"visual.blocks.{layer_idx}"
        artifact = output / f"visual_layer_{layer_idx:04d}.safetensors"
        if artifact.is_file():
            weights = reader.load_prefix(prefix + ".")
            layer = VisionBlock(weights, prefix, vision_config, device).eval()
            with torch.no_grad():
                reference = _forward_vision(layer, state, device, forward_token_budget)
            del layer
            quantized = load_file(str(artifact))
            layer = VisionBlock(_dequantized(quantized), prefix, vision_config, device).eval()
            with torch.no_grad():
                hidden = _forward_vision(layer, state, device, forward_token_budget)
            reconstruction[prefix] = _gate_layer(
                reference, hidden, state.hidden, tag=prefix, gate_config=gate_config,
            )
            _write_json(metrics_path, reconstruction)
            state = VisionState(hidden, state.cos, state.sin, state.segments)
            del layer, hidden, reference, weights, quantized
            clear_device_cache(device)
            print(
                f"  [vision layer {layer_idx}] resume artifact + reconstruction PASS",
                flush=True,
            )
            continue
        weights = reader.load_prefix(prefix + ".")
        layer = VisionBlock(weights, prefix, vision_config, device).eval()
        collectors, handles = install_collectors(layer.collector_modules(), stats_config)
        with torch.no_grad():
            reference = _forward_vision(layer, state, device, forward_token_budget)
        _remove_hooks(handles)
        quantized, layer_traces = quantize_dense_transformer_layer(
            weights, collectors, prefix=prefix, kind="vision",
            config=method_config, device=device,
            smooth_scale_min=smooth_scale_min,
            smooth_scale_max=smooth_scale_max,
        )

        del layer, collectors
        layer = VisionBlock(_dequantized(quantized), prefix, vision_config, device).eval()
        with torch.no_grad():
            hidden = _forward_vision(layer, state, device, forward_token_budget)
        reconstruction[prefix] = _gate_layer(
            reference, hidden, state.hidden, tag=prefix, gate_config=gate_config,
        )
        # Never persist a layer which fails reconstruction.
        atomic_save(quantized, artifact)
        traces.update(layer_traces)
        _write_traces(trace_path, traces)
        _write_json(metrics_path, reconstruction)
        state = VisionState(hidden.cpu(), state.cos, state.sin, state.segments)
        del layer, hidden, reference, weights, quantized
        clear_device_cache(device)
        print(f"  [vision layer {layer_idx}] JointFix+strict-GPTQ complete", flush=True)

    _write_traces(trace_path, traces)
    print(f"[vision] complete: {depth * 4} strict-GPTQ backbone weights", flush=True)


def _quantize_audio(reader: CheckpointReader, model_config: dict, manifest: str,
                    output: Path, device, method_config: JointSearchConfig,
                    stats_config: StatsConfig, *, n_samples: int,
                    forward_token_budget: int, smooth_scale_min: float,
                    smooth_scale_max: float, gate_config: dict):
    audio_config = model_config["audio_config"]
    features, feature_lens = load_audio_calibration(
        str(reader.root), manifest, n_samples=n_samples,
    )
    frontend = reader.load_exact_prefixes(("audio_tower.conv1.", "audio_tower.conv2."))
    state = prepare_audio_state(features, feature_lens, frontend, audio_config, device)
    trace_path = output / "audio_joint_search_traces.json"
    traces = _read_traces(trace_path)
    metrics_path = output / "audio_reconstruction_metrics.json"
    reconstruction = _read_traces(metrics_path)
    print(
        f"[audio] initial hidden={tuple(state.hidden.shape)} "
        f"segments={len(state.segments)} tokens={sum(state.segments)}",
        flush=True,
    )

    depth = int(audio_config["encoder_layers"])
    for layer_idx in range(depth):
        prefix = f"audio_tower.layers.{layer_idx}"
        artifact = output / f"audio_layer_{layer_idx:04d}.safetensors"
        if artifact.is_file():
            weights = reader.load_prefix(prefix + ".")
            layer = AudioLayer(weights, prefix, audio_config, device).eval()
            with torch.no_grad():
                reference = _forward_audio(layer, state, device, forward_token_budget)
            del layer
            quantized = load_file(str(artifact))
            layer = AudioLayer(_dequantized(quantized), prefix, audio_config, device).eval()
            with torch.no_grad():
                hidden = _forward_audio(layer, state, device, forward_token_budget)
            reconstruction[prefix] = _gate_layer(
                reference, hidden, state.hidden, tag=prefix, gate_config=gate_config,
            )
            _write_json(metrics_path, reconstruction)
            state = AudioState(hidden, state.segments)
            del layer, hidden, reference, weights, quantized
            clear_device_cache(device)
            print(
                f"  [audio layer {layer_idx}] resume artifact + reconstruction PASS",
                flush=True,
            )
            continue
        weights = reader.load_prefix(prefix + ".")
        layer = AudioLayer(weights, prefix, audio_config, device).eval()
        collectors, handles = install_collectors(layer.collector_modules(), stats_config)
        with torch.no_grad():
            reference = _forward_audio(layer, state, device, forward_token_budget)
        _remove_hooks(handles)
        quantized, layer_traces = quantize_dense_transformer_layer(
            weights, collectors, prefix=prefix, kind="audio",
            config=method_config, device=device,
            smooth_scale_min=smooth_scale_min,
            smooth_scale_max=smooth_scale_max,
        )

        del layer, collectors
        layer = AudioLayer(_dequantized(quantized), prefix, audio_config, device).eval()
        with torch.no_grad():
            hidden = _forward_audio(layer, state, device, forward_token_budget)
        reconstruction[prefix] = _gate_layer(
            reference, hidden, state.hidden, tag=prefix, gate_config=gate_config,
        )
        atomic_save(quantized, artifact)
        traces.update(layer_traces)
        _write_traces(trace_path, traces)
        _write_json(metrics_path, reconstruction)
        state = AudioState(hidden.cpu(), state.segments)
        del layer, hidden, reference, weights, quantized
        clear_device_cache(device)
        print(f"  [audio layer {layer_idx}] JointFix+strict-GPTQ complete", flush=True)

    _write_traces(trace_path, traces)
    print(f"[audio] complete: {depth * 6} strict-GPTQ weights", flush=True)


def quantize_omni_multimodal(
    *,
    model_dir: str,
    output_dir: str,
    vision_manifest: str | None,
    audio_manifest: str | None,
    modality: str,
    n_vision_samples: int,
    n_audio_samples: int,
    min_pixels: int,
    max_pixels: int,
    image_use_fast: bool,
    device_name: str,
    sample_rows: int,
    forward_token_budget: int,
    gptq_block_size: int,
    gptq_damp: float,
    smooth_scale_min: float,
    smooth_scale_max: float,
    max_output_rel_rmse: float,
    min_output_cosine: float,
    max_update_rel_rmse: float,
    min_update_cosine: float,
) -> None:
    devices = resolve_devices(device_name, 1)
    device = devices[0]
    if device.type == "npu":
        set_npu_compile_mode()

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    model_config = json.loads((Path(model_dir) / "config.json").read_text())
    reader = CheckpointReader(model_dir)
    method_config = JointSearchConfig(
        objective="output-recon",
        write_quant="gptq",
        gptq_block_size=gptq_block_size,
        gptq_damp=gptq_damp,
        batched_ab_enabled=False,
        batched_ab_distributed_enabled=False,
        num_iterations=1,
    )
    stats_config = StatsConfig(
        sample_limit=sample_rows,
        tokens_per_sample=sample_rows,
        collect_histogram=False,
        collect_moments=False,
    )
    forward_token_budget = int(forward_token_budget)
    if forward_token_budget <= 0:
        raise ValueError("forward_token_budget must be positive")
    if sample_rows < gptq_block_size:
        raise ValueError(
            f"sample_rows={sample_rows} must be >= gptq_block_size={gptq_block_size}; "
            "RTN fallback is disabled"
        )
    if not (0.0 < smooth_scale_min <= 1.0 <= smooth_scale_max):
        raise ValueError(
            "smooth scale bounds must satisfy 0 < min <= 1 <= max; "
            f"got {smooth_scale_min}, {smooth_scale_max}"
        )
    if not (0.0 < max_output_rel_rmse and 0.0 < max_update_rel_rmse):
        raise ValueError("reconstruction RMSE thresholds must be positive")
    if not (-1.0 <= min_output_cosine <= 1.0
            and -1.0 <= min_update_cosine <= 1.0):
        raise ValueError("reconstruction cosine thresholds must be in [-1, 1]")
    gate_config = {
        "max_output_rel_rmse": max_output_rel_rmse,
        "min_output_cosine": min_output_cosine,
        "max_update_rel_rmse": max_update_rel_rmse,
        "min_update_cosine": min_update_cosine,
    }

    if modality in ("vision", "both"):
        if not vision_manifest:
            raise ValueError("--vision-manifest is required for vision calibration")
        _quantize_vision(
            reader, model_config, vision_manifest, output, device,
            method_config, stats_config, n_samples=n_vision_samples,
            min_pixels=min_pixels, max_pixels=max_pixels,
            image_use_fast=image_use_fast,
            forward_token_budget=forward_token_budget,
            smooth_scale_min=smooth_scale_min,
            smooth_scale_max=smooth_scale_max,
            gate_config=gate_config,
        )
    if modality in ("audio", "both"):
        if not audio_manifest:
            raise ValueError("--audio-manifest is required for audio calibration")
        _quantize_audio(
            reader, model_config, audio_manifest, output, device,
            method_config, stats_config, n_samples=n_audio_samples,
            forward_token_budget=forward_token_budget,
            smooth_scale_min=smooth_scale_min,
            smooth_scale_max=smooth_scale_max,
            gate_config=gate_config,
        )

    metadata_path = output / "multimodal_quant_metadata.json"
    metadata = _read_traces(metadata_path)
    completed_modalities = set(metadata.get("completed_modalities", []))
    completed_modalities.add(modality)
    if modality == "both":
        completed_modalities.update(("vision", "audio"))
        completed_modalities.discard("both")
    metadata = {
        "model": str(Path(model_dir).resolve()),
        "completed_modalities": sorted(completed_modalities),
        "algorithm": "jointfix_strict_block_gptq_v2",
        "rtn_fallback": False,
        "gelu_cross_layer_smoothing": False,
        "smooth_scale_min": smooth_scale_min,
        "smooth_scale_max": smooth_scale_max,
        "reconstruction_gate": gate_config,
        "sample_rows": sample_rows,
        "forward_token_budget": forward_token_budget,
        "gptq_block_size": gptq_block_size,
        "gptq_damp": gptq_damp,
        "vision_manifest": vision_manifest or metadata.get("vision_manifest"),
        "audio_manifest": audio_manifest or metadata.get("audio_manifest"),
    }
    _write_json(metadata_path, metadata)
