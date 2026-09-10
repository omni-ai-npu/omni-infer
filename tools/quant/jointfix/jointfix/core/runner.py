# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""
The calibration + quantization loop — model-agnostic, method-agnostic.

ONE path: the forward machinery (`_forward_collect`) handles N devices; N=1 is
the degenerate single-device case. It is NOT a separate forked path. Per-layer
(error-propagating, 2 forwards):

  1. build with ORIGINAL weights, install stat hooks, forward calib -> collectors
  2. method.process_layer(weights, collectors) -> quantized tensors
  3. backend.save_quantized + checkpoint
  4. re-build with the QUANTIZED (dequant-to-bf16) weights, forward calib ->
     quant_hs, which becomes the NEXT layer's input (error-propagating calibration)
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import torch

from jointfix.backends.base import ModelBackend
from jointfix.core.calib_data import load_calibration_data
from jointfix.core.checkpoint import load_checkpoint, save_checkpoint
from jointfix.core import profiling as _prof
from jointfix.core.devices import clear_device_cache, resolve_devices, set_npu_compile_mode
from jointfix.core.stats import merge_collectors
from jointfix.core.modality import CalibrationInputs
from jointfix.methods.base import QuantMethod


@dataclass
class RunConfig:
    model_dir: str
    output_dir: str
    calib_data: str
    n_samples: int = 128
    seq_len: int = 2048
    device: str = "auto"
    num_devices: int = 1
    start_layer: int = 0
    end_layer: "int | None" = None
    calib_format: str = "text"
    mm_min_pixels: int = 50176
    mm_max_pixels: int = 401408
    mm_image_use_fast: bool = True
    mm_forward_token_budget: int = 4096


def _split_chunks(n: int, n_dev: int):
    """
    Split n samples across n_dev devices (earlier devices get the remainder),
    so multi-device stats are order-faithful.
    """
    base, rem = divmod(n, n_dev)
    chunks, off = [], 0
    for d in range(n_dev):
        sz = base + (1 if d < rem else 0)
        chunks.append((off, off + sz))
        off += sz
    return chunks


def _forward_collect(backend, spec, weights, hidden, devices, stats_config, collect,
                     token_is_text=None):
    """
    Forward the calib through one layer across N devices. N=1 = sequential.

    Builds a layer copy per device (sequential — NOT threaded; ThreadPool build is
    NET NEGATIVE on MoE), splits samples, forwards per-device in parallel, then
    merges the per-device collectors. Returns (new_hidden_cpu, merged_collectors).
    """
    is_ragged = isinstance(hidden, (list, tuple))
    n = len(hidden) if is_ragged else hidden.shape[0]
    n_dev = len(devices)
    chunks = _split_chunks(n, n_dev)
    layers = [backend.build_layer(spec, weights, dev) for dev in devices]

    parts = [None] * n_dev
    collectors_parts, handles = [], []
    if collect:
        for layer in layers:
            c: dict = {}
            handles.append(backend.install_stat_hooks(layer, spec.layer_idx, c, stats_config))
            collectors_parts.append(c)

    def run(d):
        s, e = chunks[d]
        if s == e:
            return
        layer, dev = layers[d], devices[d]
        outs = []
        if is_ragged:
            with torch.no_grad():
                for i in range(e - s):
                    sample = hidden[s + i].to(dev)
                    if sample.dim() == 2:
                        sample = sample.unsqueeze(0)
                    mask = (token_is_text[s + i].to(dev)
                            if token_is_text is not None else None)
                    if mask is not None:
                        backend.set_calibration_context(layer, mask)
                    outs.append(layer(sample).detach().cpu())
            parts[d] = outs
        else:
            # Preserve the original dense-text fast path: one H2D copy, keep
            # per-sample results on device, then perform one D2H copy.
            hc = hidden[s:e].to(dev)
            with torch.no_grad():
                for i in range(e - s):
                    mask = (token_is_text[s + i].to(dev)
                            if token_is_text is not None else None)
                    if mask is not None:
                        backend.set_calibration_context(layer, mask)
                    outs.append(layer(hc[i:i + 1]).detach())
            parts[d] = torch.cat(outs, dim=0).cpu()

    if n_dev == 1:
        run(0)
    else:
        with ThreadPoolExecutor(max_workers=n_dev) as ex:
            list(ex.map(run, range(n_dev)))

    for h in handles:
        for hh in h:
            hh.remove()

    # free per-device layer copies before returning (16 layers of a 92B otherwise pile up)
    del layers
    for dev in devices:
        clear_device_cache(dev)

    if is_ragged:
        new_hidden = []
        for part in parts:
            if part is not None:
                new_hidden.extend(part)
    else:
        non_empty = [p for p in parts if p is not None and p.shape[0] > 0]
        new_hidden = (torch.cat(non_empty, dim=0) if non_empty
                      else torch.zeros(0, *hidden.shape[1:]))
    merged = merge_collectors(collectors_parts) if collect else {}
    return new_hidden, merged


def _dequantize_layer(weights: dict, out: dict) -> dict:
    """Reconstruct bf16 weights from a quantized layer for the re-forward."""
    quant_raw = dict(weights)
    for name, tensor in out.items():
        scale_name = name.replace(".weight", ".weight_scale")
        if tensor.dtype == torch.int8 and scale_name in out:
            quant_raw[name] = (tensor.float() * out[scale_name].float()).to(torch.bfloat16)
        elif name.endswith(".weight_scale"):
            continue
        else:
            quant_raw[name] = tensor.to(torch.bfloat16) if tensor.is_floating_point() else tensor
    return quant_raw


def calibrate_and_quantize(backend: ModelBackend, method: QuantMethod,
                           cfg: RunConfig, input_ids=None,
                           initial_hidden=None) -> None:
    devices = resolve_devices(cfg.device, cfg.num_devices)
    device = devices[0]
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    specs = backend.layer_specs()
    end = cfg.end_layer if cfg.end_layer is not None else len(specs)

    start, written = load_checkpoint(out_dir, backend.weight_map())
    start = max(start, cfg.start_layer)
    if start > 0:
        raise NotImplementedError(
            "resume>0 needs fast-forwarding the calib through already-quantized "
            "layers — not ported in v1 (run from layer 0).")

    original_text_path = initial_hidden is None and cfg.calib_format == "text"
    if original_text_path:
        # Keep the historical language-only JointFix order exactly: tokenizer
        # loading may re-import torch_npu, so compile mode must be re-asserted
        # afterwards and immediately before embedding/the layer forward loop.
        if input_ids is None:
            input_ids = load_calibration_data(cfg.calib_data, cfg.model_dir,
                                              cfg.n_samples, cfg.seq_len)
        n = input_ids.shape[0]
        if device.type == "npu":
            set_npu_compile_mode()
        _prof.heartbeat(f"embedding {n} calib samples (seq {input_ids.shape[1]})…")
        _t_embed = _prof.clock()
        hidden = backend.embed(input_ids, device)    # [n, seq, hidden] float32
        token_is_text = None
        source = "text"
    else:
        # Multimodal preparation itself runs model components on the target
        # device. Re-assert once before preparation and again after any
        # trust_remote_code imports, just before the decoder loop.
        if device.type == "npu":
            set_npu_compile_mode()
        _t_embed = _prof.clock()

    if not original_text_path and initial_hidden is not None:
        hidden = initial_hidden
        source = "precomputed"
    elif not original_text_path and cfg.calib_format == "omni":
        if input_ids is not None:
            raise ValueError("input_ids cannot be combined with calib_format='omni'")
        from jointfix.multimodal.llm_calib import load_omni_calibration_hidden

        calibration_inputs = load_omni_calibration_hidden(
            model_dir=cfg.model_dir,
            manifest=cfg.calib_data,
            n_samples=cfg.n_samples,
            seq_len=cfg.seq_len,
            backend=backend,
            device=device,
            min_pixels=cfg.mm_min_pixels,
            max_pixels=cfg.mm_max_pixels,
            image_use_fast=cfg.mm_image_use_fast,
            forward_token_budget=cfg.mm_forward_token_budget,
        )
        hidden = calibration_inputs.hidden
        token_is_text = calibration_inputs.token_is_text
        source = "omni"
    if not original_text_path and initial_hidden is not None:
        if isinstance(initial_hidden, CalibrationInputs):
            hidden = initial_hidden.hidden
            token_is_text = initial_hidden.token_is_text
        else:
            token_is_text = None
    if isinstance(hidden, (list, tuple)):
        if not hidden or any(x.dim() not in (2, 3) for x in hidden):
            shapes = [tuple(x.shape) for x in hidden] if hidden else []
            raise ValueError(
                "ragged calibration hidden states must be a non-empty list of "
                f"[seq, hidden] or [1, seq, hidden] tensors, got {shapes}"
            )
        n = len(hidden)
        seq_desc = f"ragged[{min(x.shape[-2] for x in hidden)}..{max(x.shape[-2] for x in hidden)}]"
    else:
        if hidden.dim() != 3:
            raise ValueError(
                f"calibration hidden states must be [samples, seq, hidden], got {tuple(hidden.shape)}"
            )
        n = hidden.shape[0]
        seq_desc = str(hidden.shape[1])
    if not original_text_path:
        if device.type == "npu":
            set_npu_compile_mode()
        _prof.heartbeat(
            f"calibration inputs ready: source={source} samples={n} seq={seq_desc}"
        )
    _prof.heartbeat(f"embed done in {_prof.fmt(_prof.clock() - _t_embed)}")
    stats_config = method.stats_config()

    todo = specs[start:end]
    n_layers = len(todo)
    _run_t0 = time.time()
    for _i, spec in enumerate(todo):
        li = spec.layer_idx
        # NOTE: weight load is the suspected bottleneck on big models (SFS/network IO) and
        # used to run *before* the first clock() — time it explicitly and report it live.
        _t_load = _prof.clock()
        _prof.heartbeat(f"[layer {li}] ({_i + 1}/{n_layers}) loading weights from disk…")
        weights = backend.load_layer_weights(li)
        _t0 = _prof.clock()
        _prof.heartbeat(f"[layer {li}] weights loaded in {_prof.fmt(_t0 - _t_load)} — BF16 forward…")

        # 1. BF16 forward + collect activation stats
        collectors: dict = {}
        if method.needs_activations:
            _, collectors = _forward_collect(backend, spec, weights, hidden,
                                             devices, stats_config, collect=True,
                                             token_is_text=token_is_text)
        _t_fwd = _prof.clock()
        _prof.heartbeat(f"[layer {li}] forward done in {_prof.fmt(_t_fwd - _t0)} — search+quant…")

        # 2. search + quantize (distributed per-expert (a,b) when len(devices)>1)
        out = method.process_layer(weights, collectors, spec, backend, device, devices)
        _t_search = _prof.clock()
        _prof.heartbeat(f"[layer {li}] search+quant done in {_prof.fmt(_t_search - _t_fwd)} — persist + re-forward…")

        # 3. persist + checkpoint
        backend.save_quantized(str(out_dir), li, out, {})
        save_checkpoint(out_dir, li, written)

        # 4. re-forward with quantized weights -> propagate quant_hs
        quant_raw = _dequantize_layer(weights, out)
        _t_rf = _prof.clock()
        hidden, _ = _forward_collect(backend, spec, quant_raw, hidden,
                                     devices, stats_config, collect=False,
                                     token_is_text=token_is_text)
        _prof.emit_layer(li, n, t_forward=_t_fwd - _t0,
                         t_search_quant=_t_search - _t_fwd,
                         t_reforward=_prof.clock() - _t_rf,
                         t_load=_t0 - _t_load)
        # Always-on per-layer progress + ETA (the detailed [PROF] heartbeats above are
        # gated on JOINTFIX_PROFILE; this line is not, so a plain run still shows an ETA
        # like the monolith did).
        _done = _i + 1
        _el = time.time() - _run_t0
        _avg = _el / _done
        print(f"  [layer {li}] done {_done}/{n_layers} ({n} samples) | "
              f"elapsed {_prof.fmt(_el)} | avg {_prof.fmt(_avg)}/layer | "
              f"ETA ~{_prof.fmt(_avg * (n_layers - _done))}", flush=True)

    # persist any method-side diagnostics (jointfix -> joint_search_traces.json)
    method.dump_traces(out_dir)
