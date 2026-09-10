import os
import json
import re
import shutil
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from glob import glob
from queue import Empty, Queue
from threading import Event, Lock
from tqdm import tqdm
import torch

try:
    import torch_npu
except:
    pass

from safetensors.torch import load_file, save_file


def parse_devices(device):
    raw_devices = device if isinstance(device, (list, tuple)) else str(device).split(",")
    devices = []
    for raw_device in raw_devices:
        normalized = str(raw_device).strip()
        if not normalized:
            raise ValueError("device list contains an empty device")
        if normalized.isdigit():
            normalized = f"npu:{normalized}"
        devices.append(normalized)

    if len(devices) != len(set(devices)):
        raise ValueError(f"duplicate device in device list: {device}")
    if len(devices) > 1 and any(not item.startswith("npu:") for item in devices):
        raise ValueError("multiple devices must be explicit NPU devices such as 0,1,2,3")
    return devices


def run_file_workers(files, devices, process_file, initialize_worker=None, on_complete=None):
    if not devices:
        raise ValueError("at least one device is required")

    work_queue = Queue()
    for index, file_path in enumerate(files):
        work_queue.put((index, file_path))

    results = [None] * len(files)
    stop_event = Event()
    callback_lock = Lock()

    def worker(device):
        if initialize_worker is not None:
            initialize_worker(device)
        while not stop_event.is_set():
            try:
                index, file_path = work_queue.get_nowait()
            except Empty:
                return
            try:
                results[index] = process_file(file_path, device)
                if on_complete is not None:
                    with callback_lock:
                        on_complete()
            except Exception as error:
                stop_event.set()
                raise RuntimeError(
                    f"failed to process {file_path} on {device}"
                ) from error
            finally:
                work_queue.task_done()

    with ThreadPoolExecutor(
        max_workers=len(devices), thread_name_prefix="int4-quant"
    ) as executor:
        futures = [executor.submit(worker, device) for device in devices]
        for future in futures:
            future.result()

    return results


class QType:
    exp_bits: int = -1
    man_bits: int = -1
    k_bits: int = -1
    k_outer_bits: int = 0
    blk_size: int = -1
    exp_offset: int = 0
    numbits: int = 4
    is_act_integer: bool = False

    def __init__(self, num_step=50, group_size=0, act_integer=False, num_bits=4, symmetric=False):
        self.num_step = num_step
        self.blk_size = group_size
        self.is_act_integer = act_integer
        self.numbits = num_bits
        self.ssz_sym = symmetric
        self.q_dim = -1

    @classmethod
    def from_args(cls, args):
        if hasattr(args, "num_step"):
            return cls(args.num_step, args.group_size, args.act_integer, args.num_bits, args.symmetric)

        match = re.fullmatch(r"sszs(\d+)g(\d+)a(\d+)b(\d+)sym(\d+)", args.qtype)
        if match is None:
            raise ValueError(f"invalid qtype: {args.qtype}")
        num_step, group_size, act_integer, num_bits, symmetric = map(int, match.groups())
        return cls(num_step, group_size, bool(act_integer), num_bits, bool(symmetric))

    @property
    def desc(self):
        return (
            f"sszs{self.num_step}g{self.blk_size}a{int(self.is_act_integer)}"
            f"b{self.numbits}sym{int(self.ssz_sym)}"
        )

    def dim_(self, dim: int):
        self.q_dim = dim
        return self

    def dim(self, dim: int):
        out = deepcopy(self)
        out.q_dim = dim
        return out

    def copy(self):
        return deepcopy(self)

    def __repr__(self) -> str:
        return str(f'QType: {self.desc}   Dim: {self.q_dim}   ExpOffset: {self.exp_offset}')


def get_qbits_minmax(numbits, is_sym):
    BIT_MAX = 2 ** (numbits - 1) - 1
    BIT_MIN = -BIT_MAX if is_sym else -BIT_MAX - 1
    return BIT_MIN, BIT_MAX


def get_scale_offset(x, qW_min, qW_max, is_sym, is_act_integer, clip_ratio=1):
    scale = None
    offset = None
    if is_sym:
        # Symmetric quantization: s = max(|x|) / q_max, with zero-point z = 0.
        xmax = torch.abs(x).max(dim=-1, keepdim=True)[0]
        if is_act_integer:
            # Integer activation scale: s = max(round(s), 1).
            scale = torch.round(xmax / qW_max).clamp(min=1)
        else:
            scale = (xmax / qW_max).clamp(min=1e-5)
    else:
        # Asymmetric quantization maps [x_min, x_max] to [q_min, q_max].
        xmax = x.max(dim=-1, keepdim=True)[0] * clip_ratio
        xmin = x.min(dim=-1, keepdim=True)[0] * clip_ratio
        # Expand near-constant groups to a valid range containing zero.
        compare = ((xmax - xmin) < 1e-5).to(torch.int32)
        xmax = xmax * (1 - compare) + torch.max(torch.abs(xmax), torch.abs(xmin)) * compare
        xmin = xmin * (1 - compare)
        # Scale: s = (x_max - x_min) / (q_max - q_min).
        scale = (xmax - xmin).clamp(min=1e-5) / (qW_max - qW_min)
        if is_act_integer:
            scale = torch.round(scale).clamp(min=1)
        else:
            scale = scale.clamp(min=1e-5)
        # Zero-point: z = round(-x_min / s) + q_min.
        offset = torch.round(-xmin / scale) + qW_min
    return scale, offset


def get_quant(x, qW_min, qW_max, scale, offset=None):
    if offset is not None:
        # Asymmetric quantization: q = clip(round(x / s + z), q_min, q_max).
        return torch.round(x / scale + offset).clamp(min=qW_min, max=qW_max)
    else:
        # Symmetric quantization: q = clip(round(x / s), q_min, q_max).
        return torch.round(x / scale).clamp(min=qW_min, max=qW_max)


def get_dequant(x_quant, qW_min, qW_max, scale, offset=None):
    if offset is not None:
        # Asymmetric dequantization: x_hat = (q - z) * s.
        return (x_quant - offset) * scale
    else:
        # Symmetric dequantization: x_hat = q * s.
        return x_quant * scale


L_mode = 2


def get_mseloss(x, qW_min, qW_max, scale=None, offset=None, quant=None, dequant=None, mode=L_mode):
    if quant is None and dequant is None:
        quant = get_quant(x, qW_min, qW_max, scale, offset)
    if dequant is None:
        dequant = get_dequant(quant, qW_min, qW_max, scale, offset)
    # Quantization loss: L = mean(|x - x_hat|^p), with p = 2 by default.
    return torch.mean(torch.pow(torch.abs(x - dequant), mode), dim=-1, keepdim=True)


loss_function = get_mseloss


def quant_ssz(
    x: torch.Tensor,
    Q: QType,
    qdim: int,
    init_scale=None,
    init_offset=None,
    init_quant=None,
    w8=False,
    w4=False,
    clip_ratio=1.0,
):
    num_step = Q.num_step
    groupsize = Q.blk_size
    is_act_integer = Q.is_act_integer
    numbits = Q.numbits
    is_ssz_sym = Q.ssz_sym
    shape = x.shape
    if groupsize != 0:
        # Group-wise quantization: x -> [N, group_size].
        shaped_x = x.view(-1, groupsize)
    else:
        shaped_x = x
        groupsize = shaped_x.shape[-1]
    # The n-bit integer range uses q_max = 2^(n-1)-1; symmetry determines q_min.
    qW_min, qW_max = get_qbits_minmax(numbits, is_sym=is_ssz_sym)

    if init_offset is not None and init_scale is not None and init_quant is not None:
        scale, offset, quant = init_scale, init_offset, init_quant
    else:
        # Initialize the scale and zero-point, then compute q = Q(x; s, z).
        scale, offset = get_scale_offset(
            shaped_x,
            qW_min,
            qW_max,
            is_sym=is_ssz_sym,
            is_act_integer=is_act_integer,
            clip_ratio=clip_ratio,
        )
        scale = scale.clamp(min=1e-5)
        quant = get_quant(shaped_x, qW_min, qW_max, scale, offset=offset)

    # Initial dequantized value: x_hat = D(q; s, z).
    dequant = get_dequant(quant, qW_min, qW_max, scale, offset=offset)
    bestScale = scale
    if is_ssz_sym:
        offset = 0
    bestOffset = offset
    bestQuant = quant
    bestMse = loss_function(shaped_x, qW_min, qW_max, scale=scale, offset=offset, quant=quant, dequant=dequant)
    for i in range(num_step):
        # Update s by least squares with q and z fixed: s = Σ[(q-z)x] / Σ[(q-z)^2].
        a = quant - offset
        if is_act_integer:
            scale = torch.round(torch.sum(a * shaped_x, dim=-1, keepdim=True) / torch.sum(a * a, dim=-1, keepdim=True)).clamp(min=1)
        else:
            scale = torch.sum(a * shaped_x, dim=-1, keepdim=True) / torch.sum(a * a, dim=-1, keepdim=True).clamp(min=1e-5)
        # Update z with q and s fixed: z = round(mean(q*s-x) / s).
        offset = torch.sum(quant * scale - shaped_x, dim=-1, keepdim=True) / groupsize / scale
        if is_ssz_sym:
            offset= 0
        # Requantize with the updated parameters and evaluate the current loss.
        quant = get_quant(shaped_x, qW_min, qW_max, scale, offset=offset)
        dequant = get_dequant(quant, qW_min, qW_max, scale, offset=offset)
        currentMse = loss_function(shaped_x, qW_min, qW_max, scale=scale, offset=offset, quant=quant, dequant=dequant)
        # Stop early when both relative and absolute loss changes converge.
        mask1 = (bestMse - currentMse) / bestMse.clamp(min=1e-4) < 1e-10
        mask2 = torch.abs(bestMse - currentMse) < 1e-10
        if torch.sum(torch.logical_and(torch.logical_not(mask1), torch.logical_not(mask2))) == 0:
            break
        # Retain the lower-loss parameters per group: best = current if L_current < L_best.
        if is_ssz_sym:
            mask = (currentMse < bestMse).to(torch.int32)
            bestMse = currentMse * mask + bestMse * (1 - mask)
            bestScale = scale * mask + bestScale * (1 - mask)
            bestOffset = offset * mask + bestOffset * (1 - mask)
            bestQuant = quant * mask + bestQuant * (1 - mask)
        else:
            valid = torch.isfinite(scale) & torch.isfinite(offset) & torch.isfinite(currentMse)
            mask = valid & (currentMse < bestMse)
            bestMse = torch.where(mask, currentMse, bestMse)
            bestScale = torch.where(mask, scale, bestScale)
            bestOffset = torch.where(mask, offset, bestOffset)
            bestQuant = torch.where(mask, quant, bestQuant)

    # Reconstruct with the optimal parameters: x_hat = (q_best - z_best) * s_best.
    recovered = get_dequant(bestQuant, qW_min, qW_max, bestScale, bestOffset)
    recovered = recovered.view(shape)
    if w8:
        return bestQuant.to(torch.int8), bestScale

    if w4:
        bestQuantInt4 = bestQuant.view(shape).to(torch.int8)
        # Store the FP32 scale bit pattern in the lower 32 bits of an INT64 value.
        bestScale = bestScale.view(shape[0], -1).T.to(torch.float32).view(torch.int32)
        bestScaleInt64 = torch.zeros(bestScale.shape, dtype=torch.int64).view(torch.int32).reshape(-1, 2)
        bestScaleInt64[:, 0] = bestScale.reshape(-1)
        bestScaleInt64 = bestScaleInt64.view(torch.int64).reshape(bestScale.shape)

        # Compensate for the unsigned INT4 offset: bias = 8 * Σx_hat.
        bias = (8 * recovered.to(torch.float32)).sum(dim=-1)
        # The runtime consumes this as an additive term: (q + offset) * scale.
        # Store -z so it remains equivalent to the quantizer's (q - z) * scale.
        offset = None if is_ssz_sym else bestOffset.view(shape[0], -1).T.to(torch.float32)
        return bestQuantInt4, bestScaleInt64, bias, offset
    return recovered


def weight_quant(tensor: torch.Tensor):
    assert tensor.dim() == 2
    qmax = 127.0
    abs_max = torch.abs(tensor).max(dim=1, keepdim=True)[0]
    scale = abs_max / qmax
    assert scale.shape == (tensor.shape[0], 1)
    quantized = torch.round(tensor / scale)
    quantized = torch.clamp(quantized, -qmax, qmax)
    return quantized.to(torch.int8), scale.to(torch.float32)


def pack_4bit(x):
    x = x.T.contiguous()
    shape = x.shape
    x = x.view(-1, 2)
    x1 = x[:, 0]
    x2 = x[:, 1]
    y_x2 = torch.bitwise_left_shift(x2, 4)
    y_x1 = x1 & 0b00001111
    y = torch.bitwise_or(y_x1, y_x2)
    y = y.view(shape[0], shape[1] // 2)
    return y.T.contiguous()


def copy_model_files(bf16_path, output_path):
    source = os.path.abspath(bf16_path)
    destination = os.path.abspath(output_path)
    if os.path.commonpath([source, destination]) == source:
        raise ValueError("output_path must be outside bf16_path")

    def copy_and_print(src, dst):
        copied_file = shutil.copy2(src, dst)
        print(f"copied: {os.path.relpath(src, source)}", flush=True)
        return copied_file

    print(f"copying non-safetensors files from {bf16_path} to {output_path}", flush=True)
    shutil.copytree(
        source,
        destination,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("*.safetensors"),
        copy_function=copy_and_print,
    )


def main(
    args,
    bf16_path,
    output_path,
    pangu_mode,
    model_name="deepseek-ai/DeepSeek-R1",
    disable_names=None,
    *legacy_args,
):
    # Keep older entry points working while the removed post-process flag is phased out.
    if len(legacy_args) > 1:
        raise TypeError("main() accepts at most one legacy positional argument")
    if disable_names is None:
        disable_names = []
    clip_ratio = getattr(args, "clip_ratio", 1.0)
    quant_prefix = "quant_model_weight_w4a8_dynamic"
    w4_type = QType.from_args(args)
    torch.set_default_dtype(torch.bfloat16)
    copy_model_files(bf16_path, output_path)
    model_index_file = os.path.join(output_path, "model.safetensors.index.json")
    with open(model_index_file, "r") as f:
        model_index = json.load(f)
    weight_map = model_index["weight_map"]
    scale_count = len([key for key in weight_map.keys() if key.endswith("_scale_inv")])

    safetensor_files = list(glob(os.path.join(bf16_path, "*.safetensors")))
    safetensor_files.sort()
    if args.file_count:
        safetensor_files = safetensor_files[:args.file_count]

    def weight_is_w4(weight_name):
        if 'experts' not in weight_name:
            return False
        if 'shared_experts' in weight_name:
            return False
        w4_list = ['up_proj', 'gate_proj', 'down_proj']
        for name in w4_list:
            if name in weight_name:
                return True
        return False

    def initialize_worker(device):
        if device.startswith("npu"):
            torch.npu.set_device(device)

    def process_file(safetensor_file, device):
        file_name = os.path.basename(safetensor_file)
        file_name = file_name.replace("model", quant_prefix)

        state_dict = load_file(safetensor_file, device=device)
        new_state_dict = {}
        file_weight_map = {}
        file_quant_count = 0
        for weight_name, weight in state_dict.items():
            if weight_name in disable_names:
                new_state_dict[weight_name] = weight
                file_weight_map[weight_name] = file_name
                continue
            scale_inv_name = f"{weight_name}_scale_inv"
            if scale_inv_name in weight_map or pangu_mode:
                assert weight.element_size() == 2
                file_quant_count += 1
                if weight_is_w4(weight_name):
                    int4_weight, int4_scale, bias, offset = quant_ssz(
                        weight,
                        w4_type,
                        -1,
                        w4=True,
                        clip_ratio=clip_ratio,
                    )

                    new_state_dict[weight_name] = pack_4bit(int4_weight)
                    new_scale_int4 = scale_inv_name.replace("_scale_inv", "_int4_scale")

                    new_state_dict[new_scale_int4] = int4_scale

                    file_weight_map[weight_name] = file_name
                    file_weight_map[new_scale_int4] = file_name
                    if w4_type.ssz_sym:
                        new_bias = scale_inv_name.replace("_scale_inv", "_bias")
                        new_state_dict[new_bias] = bias
                        file_weight_map[new_bias] = file_name
                    if offset is not None:
                        new_offset = scale_inv_name.replace("_scale_inv", "_offset")
                        new_state_dict[new_offset] = offset
                        file_weight_map[new_offset] = file_name
                else:
                    int8_weight, scale_inv = weight_quant(weight)
                    new_state_dict[weight_name] = int8_weight
                    new_scale_name = scale_inv_name.replace("_scale_inv", "_scale")
                    new_state_dict[new_scale_name] = scale_inv

                    file_weight_map[weight_name] = file_name
                    file_weight_map[new_scale_name] = file_name
            else:
                new_state_dict[weight_name] = weight
                file_weight_map[weight_name] = file_name

        new_safetensor_file = os.path.join(output_path, file_name)
        save_file(new_state_dict, new_safetensor_file)
        del state_dict
        del new_state_dict
        return file_quant_count, file_weight_map

    devices = parse_devices(args.device)
    print(f"quantizing with {len(devices)} worker(s): {', '.join(devices)}")
    with tqdm(total=len(safetensor_files)) as progress:
        file_results = run_file_workers(
            safetensor_files,
            devices,
            process_file,
            initialize_worker=initialize_worker,
            on_complete=progress.update,
        )

    quant_count = sum(file_quant_count for file_quant_count, _ in file_results)
    new_weight_map = {}
    for _, file_weight_map in file_results:
        new_weight_map.update(file_weight_map)
    print(quant_count, scale_count)
    print(f"{quant_count} weights are quantized")

    with open(model_index_file, "r") as f:
        model_index = json.load(f)
    model_index["weight_map"] = new_weight_map
    with open(model_index_file, "w", encoding="utf-8") as f:
        json.dump(model_index, f, indent=2, ensure_ascii=False, sort_keys=True)
    print(f"model.safetensors.index.json modified and saved to {model_index_file}")
