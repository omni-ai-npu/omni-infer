import optiquant.int8 as qint8
import optiquant.int4 as qint4
import optiquant.w4group_to_w4channel as qint4_pergroup_to_perchannel
import optiquant.faquant as faquant
from argparse import ArgumentParser
import json
import os


_FRAMEWORK_IGNORE_SUFFIXES = (
    ".attn_mhc_module.phi",
    ".embed_tokens",
    ".mlp.gate",
    ".mlp.shared_experts.down_proj",
    ".mlp.shared_experts.gate_proj",
    ".mlp.shared_experts.up_proj",
    ".mlp_mhc_module.phi",
    ".self_attn.indexer.weights_proj",
    ".self_attn.indexer.wk",
    ".self_attn.indexer.wq_b",
    ".self_attn.kv_a_proj_with_mqa",
    ".self_attn.kv_b_proj",
    ".self_attn.q_a_proj",
    ".shared_head.head",
)
_FRAMEWORK_IGNORE_GLOBALS = {
    "lm_head",
    "model.embed_tokens",
    "model.merge_mhc_module.phi",
}


def _is_framework_ignore_module(module_name):
    return (
        module_name in _FRAMEWORK_IGNORE_GLOBALS
        or module_name.endswith(_FRAMEWORK_IGNORE_SUFFIXES)
    )


def build_quantization_config(
    output_path,
    disable_names,
    expert_num_bits,
    weight_is_symmetric,
    quantize="w4a8_dynamic",
):
    model_index_path = os.path.join(
        output_path, "model.safetensors.index.json"
    )
    with open(model_index_path, "r") as f:
        weight_map = json.load(f)["weight_map"]

    existing_weights = set(weight_map)
    ignores = []
    for weight_name in set(disable_names):
        if not weight_name.endswith(".weight"):
            continue
        if weight_name not in existing_weights:
            continue
        module_name = weight_name[:-len(".weight")]
        if _is_framework_ignore_module(module_name):
            ignores.append(module_name)
    ignores.sort()

    weight_num_bits = (
        {"mlp.experts": 4, "proj": 8}
        if expert_num_bits == 4
        else 8
    )
    weights = {
        "type": "int",
        "num_bits": weight_num_bits,
        "symmetric": weight_is_symmetric,
        "strategy": "channel",
        "dynamic": False,
        "observer": "minmax",
        "actorder": None,
        "group_size": None,
        "block_structure": None,
        "observer_kwargs": {},
    }
    if not weight_is_symmetric:
        weights["asymmetric_group"] = ["mlp.experts"]

    return {
        "quant_method": "compressed-tensors",
        "quantize": quantize,
        "format": "int-quantized",
        "quantization_status": "compressed",
        "config_groups": {
            "group_0": {
                "format": "int-quantized",
                "targets": ["Linear"],
                "weights": weights,
                "input_activations": {
                    "type": "int",
                    "num_bits": 8,
                    "symmetric": True,
                    "strategy": "token",
                    "dynamic": True,
                    "actorder": None,
                    "group_size": None,
                    "block_structure": None,
                    "observer": None,
                    "observer_kwargs": {},
                },
                "output_activations": None,
            }
        },
        "ignore": ignores,
        "sparsity_config": {},
        "transform_config": {},
        "global_compression_ratio": None,
        "kv_cache_scheme": None,
    }


def init_pangu_flash_disable_names():
    disable_names = []
    for i in range(49):
        disable_names.append(f"model.layers.{i}.self_attn.q_a_proj.weight")
        disable_names.append(f"model.layers.{i}.self_attn.q_a_layernorm.weight")
        disable_names.append(f"model.layers.{i}.self_attn.kv_a_proj_with_mqa.weight")
        disable_names.append(f"model.layers.{i}.self_attn.kv_a_layernorm.weight")
        disable_names.append(f"model.layers.{i}.self_attn.kv_b_proj.weight")
        disable_names.append(f"model.layers.{i}.self_attn.indexer.wq_b.weight")
        disable_names.append(f"model.layers.{i}.self_attn.indexer.wk.weight")
        disable_names.append(f"model.layers.{i}.self_attn.indexer.k_norm.weight")
        disable_names.append(f"model.layers.{i}.self_attn.indexer.weights_proj.weight")
        disable_names.append(f"model.layers.{i}.self_attn.qa_conv.weight")
        disable_names.append(f"model.layers.{i}.self_attn.compresskv_conv.weight")
        disable_names.append(f"model.layers.{i}.self_attn.o_conv.weight")
        disable_names.append(f"model.layers.{i}.self_attn.param_sink_compressed_kv")
        disable_names.append(f"model.layers.{i}.self_attn.param_sink_k_pe")

        disable_names.append(f"model.layers.{i}.mlp.gate.weight")
        disable_names.append(f"model.layers.{i}.mlp.e_score_correction_bias")
        disable_names.append(f"model.layers.{i}.mlp.gate.e_score_correction_bias")
        disable_names.append(f"model.layers.{i}.mlp.shared_experts.gate_proj.weight")
        disable_names.append(f"model.layers.{i}.mlp.shared_experts.up_proj.weight")
        disable_names.append(f"model.layers.{i}.mlp.shared_experts.down_proj.weight")

        disable_names.append(f"model.layers.{i}.input_layernorm.weight")
        disable_names.append(f"model.layers.{i}.pre_mlp_layernorm.weight")
        disable_names.append(f"model.layers.{i}.post_attention_layernorm.weight")
        disable_names.append(f"model.layers.{i}.post_mlp_layernorm.weight")
        disable_names.append(f"model.layers.{i}.q_a_layernorm.weight")
        disable_names.append(f"model.layers.{i}.kv_a_layernorm.weight")
        disable_names.append(f"model.layers.{i}.indexer.k_norm.weight")
        disable_names.append(f"model.layers.{i}.block_post_layernorm.weight")
        disable_names.append(f"model.layers.{i}.enorm.weight")
        disable_names.append(f"model.layers.{i}.hnorm.weight")
        disable_names.append(f"model.layers.{i}.shared_head.norm.weight")
        disable_names.append(f"model.layers.{i}.model.norm.weight")

        disable_names.append(f"model.layers.{i}.self_attn.q_a_layernorm.weight")
        disable_names.append(f"model.layers.{i}.self_attn.kv_a_layernorm.weight")
        disable_names.append(f"model.layers.{i}.attn_mhc_module.phi.weight")
        disable_names.append(f"model.layers.{i}.mlp_mhc_module.phi.weight")
        disable_names.append(f"model.layers.{i}.attn_mhc_module.branch_alpha")
        disable_names.append(f"model.layers.{i}.attn_mhc_module.branch_beta")
        disable_names.append(f"model.layers.{i}.attn_mhc_module.norm_gamma")
        disable_names.append(f"model.layers.{i}.mlp_mhc_module.branch_alpha")
        disable_names.append(f"model.layers.{i}.mlp_mhc_module.branch_beta")
        disable_names.append(f"model.layers.{i}.mlp_mhc_module.norm_gamma")

        if i in (46, 47, 48):
            disable_names.append(f"model.layers.{i}.embed_tokens.weight")
            disable_names.append(f"model.layers.{i}.shared_head.head.weight")

    disable_names.append("lm_head.weight")
    disable_names.append("model.norm.weight")
    disable_names.append("model.embed_tokens.weight")
    disable_names.append("model.merge_mhc_module.phi.weight")
    disable_names.append("model.merge_mhc_module.branch_alpha_pre")
    disable_names.append("model.merge_mhc_module.branch_beta_pre")
    disable_names.append("model.merge_mhc_module.norm_gamma")
    return disable_names


def init_pangu_pro_disable_names():
    disable_names = []
    for i in range(53):
        disable_names.append(f"model.layers.{i}.self_attn.q_a_proj.weight")
        disable_names.append(f"model.layers.{i}.self_attn.q_a_layernorm.weight")
        disable_names.append(f"model.layers.{i}.self_attn.kv_a_proj_with_mqa.weight")
        disable_names.append(f"model.layers.{i}.self_attn.kv_a_layernorm.weight")
        disable_names.append(f"model.layers.{i}.self_attn.kv_b_proj.weight")
        disable_names.append(f"model.layers.{i}.self_attn.compresskv_conv.weight")
        disable_names.append(f"model.layers.{i}.self_attn.qa_conv.weight")
        disable_names.append(f"model.layers.{i}.self_attn.o_conv.weight")
        disable_names.append(f"model.layers.{i}.self_attn.param_sink_compressed_kv")
        disable_names.append(f"model.layers.{i}.self_attn.param_sink_k_pe")
        disable_names.append(f"model.layers.{i}.self_attn.indexer.wq_b.weight")
        disable_names.append(f"model.layers.{i}.self_attn.indexer.wk.weight")
        disable_names.append(f"model.layers.{i}.self_attn.indexer.k_norm.weight")
        disable_names.append(f"model.layers.{i}.self_attn.indexer.weights_proj.weight")

        disable_names.append(f"model.layers.{i}.mlp.gate.weight")
        disable_names.append(f"model.layers.{i}.mlp.shared_experts.gate_proj.weight")
        disable_names.append(f"model.layers.{i}.mlp.shared_experts.up_proj.weight")
        disable_names.append(f"model.layers.{i}.mlp.shared_experts.down_proj.weight")
        disable_names.append(f"model.layers.{i}.mlp.e_score_correction_bias")

        disable_names.append(f"model.layers.{i}.input_layernorm.weight")
        disable_names.append(f"model.layers.{i}.pre_mlp_layernorm.weight")
        disable_names.append(f"model.layers.{i}.post_attention_layernorm.weight")
        disable_names.append(f"model.layers.{i}.post_mlp_layernorm.weight")
        disable_names.append(f"model.layers.{i}.q_a_layernorm.weight")
        disable_names.append(f"model.layers.{i}.kv_a_layernorm.weight")
        disable_names.append(f"model.layers.{i}.self_attn.q_a_layernorm.weight")
        disable_names.append(f"model.layers.{i}.self_attn.kv_a_layernorm.weight")
        disable_names.append(f"model.layers.{i}.block_post_layernorm.weight")
        disable_names.append(f"model.layers.{i}.model.norm.weight")
        disable_names.append(f"model.layers.{i}.indexer.k_norm.weight")
        disable_names.append(f"model.layers.{i}.attn_mhc_module.phi.weight")
        disable_names.append(f"model.layers.{i}.mlp_mhc_module.phi.weight")
        disable_names.append(f"model.layers.{i}.attn_mhc_module.branch_alpha")
        disable_names.append(f"model.layers.{i}.attn_mhc_module.branch_beta")
        disable_names.append(f"model.layers.{i}.attn_mhc_module.norm_gamma")
        disable_names.append(f"model.layers.{i}.mlp_mhc_module.branch_alpha")
        disable_names.append(f"model.layers.{i}.mlp_mhc_module.branch_beta")
        disable_names.append(f"model.layers.{i}.mlp_mhc_module.norm_gamma")

        if i in (50, 51, 52):
            disable_names.append(f"model.layers.{i}.embed_tokens.weight")
            disable_names.append(f"model.layers.{i}.shared_head.head.weight")
            disable_names.append(f"model.layers.{i}.enorm.weight")
            disable_names.append(f"model.layers.{i}.hnorm.weight")
            disable_names.append(f"model.layers.{i}.shared_head.norm.weight")

    disable_names.append("lm_head.weight")
    disable_names.append("model.norm.weight")
    disable_names.append("model.embed_tokens.weight")
    disable_names.append("model.merge_mhc_module.phi.weight")
    disable_names.append("model.merge_mhc_module.branch_alpha_pre")
    disable_names.append("model.merge_mhc_module.branch_beta_pre")
    disable_names.append("model.merge_mhc_module.norm_gamma")
    return disable_names


def get_disable_names(model_variant):
    return {
        "flash": init_pangu_flash_disable_names,
        "pro": init_pangu_pro_disable_names,
    }[model_variant]()


def normalize_device(device):
    return f"npu:{device}" if device.isdigit() else device


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input-bf16-hf-path", type=str, required=True, help="bf16 weight path")
    parser.add_argument("--output-path", type=str, required=True, help="quantized weight path")
    parser.add_argument(
        "--device",
        type=str,
        required=True,
        help=(
            "NPU device index or comma-separated indices, such as 2 or "
            "0,1,2,3; cpu is also supported for single-device quantization"
        ),
    )
    parser.add_argument("--file_count", type=int, default=0, help="File count when loading model")
    parser.add_argument("--model-name", type=str, default="pangu", help="Huggingface repo name")
    parser.add_argument(
        "--model-variant",
        choices=("flash", "pro"),
        default="pro",
        help="Pangu model variant used to select disabled weights",
    )

    parser.add_argument("--pangu-mode", default=False, action="store_true", help="pangu mode")
    parser.add_argument("--w4", default=False, action="store_true", help="int4 quantization flag")
    parser.add_argument("--clip-ratio", type=float, default=0.95, help="asymmetric INT4 clipping ratio")
    parser.add_argument("--pergroup-to-perchannel", default=False, action="store_true", help="pergroup-to-perchannel")
    parser.add_argument("--num-step", type=int, default=50, help="SSZ optimization steps")
    parser.add_argument("--group-size", type=int, default=0, help="weight quantization group size")
    parser.add_argument("--act-integer", action="store_true", help="use integer activation scales")
    parser.add_argument("--num-bits", type=int, choices=[4], default=4, help="weight quantization bits")
    parser.add_argument(
        "--asymmetric",
        dest="symmetric",
        action="store_false",
        default=True,
        help="use asymmetric weight quantization",
    )
    parser.add_argument("--c8-calib-path", type=str, default=None, help="mla c8 calibration data path")
    parser.add_argument("--kvs-safetensor-name", type=str, default=None, help="mla c8 (faquant) safetensor name")

    args = parser.parse_args()
    args.device = normalize_device(args.device)
    args.qtype = qint4.QType.from_args(args).desc

    if args.c8_calib_path is not None:
        faquant.main(args, args.output_path, args.c8_calib_path, args.kvs_safetensor_name)

    if args.w4:
        disable_names = get_disable_names(args.model_variant)
        if args.pergroup_to_perchannel:
            qint4_pergroup_to_perchannel.main(args, args.input_bf16_hf_path, args.output_path, args.model_name)
        else:
            qint4.main(args, args.input_bf16_hf_path, args.output_path, args.pangu_mode, args.model_name, disable_names)
        weight_is_symmetric = args.symmetric
    else:
        qint8.main(args, args.input_bf16_hf_path, args.output_path, args.pangu_mode, args.model_name)
        disable_names = [
            f"model.layers.{i}.self_attn.kv_b_proj.weight"
            for i in range(49)
        ]
        weight_is_symmetric = True

    quant_config = build_quantization_config(
        output_path=args.output_path,
        disable_names=disable_names,
        expert_num_bits=args.num_bits if args.w4 else 8,
        weight_is_symmetric=weight_is_symmetric,
        quantize="w4a8_dynamic" if args.w4 else "w8a8_dynamic",
    )

    config_path = os.path.join(args.output_path, "config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)

    config["quantization_config"] = quant_config

    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
