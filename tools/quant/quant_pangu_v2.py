import optiquant.int8 as qint8
import optiquant.int4 as qint4
import optiquant.w4group_to_w4channel as qint4_pergroup_to_perchannel
import optiquant.faquant as faquant
from argparse import ArgumentParser
import json
import os

def init_disable_names():
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
        # *_mhc_module.branch_alpha/branch_beta/norm_gamma
        disable_names.append(f"model.layers.{i}.attn_mhc_module.branch_alpha")
        disable_names.append(f"model.layers.{i}.attn_mhc_module.branch_beta")
        disable_names.append(f"model.layers.{i}.attn_mhc_module.norm_gamma")
        disable_names.append(f"model.layers.{i}.mlp_mhc_module.branch_alpha")
        disable_names.append(f"model.layers.{i}.mlp_mhc_module.branch_beta")
        disable_names.append(f"model.layers.{i}.mlp_mhc_module.norm_gamma")
        if i not in (46, 47, 48):
            continue
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
        # Attention(MLA)层
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

        # MoE层
        disable_names.append(f"model.layers.{i}.mlp.gate.weight")
        disable_names.append(f"model.layers.{i}.mlp.shared_experts.gate_proj.weight")
        disable_names.append(f"model.layers.{i}.mlp.shared_experts.up_proj.weight")
        disable_names.append(f"model.layers.{i}.mlp.shared_experts.down_proj.weight")
        disable_names.append(f"model.layers.{i}.mlp.e_score_correction_bias")

        # MHC / MOME / Norm / 嵌入 / 输出头
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

        # MTP层
        if i not in (50, 51, 52):
            continue
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

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input-bf16-hf-path", type=str, required=True, help="bf16 weight path")
    parser.add_argument("--output-path", type=str, required=True, help="quantized weight path")
    parser.add_argument("--device", type=str, required=True, help="support cpu and npu")
    parser.add_argument("--file_count", type=int, default=0, help="File count when loading model")
    parser.add_argument("--model-name", type=str, default="deepseek-ai/DeepSeek-R1", help="Huggingface repo name")

    parser.add_argument("--pangu-mode", default=False, action="store_true", help="pangu mode")
    parser.add_argument("--w4", default=False, action="store_true", help="int4 quantization flag")
    parser.add_argument("--weight-post-process", default=False, action="store_true", help="add post process for bias or offset")
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
    args.qtype = qint4.QType.from_args(args).desc

    if args.c8_calib_path is not None:
        faquant.main(args, args.output_path, args.c8_calib_path, args.kvs_safetensor_name)

    if args.w4:
        if args.pergroup_to_perchannel:
            qint4_pergroup_to_perchannel.main(args, args.input_bf16_hf_path, args.output_path, args.model_name)
        else:
            qint4.main(args, args.input_bf16_hf_path, args.output_path, args.pangu_mode, args.model_name, init_disable_names(), args.weight_post_process)
        num_bits = {"self_attn.q_b_proj": 8, "self_attn.o_proj": 8, "mlp.down_proj": 8, "mlp.gate_up_proj": 8,
                    "mlp.experts": args.num_bits}
        weight_is_symmetric = args.symmetric
    else:
        qint8.main(args, args.input_bf16_hf_path, args.output_path, args.pangu_mode, args.model_name)
        num_bits = 8
        weight_is_symmetric = True

    ignores = []
    for i in range(49):
        ignore = f"model.layers.{i}.self_attn.kv_b_proj"
        ignores.append(ignore)

    quant_config = {"config_groups": {"group_0": {}}, "format": "int-quantized",
                    "global_compression_ratio": 1.5943962512751309, "ignore": ignores, "kv_cache_scheme": None,
                    "quant_method": "compressed-tensors", "quantization_status": "compressed"}
    quant_config["config_groups"]["group_0"]["input_activations"] = {"actorder": None, "block_structure": None,
                                                                     "dynamic": True, "group_size": None, "num_bits": 8,
                                                                     "observer": "memoryless", "observer_kwargs": {},
                                                                     "strategy": "token", "symmetric": True,
                                                                     "type": "int"}
    quant_config["config_groups"]["group_0"]["output_activations"] = None
    quant_config["config_groups"]["group_0"]["targets"] = ["Linear"]
    quant_config["config_groups"]["group_0"]["weights"] = {"actorder": None, "block_structure": None, "dynamic": False,
                                                           "group_size": None, "num_bits": num_bits,
                                                           "observer": "minmax", "observer_kwargs": {},
                                                           "strategy": "channel", "symmetric": weight_is_symmetric,
                                                           "type": "int"}
    if not weight_is_symmetric:
        quant_config["config_groups"]["group_0"]["weights"]["asymmetric_group"] = ["mlp.experts"]

    config_path = os.path.join(args.output_path, "config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)

    config["quantization_config"] = quant_config

    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
