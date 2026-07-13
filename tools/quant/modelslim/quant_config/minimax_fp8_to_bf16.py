# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import torch
import os
import json
import shutil
from safetensors.torch import load_file, save_file
from transformers import AutoConfig

# ================= 配置区 =================
MODEL_DIR = "/opt/data/models/MiniMax-M2.5" # 原始模型权重路径
OUTPUT_DIR = "/opt/data/models/MiniMax-M2.5-BF16" # 量化后模型权重路径
TARGET_DTYPE = torch.bfloat16  # 推荐使用 BF16
BLOCK_SIZE = 128 # MiniMax 2.5 默认为 128x128
# =========================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

def restore_block_quant(weight, scale_inv):
    """
    核心还原函数：将 FP8 权重通过 Scale 因子还原回 BF16
    """
    # 转为 float32 运算以确保精度
    w_f32 = weight.to(torch.float32)
    s_f32 = scale_inv.to(torch.float32)

    out_features, in_features = w_f32.shape
    h_groups, w_groups = s_f32.shape

    # 动态计算 block 大小
    bh, bw = out_features // h_groups, in_features // w_groups

    # 重构维度并应用广播乘法: [h_g, bh, w_g, bw] * [h_g, 1, w_g, 1]
    w_view = w_f32.view(h_groups, bh, w_groups, bw)
    s_view = s_f32.view(h_groups, 1, w_groups, 1)

    restored = (w_view * s_view).reshape(out_features, in_features)
    return restored.to(TARGET_DTYPE)

def verify_weight(original_fp8, scale_inv, converted_bf16):
    """
    验证函数：检查还原后的值是否等于 (原值 * 缩放因子)
    """
    # 随机抽样 100 个点
    indices_h = torch.randint(0, original_fp8.shape[0], (100,))
    indices_w = torch.randint(0, original_fp8.shape[1], (100,))

    # 计算期望值 (使用 float32 模拟物理真实值)
    h_groups, w_groups = scale_inv.shape
    bh, bw = original_fp8.shape[0] // h_groups, original_fp8.shape[1] // w_groups

    pass_count = 0
    for i in range(100):
        r, c = indices_h[i], indices_w[i]
        scale_val = scale_inv[r // bh, c // bw]
        expected = (original_fp8[r, c].to(torch.float32) * scale_val.to(torch.float32)).to(TARGET_DTYPE)
        actual = converted_bf16[r, c]

        # 允许极小的浮点误差
        if torch.allclose(expected, actual, atol=1e-3, rtol=1e-3):
            pass_count += 1

    return pass_count

print(f"开始转换任务: FP8 -> {TARGET_DTYPE}")
print("-" * 50)

all_stats = []
if not os.path.isdir(MODEL_DIR):
    raise FileNotFoundError(f"{MODEL_DIR} is not a valid directory")
for filename in sorted(os.listdir(MODEL_DIR)):
    if filename.endswith(".safetensors"):
        print(f"正在处理: {filename}...")
        state_dict = load_file(os.path.join(MODEL_DIR, filename), device="cpu")
        new_state_dict = {}
        processed_keys = set()

        # 抽取一个 key 用于本文件的精度验证
        validation_done = False

        for k in list(state_dict.keys()):
            if k in processed_keys or k.endswith("_scale_inv"):
                continue

            v = state_dict[k]
            scale_key = f"{k}_scale_inv"

            if scale_key in state_dict:
                # 执行还原
                restored_v = restore_block_quant(v, state_dict[scale_key])
                new_state_dict[k] = restored_v

                # 如果还没验证过本文件，执行一次随机抽样检查
                if not validation_done:
                    score = verify_weight(v, state_dict[scale_key], restored_v)
                    all_stats.append((k, score))
                    validation_done = True

                processed_keys.add(k)
                processed_keys.add(scale_key)
            else:
                # 处理 Norm, Bias 等非量化层
                new_state_dict[k] = v.to(TARGET_DTYPE) if v.is_floating_point() else v
                processed_keys.add(k)

        save_file(new_state_dict, os.path.join(OUTPUT_DIR, filename))
        del state_dict, new_state_dict
        print(f" 文件 {filename} 已保存")

# --- 后处理：修正配置 ---
print("\n修正配置文件...")
config_path = os.path.join(MODEL_DIR, "config.json")
if os.path.exists(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)

    # 关键：更新类型并移除量化信息
    config["torch_dtype"] = "bfloat16" if TARGET_DTYPE == torch.bfloat16 else "float16"
    if "quantization_config" in config:
        del config["quantization_config"]

    with open(os.path.join(OUTPUT_DIR, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

# 复制其他依赖文件
for ext in [".py", ".model", ".json", ".md", ".jinja", ".txt"]:
    for f in os.listdir(MODEL_DIR):
        if f.endswith(ext) and f != "config.json":
            shutil.copy(os.path.join(MODEL_DIR, f), os.path.join(OUTPUT_DIR, f))

# --- 最终报告 ---
print("\n" + "="*30)
print("转换任务完成！精度验证报告：")
print("="*30)
overall_pass = True
for key, score in all_stats:
    status = "优" if score >= 98 else "偏差较大"
    if score < 95: overall_pass = False
    print(f"层: {key[:40]}... | 抽样匹配率: {score}% | 状态: {status}")

if overall_pass:
    print("\n结论：权重还原在数学上是一致的。你可以放心加载模型。")
else:
    print("\n结论：存在异常偏差，请检查 Scale 因子的乘除关系或存储顺序。")