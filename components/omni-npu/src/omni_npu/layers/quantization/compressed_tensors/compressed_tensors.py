# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import os
from typing import Any, List, Optional

import torch
from compressed_tensors.quantization import QuantizationArgs, QuantizationStrategy

from vllm.model_executor.layers.fused_moe.layer import FusedMoE
from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
    QUANTIZATION_SCHEME_MAP_TYPE,
    CompressedTensorsConfig,
    CompressedTensorsLinearMethod
)
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import CompressedTensorsScheme
from vllm.model_executor.layers.quantization.compressed_tensors.utils import (
    find_matched_target,
    is_activation_quantization_format,
    should_ignore_layer
)

from omni_npu.layers.quantization.compressed_tensors.compressed_tensors_moe import \
    NPUCompressedTensorsW8A8Int8MoEMethod, NPUCompressedTensorsW4A8Int4MoEMethod
from omni_npu.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_int8 import \
    NPUCompressedTensorsW8A8Int8
    

NPU_COMPRESSED_TENSORS = "npu-compressed-tensors"


@register_quantization_config(NPU_COMPRESSED_TENSORS)
class NPUCompressedTensorsConfig(CompressedTensorsConfig):
    """Config class for NPU

    This class is a general class that parse quantization configs
    that are supported on NPU hardware.
    """

    @classmethod
    def _quantization_scheme_map_from_config(
        cls, config: dict[str, Any]
    ) -> QUANTIZATION_SCHEME_MAP_TYPE:
        """
        :param config: The `quantization_config` dictionary from config.json
        :return: A dictionary mapping target layer names to their corresponding
            quantization_args for weights and input activations
        """
        target_scheme_map: dict[str, Any] = dict()
        config_groups = config.get("config_groups", dict())
        for _, quant_config in config_groups.items():
            targets = quant_config.get("targets")
            for target in targets:
                target_scheme_map[target] = {}
                # adapt: do not validate parameters
                module_num_bits = quant_config.get("weights").get("num_bits")
                quant_config["weights"]["num_bits"] = 0
                target_scheme_map[target]["weights"] = QuantizationArgs.parse_obj(quant_config.get("weights"))
                quant_config["weights"]["num_bits"] = module_num_bits
                target_scheme_map[target]["weights"].num_bits = module_num_bits
                try:
                    target_scheme_map[target]["input_activations"] = (
                        QuantizationArgs.parse_obj(quant_config.get("input_activations"))
                    )
                except Exception:
                    target_scheme_map[target]["input_activations"] = None
        return target_scheme_map

    def get_scheme(
            self,
            layer: torch.nn.Module,
            layer_name: Optional[str] = None) -> "CompressedTensorsScheme":
        if should_ignore_layer(
            layer_name, ignore=self.ignore, fused_mapping=self.packed_modules_mapping
        ):
            return None

        matched_target = find_matched_target(
            layer_name=layer_name,
            module=layer,
            targets=self.target_scheme_map.keys())
        # Find the quant_scheme
        scheme_dict = self.target_scheme_map[matched_target]
        # Adapter: pass layer_name
        scheme = self._get_scheme_from_parts(
            layer_name=layer_name,
            weight_quant=scheme_dict["weights"],
            input_quant=scheme_dict["input_activations"])
        return scheme

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.int8, torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        raise NotImplementedError(
            "NPU hardware dose not support \"get_min_capability\" feature.")

    @staticmethod
    def _is_dynamic_token_w8a8(
        weight_quant: QuantizationArgs, input_quant: QuantizationArgs, weight_num_bits: int
    ) -> bool:
        is_8_bits = weight_num_bits == input_quant.num_bits == 8
        weight_strategy = (
            weight_quant.strategy == QuantizationStrategy.TENSOR.value
            or weight_quant.strategy == QuantizationStrategy.CHANNEL.value
            or weight_quant.strategy == QuantizationStrategy.GROUP.value)
        is_token = (weight_strategy and input_quant.strategy == QuantizationStrategy.TOKEN.value)
        is_dynamic = not weight_quant.dynamic and input_quant.dynamic

        # Both symmetric and asymmetric input quantization supported.
        # Only symmetric weight quantization supported.
        return is_8_bits and is_token and weight_quant.symmetric and is_dynamic

    @staticmethod
    def _is_dynamic_token_w4a8_int(
        weight_quant: QuantizationArgs, input_quant: QuantizationArgs, weight_num_bits: int
    ) -> bool:
        is_weight_4_bits = weight_num_bits == 4
        is_activation_8_bits = input_quant.num_bits == 8
        weight_strategy = (
            weight_quant.strategy == QuantizationStrategy.TENSOR.value
            or weight_quant.strategy == QuantizationStrategy.CHANNEL.value
            or weight_quant.strategy == QuantizationStrategy.GROUP.value
        )
        is_token = (weight_strategy and input_quant.strategy == QuantizationStrategy.TOKEN.value)
        is_dynamic = not weight_quant.dynamic and input_quant.dynamic

        # Both symmetric and asymmetric input quantization supported.
        # Only symmetric weight quantization supported.
        return is_weight_4_bits and is_activation_8_bits and is_token and weight_quant.symmetric and is_dynamic

    def _get_weight_num_bits(
        self,
        layer_name: str,
        weight_quant: QuantizationArgs
    ) -> int:
        if isinstance(weight_quant.num_bits, dict):
            for module, module_num_bits in weight_quant.num_bits.items():
                if module in layer_name:
                    return module_num_bits
            raise ValueError(
                "weight name mismatch, please check weights num_bits in config.json "
                f"and model weight name. layer_name={layer_name}"
            )
        else:
            return weight_quant.num_bits

    def _get_scheme_from_parts(
        self,
        weight_quant: QuantizationArgs,
        input_quant: QuantizationArgs,
        fmt: Optional[str] = None,
        layer_name: Optional[str] = None,
    ) -> "CompressedTensorsScheme":
        # use the per-layer format if defined, otherwise, use global format
        fmt = fmt if fmt is not None else self.quant_format

        act_quant_format = is_activation_quantization_format(fmt)
        if act_quant_format:
            weight_num_bits = self._get_weight_num_bits(layer_name, weight_quant)
            if weight_num_bits == 16:
                return None

            if self._is_dynamic_token_w8a8(weight_quant, input_quant, weight_num_bits):
                return NPUCompressedTensorsW8A8Int8(
                    strategy=weight_quant.strategy,
                    is_static_input_scheme=False,
                    input_symmetric=input_quant.symmetric,
                )

            if self._is_dynamic_token_w4a8_int(weight_quant, input_quant, weight_num_bits):
                raise NotImplementedError

        raise NotImplementedError("No compressed-tensors compatible scheme was found.")

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg,
                                     user_quant) -> Optional[str]:
        quant_method = hf_quant_cfg['quant_method']
        if torch.npu.is_available() and quant_method == 'compressed-tensors':
            return NPU_COMPRESSED_TENSORS
        return None

    def get_moe_method(self, layer: FusedMoE) -> Optional[QuantizeMethodBase]:
        if "Linear" in self.target_scheme_map:
            matched_target = "Linear"
        else:
            # May have instead defined the linear layers in the fused model

            fused_layers = ["re:.*down_proj.*", "re:.*gate_proj.*", "re:.*up_proj.*"]
            current_scheme = None
            for fused_layer in fused_layers:
                # Check if one of the fused layers are defined in quant_config
                matched_target = find_matched_target(
                    layer_name=fused_layer,
                    module=layer,
                    targets=self.target_scheme_map.keys(),
                    fused_mapping=self.packed_modules_mapping,
                )

                # Only valid if down_proj, gate_proj, and up_proj
                # are mapped to the same quant scheme in the quant_config
                if current_scheme is None:
                    current_scheme = self.target_scheme_map.get(matched_target)
                else:
                    assert current_scheme == self.target_scheme_map.get(
                        matched_target
                    )

        weight_quant = self.target_scheme_map[matched_target].get("weights")
        input_quant = self.target_scheme_map[matched_target].get(
            "input_activations"
        )
        weight_num_bits = self._get_weight_num_bits("mlp.experts", weight_quant)
        if self._is_dynamic_token_w8a8(weight_quant, input_quant, weight_num_bits):
            return NPUCompressedTensorsW8A8Int8MoEMethod(
                weight_quant, input_quant, layer)
        elif self._is_dynamic_token_w4a8_int(weight_quant, input_quant, weight_num_bits):
            layer.weight_num_bits = 4
            return NPUCompressedTensorsW4A8Int4MoEMethod(weight_quant, input_quant, layer)
        else:
            raise RuntimeError(
                f"Unsupported FusedMoe scheme: {weight_quant}, {input_quant}"
            )

    def get_fc_method(
        self,
        layer: torch.nn.Module,
        layer_name: str = None,
        sharded_linear: bool = False,
    ) -> QuantizeMethodBase:
        from omni_npu.v1.layers.linear import (
            UnquantizedFlashCommLinearMethod,
            UnquantizedShardedLinearMethod,
        )
        from omni_npu.v1.layers.quantization.compressed_tensors.npu_compressed_tensors_linear import (
            W8A8Int8FCLinearMethod,
            W8A8Int8ShardedLinearMethod,
        )

        if should_ignore_layer(
            layer_name, ignore=self.ignore, fused_mapping=self.packed_modules_mapping
        ):
            if sharded_linear:
                return UnquantizedShardedLinearMethod()
            return UnquantizedFlashCommLinearMethod()

        matched_target = find_matched_target(
            layer_name=layer_name,
            module=layer,
            targets=self.target_scheme_map.keys())
        scheme_dict = self.target_scheme_map[matched_target]
        weight_quant = scheme_dict["weights"]
        input_quant = scheme_dict["input_activations"]
        fmt = self.quant_format

        act_quant_format = is_activation_quantization_format(fmt)
        if act_quant_format:
            weight_num_bits = self._get_weight_num_bits(layer_name, weight_quant)
            if weight_num_bits == 16:
                if sharded_linear:
                    return UnquantizedShardedLinearMethod()
                return UnquantizedFlashCommLinearMethod()

            if self._is_dynamic_token_w8a8(weight_quant, input_quant, weight_num_bits):
                if sharded_linear:
                    return W8A8Int8ShardedLinearMethod()
                return W8A8Int8FCLinearMethod(self)

            if self._is_dynamic_token_w4a8_int(weight_quant, input_quant, weight_num_bits):
                raise NotImplementedError

        raise NotImplementedError("No compressed-tensors compatible method was found.")

    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
    ) -> Optional["QuantizeMethodBase"]:

        if isinstance(layer, LinearBase):
            # collect schemes
            quant_scheme = self.get_scheme(layer=layer, layer_name=prefix)
            # choose quantization method
            quant_method: LinearMethodBase = UnquantizedLinearMethod()
            if quant_scheme is not None:
                layer.scheme = quant_scheme
                quant_method = CompressedTensorsLinearMethod(self)
            return quant_method
        if isinstance(layer, FusedMoE):
            return self.get_moe_method(layer)

        if "omni_custom_models" in os.environ.get("VLLM_PLUGINS", ""):
            from omni_npu.v1.layers.fused_mlp.layer import FusedMLP
            from omni_npu.v1.layers.linear import FlashCommLinearBase, ShardedLinear
            from omni_npu.v1.layers.quantization.compressed_tensors.npu_compressed_tensors_linear import (
                W8A8Int8MlpMethod,
            )
            if isinstance(layer, FlashCommLinearBase):
                return self.get_fc_method(layer, prefix)
            if isinstance(layer, ShardedLinear):
                return self.get_fc_method(layer, prefix, sharded_linear=True)
            if isinstance(layer, FusedMLP):
                return W8A8Int8MlpMethod(self)
        return None
