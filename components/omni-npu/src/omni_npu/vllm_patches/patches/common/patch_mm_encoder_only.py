# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright contributors to the vLLM project.

from collections.abc import Iterable
from typing import Any, TypeVar

import torch
import torch.nn as nn

from vllm.model_executor.models import adapters
from vllm.config import VllmConfig

from omni_npu.vllm_patches.core import VLLMPatch, register_patch

_T = TypeVar("_T", bound=type[nn.Module])


@register_patch("AdaptersPatch", adapters)
class AdaptersPatch(VLLMPatch):
    _attr_names_to_apply = ['as_mm_encoder_only_model']

    def as_mm_encoder_only_model(cls: _T) -> _T:
        """
        Subclass an existing vLLM vl model to support mm encoder only for
        EPD encoder instances.
        """
        if not hasattr(cls, "embed_multimodal"):
            # Submodel case: return the original class.
            return cls

        if not hasattr(cls, "get_language_model_spec"):
            raise TypeError(f"{cls} need to implement `get_language_model_spec` method.")

        lm_spec = cls.get_language_model_spec()

        # Handle both single model and list of models
        lm_model_cls, lm_attr = lm_spec
        if isinstance(lm_model_cls, list):
            # Multiple models: ([model_cls1, model_cls2, ...], attr)
            lm_model_classes = lm_model_cls
        else:
            # Single model: (model_cls, attr)
            lm_model_classes = [lm_model_cls]

        if None in lm_model_classes or lm_attr is None:
            raise TypeError(
                f"{cls}.get_language_model_spec() must return valid model classes and attributes"
            )

        class DummyLM(nn.Module):
            def __init__(self, *args, **kwargs):
                self.make_empty_intermediate_tensors = None

        class ModelForMMEncoderOnly(cls):
            def __init__(
                self,
                *,
                vllm_config: "VllmConfig",
                prefix: str = "",
                **kwargs: Any,
            ) -> None:
                self.is_mm_encoder_only_model = True
                # Store original __init__ methods for all language models
                origin_inits = [model_cls.__init__ for model_cls in lm_model_classes]
                try:
                    # Replace all language models' __init__ with DummyLM.__init__
                    for model_cls in lm_model_classes:
                        model_cls.__init__ = DummyLM.__init__
                    super().__init__(vllm_config=vllm_config, prefix=prefix, **kwargs)

                    # Delete the language model attribute
                    if hasattr(self, lm_attr):
                        delattr(self, lm_attr)
                finally:
                    # Restore all original __init__ methods
                    for model_cls, origin_init in zip(lm_model_classes, origin_inits):
                        model_cls.__init__ = origin_init

            def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
                from vllm.model_executor.models.utils import AutoWeightsLoader

                origin_init_ = AutoWeightsLoader.__init__

                def _new_init_(self, *args, **kwargs):
                    origin_init_(self, *args, **kwargs)
                    self.skip_prefixes = (self.skip_prefixes or []) + [f"{lm_attr}."]

                try:
                    AutoWeightsLoader.__init__ = _new_init_
                    result = super().load_weights(weights)
                finally:
                    AutoWeightsLoader.__init__ = origin_init_
                return result

        return ModelForMMEncoderOnly  # type: ignore
