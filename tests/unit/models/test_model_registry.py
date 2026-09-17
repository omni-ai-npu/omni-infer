# SPDX-License-Identifier: MIT

from unittest.mock import MagicMock, patch

from omni_npu.v1 import models as omni_models


def test_v3_state_configuration_preserves_v2():
    from vllm.model_executor.models.interfaces import is_hybrid
    from vllm.v1.worker.gpu.model_states.default import DefaultModelState

    from omni_npu.v1.models.pangu.pangu_v2_moe import OpenPanguV2ForCausalLM
    from omni_npu.v1.models.pangu.pangu_v3_moe import OpenPanguV3ForCausalLM
    from omni_npu.worker.npu.mome_model_state import MomeModelState

    assert is_hybrid(OpenPanguV2ForCausalLM) is True
    assert is_hybrid(OpenPanguV3ForCausalLM) is False
    assert OpenPanguV2ForCausalLM.get_model_state_cls() is MomeModelState
    assert OpenPanguV3ForCausalLM.get_model_state_cls() is DefaultModelState


def test_register_models_uses_openpangu_v2_architectures():
    registry = MagicMock()
    with patch.object(omni_models, "ModelRegistry", registry):
        omni_models.register_models()

    registry.register_model.assert_any_call(
        "OpenPanguV2ForCausalLM",
        "omni_npu.v1.models.pangu.pangu_v2_moe:OpenPanguV2ForCausalLM",
    )
    registry.register_model.assert_any_call(
        "OpenPanguV2MTPModel",
        "omni_npu.v1.models.pangu.pangu_v2_moe_mtp:OpenPanguV2MTP",
    )
    registry.register_model.assert_any_call(
        "PanguUltraMoEForCausalLM",
        "omni_npu.v1.models.pangu.pangu_ultra_moe:PanguUltraMoEForCausalLM",
    )
    registry.register_model.assert_any_call(
        "OpenPanguMTPModel",
        "omni_npu.v1.models.pangu.pangu_ultra_moe_mtp:OpenPanguMTP",
    )
    assert registry.register_model.call_count == 5

    registry.register_model.assert_any_call(
        "OpenPanguV3ForCausalLM",
        "omni_npu.v1.models.pangu.pangu_v3_moe:OpenPanguV3ForCausalLM",
    )
