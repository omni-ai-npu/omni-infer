# SPDX-License-Identifier: MIT

from unittest.mock import MagicMock, patch

from omni_npu.v1 import models as omni_models


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
    assert registry.register_model.call_count == 4

