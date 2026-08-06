# SPDX-License-Identifier: MIT

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import omni.v1.models as models_mod


def test_register_models_uses_mock_qwen2_when_capture_env_enabled():
    qwen2_cls = object()
    mock_qwen2_cls = object()
    qwen2_module = SimpleNamespace(Qwen2ForCausalLM=qwen2_cls)
    mock_factory = MagicMock(return_value=mock_qwen2_cls)
    mock_module = SimpleNamespace(mock_model_class_factory=mock_factory)

    def fake_import_module(name):
        if name == "vllm.model_executor.models.qwen2":
            return qwen2_module
        if name == "omni.v1.models.mock.mock":
            return mock_module
        raise AssertionError(f"unexpected import: {name}")

    with patch.dict(
        "os.environ",
        {
            "RANDOM_MODE": "0",
            "CAPTURE_MODE": "1",
            "REPLAY_MODE": "0",
        },
        clear=False,
    ), patch.object(
        models_mod,
        "import_module",
        side_effect=fake_import_module,
    ), patch.object(models_mod.ModelRegistry, "register_model") as register_model:
        models_mod.register_models()

    mock_factory.assert_called_once_with(qwen2_cls)
    register_model.assert_any_call("Qwen2ForCausalLM", mock_qwen2_cls)
