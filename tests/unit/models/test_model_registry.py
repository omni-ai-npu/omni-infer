# SPDX-License-Identifier: MIT

# test_register_models_uses_mock_qwen2_when_capture_env_enabled was removed during the
# vLLM 0.25.1 migration: the capture/replay mock-model path it covered was not migrated
# (there is no omni/v1/models/mock/ package and register_models() no longer has a
# CAPTURE_MODE branch), and it pinned Qwen2ForCausalLM, an open-source model that is
# out of scope for this stage.
