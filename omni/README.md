# omni-npu (vLLM NPU Plugin)

`omni-npu` is an out-of-tree vLLM platform plugin adapted for vLLM 0.25.1
on Ascend NPU devices through `torch_npu`.

- It is discovered through vLLM plugin entry points.
- It provides the NPU platform, worker, model-runner integration, custom
  models, and runtime patches used by OmniInfer.
- It keeps vLLM's serving interfaces while replacing device-specific runtime
  behavior with NPU implementations.

## Requirements

- Python 3.11 or newer
- vLLM 0.25.1
- A matching PyTorch, `torch_npu`, CANN, and Ascend driver stack

## Installation

Install the matching vLLM and Ascend runtime first, then install this package:

```bash
pip install -e .
```

The following entry points are registered:

- `vllm.platform_plugins`: NPU platform discovery
- `vllm.general_plugins`: Omni patches and custom model registration
- `vllm.stat_logger_plugins`: Omni NPU metrics

## Development

- [Configuration development guide](docs/config_dev_guide.md): environment
  variables and `OmniAdditionalConfig`
- [ModelExtraConfig guide](src/omni_npu/model_config/README.md): model-specific
  configuration and best-practice JSON files
- [ValidationRule development guide](docs/config_validation_rules.md): declarative
  startup validation

Run the configuration tests without NPU hardware:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
pytest -q -p no:cacheprovider tests/config
```

Hardware-dependent builds and integration tests must run in an environment
with the matching Ascend toolchain.

## License

MIT
