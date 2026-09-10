# Testing Guide for omni-npu

## Overview

Comprehensive test suite for the omni-npu package with clear separation between unit and integration tests.

**Current Status:** NPUCommunicator tests complete (28+ tests). Serves as template for testing other components.

The test suite is organized into two categories:

1. **Unit Tests** (`tests/unit/`) - Do NOT require NPU hardware
2. **Integration Tests** (`tests/integration/`) - REQUIRE NPU hardware

## Quick Start

### Run unit tests (no NPU required)
```bash
./run_tests.sh unit
```

### Run integration tests (requires NPU)
```bash
./run_tests.sh integration
```

### Run all tests
```bash
./run_tests.sh all
# or simply
./run_tests.sh
```

## Test Categories

### Unit Tests
Located in `tests/unit/`, these tests use mocking and do not require NPU hardware.

**Currently Testing (NPUCommunicator):**
- ✅ Initialization logic and error handling
- ✅ API contracts and method signatures
- ✅ Correct delegation to torch.distributed
- ✅ Edge cases (world_size=1, negative dims, etc.)

**To Be Added:**
- 🔲 NPUPlatform: Device management, memory operations
- 🔲 Attention backends: Attention mechanisms, MLA
- 🔲 NPU Worker & Model Runner: Batch processing, model execution
- 🔲 Sampler: Sampling strategies

**Run with:**
```bash
pytest tests/unit/ -v                    # All unit tests
pytest tests/unit/distributed/ -v       # Just communicator tests
```

### Integration Tests
Located in `tests/integration/`, these tests require actual NPU hardware.

**Currently Testing (NPUCommunicator):**
- ✅ Real NPU device operations
- ✅ End-to-end distributed communication
- ✅ Multi-device tensor operations
- ✅ Memory allocation and management

**To Be Added:**
- 🔲 Attention backends: End-to-end attention with real NPU
- 🔲 NPU Worker: End-to-end model inference workflows

**Run with:**
```bash
pytest tests/integration/ -v                                # All integration tests
pytest tests/integration/distributed/ -v                    # Just communicator tests
```

**Multi-device tests:**
```bash
torchrun --nproc_per_node=2 -m pytest tests/integration/distributed/test_communicator.py::TestNPUCommunicatorMultiDevice
```

## Test Coverage

Generate coverage reports:
```bash
pytest tests/unit/ --cov=omni_npu --cov-report=html --cov-report=term
```

View HTML coverage report:
```bash
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

## CI/CD Integration

### GitHub Actions / GitLab CI (no NPU hardware)
```yaml
- name: Run unit tests
  run: |
    pip install -e ".[test]"
    pytest tests/unit/ -v --cov=omni_npu
```

### On-premise CI with NPU hardware
```yaml
- name: Run all tests
  run: |
    pip install -e ".[test,npu]"
    pytest tests/ -v --cov=omni_npu
```

## Writing New Tests

### Adding Unit Tests
1. Create test file in `tests/unit/`
2. Use mocking for torch.npu and vLLM dependencies
3. Test logic, not hardware functionality
4. Example:
```python
def test_my_feature(self):
    with patch.object(torch, 'npu', create=True):
        # Your test code here
        pass
```

### Adding Integration Tests
1. Create test file in `tests/integration/`
2. Use `@skipif_no_npu` decorator
3. Test with real NPU hardware
4. Example:
```python
@skipif_no_npu
class TestMyFeature(unittest.TestCase):
    def test_with_real_npu(self):
        device = torch.device('npu:0')
        # Your test code here
```

## Troubleshooting

### "NPU hardware not available"
Integration tests are automatically skipped if NPU is not detected. This is expected on systems without NPU hardware.

### "Distributed environment not initialized"
Multi-device tests require `torchrun`. Run with:
```bash
torchrun --nproc_per_node=N -m pytest tests/integration/...
```

### Import errors
Install test dependencies:
```bash
pip install -e ".[test]"
```

For NPU hardware:
```bash
pip install -e ".[test,npu]"
```

## Test Metrics

Current test coverage:
- Unit tests: 20+ test cases covering all NPUCommunicator methods
- Integration tests: 8+ test cases for real hardware validation
- Target coverage: >80% for production code
