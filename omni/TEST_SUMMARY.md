# Test Suite Summary for omni-npu

## Overview

Comprehensive test suite for the omni-npu package with clear separation between unit tests (no hardware required) and integration tests (NPU hardware required).

The test directory structure mirrors the source code structure in `src/omni_npu/` for easy navigation and maintenance. Currently implemented tests for `NPUCommunicator` serve as a template for testing other components.

## Files Created

### Test Files

**Current Structure:**
```
tests/
├── __init__.py
├── README.md                                    # Test documentation and usage
├── QUICKSTART.md                                # Quick reference guide
├── TESTING.md                                   # Detailed testing guide
├── STRUCTURE.md                                 # Test structure documentation
│
├── unit/                                        # Unit tests (no NPU required)
│   ├── __init__.py
│   └── distributed/
│       ├── __init__.py
│       └── test_communicator.py                # ✅ 20+ unit tests for NPUCommunicator
│
└── integration/                                 # Integration tests (NPU required)
    ├── __init__.py
    └── distributed/
        ├── __init__.py
        └── test_communicator.py                # ✅ 8+ integration tests for NPUCommunicator
```

**Planned Structure** (to be implemented):
```
tests/
├── unit/
│   ├── distributed/
│   │   └── test_communicator.py                # ✅ Implemented
│   ├── attention/
│   │   └── backends/
│   │       ├── test_attention.py               # 🔲 TODO: Attention backend tests
│   │       └── test_mla.py                     # 🔲 TODO: MLA backend tests
│   ├── v1/
│   │   ├── sample/
│   │   │   └── test_sampler.py                 # 🔲 TODO: Sampler tests
│   │   └── worker/
│   │       ├── test_npu_model_runner.py        # 🔲 TODO: Model runner tests
│   │       └── test_npu_worker.py              # 🔲 TODO: NPU worker tests
│   └── test_platform.py                        # 🔲 TODO: NPUPlatform tests
│
└── integration/
    ├── distributed/
    │   └── test_communicator.py                # ✅ Implemented
    ├── attention/
    │   └── backends/
    │       └── test_attention.py               # 🔲 TODO: End-to-end attention tests
    └── v1/
        └── worker/
            └── test_npu_worker.py              # 🔲 TODO: End-to-end worker tests
```

### Configuration Files
- `pytest.ini` - Pytest configuration with markers and coverage settings
- `pyproject.toml` - Updated with test dependencies and pytest config
- `run_tests.sh` - Shell script to run different test suites

## Test Coverage

### Currently Implemented: NPUCommunicator

#### Unit Tests (tests/unit/distributed/)
**No NPU hardware required** - Uses mocking

✅ **Initialization Tests**
- Test with torch.npu available
- Test without torch.npu (raises RuntimeError)

✅ **Collective Operations**
- `all_reduce()` - Delegation to torch.distributed
- `all_gather()` - Shape transformation logic
- `reduce_scatter()` - World size handling, delegation
- `reduce_scatterv()` - Variable sizes support
- `all_gatherv()` - NotImplementedError for dim != 0
- `gather()` - Destination vs non-destination rank behavior

✅ **Point-to-Point Operations**
- `send()` - Explicit and default destination
- `recv()` - Tensor creation with correct shape/dtype

✅ **Edge Cases**
- World size = 1 optimization
- Negative dimension handling
- Destroy method

#### Integration Tests (tests/integration/distributed/)
**Requires NPU hardware** - Real device testing

✅ **Single Device Tests**
- NPU device availability
- Tensor creation on NPU
- Memory allocation/deallocation
- Basic NPU operations (add, matmul)
- Communicator initialization with real NPU

✅ **Multi-Device Tests** (requires torchrun)
- Real all_reduce with multiple NPUs
- Real all_gather with multiple NPUs
- Real send/recv between ranks

### To Be Implemented

#### NPUPlatform (tests/unit/test_platform.py)
🔲 Platform detection and initialization
🔲 Device management (set_device, get_device_name, device_count)
🔲 Memory management methods
🔲 Platform-specific operations

#### Attention Backends (tests/unit/attention/backends/)
🔲 Attention mechanism tests
🔲 MLA (Multi-head Latent Attention) tests
🔲 Attention builder tests
🔲 Integration with NPU kernels

#### NPU Worker (tests/unit/v1/worker/)
🔲 NPUWorker initialization and configuration
🔲 NPUModelRunner model loading and execution
🔲 Batch processing and scheduling
🔲 Memory management

#### Sampler (tests/unit/v1/sample/)
🔲 Sampling strategies
🔲 Temperature and top-k/top-p sampling
🔲 Batch sampling operations

## Running Tests

### Quick Commands

```bash
# Unit tests only (no NPU)
./run_tests.sh unit

# Integration tests only (requires NPU)
./run_tests.sh integration

# All tests
./run_tests.sh all
```

### Using pytest directly

```bash
# Run only unit tests
pytest tests/unit/ -v

# Run specific component tests
pytest tests/unit/distributed/ -v
pytest tests/integration/distributed/ -v

# Run with markers
pytest -m unit -v                    # Only unit tests
pytest -m integration -v             # Only integration tests
pytest -m "not multi_device" -v      # Skip multi-device tests

# Run with coverage
pytest tests/unit/ --cov=omni_npu --cov-report=html
```

### Multi-device tests

```bash
# Run multi-device tests with torchrun (2 NPUs)
torchrun --nproc_per_node=2 -m pytest tests/integration/distributed/test_communicator.py::TestNPUCommunicatorMultiDevice -v

# Run with 4 NPUs
torchrun --nproc_per_node=4 -m pytest tests/integration/distributed/test_communicator.py::TestNPUCommunicatorMultiDevice -v
```

## Test Markers

Tests are marked with pytest markers for easy filtering:

- `@pytest.mark.unit` - Unit tests (no hardware)
- `@pytest.mark.integration` - Integration tests (requires NPU)
- `@pytest.mark.multi_device` - Requires multiple NPUs
- `@pytest.mark.slow` - Long-running tests

## CI/CD Integration

### Without NPU Hardware (GitHub Actions, etc.)
```yaml
- run: pip install -e ".[test]"
- run: pytest tests/unit/ -v --cov=omni_npu
```

### With NPU Hardware (On-premise CI)
```yaml
- run: pip install -e ".[test,npu]"
- run: pytest tests/ -v --cov=omni_npu
```

## Key Features

✅ **Mirrors Source Structure**
- Test directory layout matches `src/omni_npu/` structure
- Easy to find tests for any component
- Scalable as codebase grows

✅ **Separation of Concerns**
- Unit tests don't require NPU hardware
- Integration tests automatically skip if NPU unavailable
- Multi-device tests use torchrun with HCCL backend

✅ **Comprehensive Coverage**
- All NPUCommunicator methods tested
- Edge cases and error handling covered
- Real hardware validation included

✅ **Easy to Run**
- Simple shell script interface (`./run_tests.sh`)
- Pytest markers for filtering
- Clear documentation (4 guide files)

✅ **CI/CD Ready**
- Can run unit tests anywhere
- Integration tests skip gracefully without NPU
- Coverage reporting included

## Test Statistics

### Current Status
- **Total test cases**: 28+ (NPUCommunicator only)
- **Unit tests**: 20+ (no hardware required)
- **Integration tests**: 8+ (NPU required)
- **Components tested**: 1/6 (NPUCommunicator)
- **Current coverage**: ~15% of codebase

### Target Goals
- **Components to test**: 
  - ✅ NPUCommunicator (distributed)
  - 🔲 NPUPlatform
  - 🔲 Attention backends (2 backends)
  - 🔲 NPU Worker & Model Runner
  - 🔲 Sampler
- **Target coverage**: >80% for all components
- **Estimated total tests**: 100+ when complete

## Next Steps

1. **Run unit tests** to verify setup:
   ```bash
   ./run_tests.sh unit
   ```

2. **On NPU hardware**, run integration tests:
   ```bash
   ./run_tests.sh integration
   ```

3. **Add tests for remaining components** (priority order):
   
   **High Priority:**
   - `tests/unit/test_platform.py` - NPUPlatform tests
   - `tests/unit/v1/worker/test_npu_worker.py` - NPUWorker tests
   - `tests/unit/v1/worker/test_npu_model_runner.py` - Model runner tests
   
   **Medium Priority:**
   - `tests/unit/attention/backends/test_attention.py` - Attention backend tests
   - `tests/unit/attention/backends/test_mla.py` - MLA tests
   
   **Lower Priority:**
   - `tests/unit/v1/sample/test_sampler.py` - Sampler tests
   
   **Template to follow:** Use `tests/unit/distributed/test_communicator.py` as reference

4. **Integrate into CI/CD pipeline**:
   - Unit tests run on all commits (no NPU required)
   - Integration tests run on NPU hardware nodes
   - Coverage reports generated automatically

## Documentation

- **README.md** - Overview and basic usage
- **QUICKSTART.md** - Quick reference for common commands
- **TESTING.md** - Detailed testing guide and best practices
- **STRUCTURE.md** - Test directory structure and conventions
