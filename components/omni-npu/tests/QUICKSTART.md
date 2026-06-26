# Quick Start - omni-npu Test Suite

## TL;DR

**Current Status:** NPUCommunicator tests complete. Use as template for other components.

```bash
# Install dependencies
pip install -e ".[test]"

# Run unit tests (no NPU required)
./run_tests.sh unit

# Run integration tests (requires NPU)
./run_tests.sh integration

# Run all tests
./run_tests.sh all
```

## Test Organization

Tests mirror the source code structure in `src/omni_npu/`:

```
tests/
├── unit/                           # No NPU hardware required ✅
│   ├── __init__.py
│   └── distributed/
│       ├── __init__.py
│       └── test_communicator.py   # NPUCommunicator unit tests
└── integration/                    # Requires NPU hardware ⚠️
    ├── __init__.py
    └── distributed/
        ├── __init__.py
        └── test_communicator.py   # NPUCommunicator integration tests
```

## Common Commands

| Command | Description |
|---------|-------------|
| `./run_tests.sh unit` | Run unit tests only (no NPU) |
| `./run_tests.sh integration` | Run integration tests (needs NPU) |
| `./run_tests.sh all` | Run all tests |
| `pytest tests/unit/ -v` | Run unit tests with pytest |
| `pytest -m unit` | Run tests marked as unit |
| `pytest -m "not integration"` | Skip integration tests |
| `pytest --cov=omni_npu` | Run with coverage |

## What Gets Tested

### Currently Implemented (NPUCommunicator)

**Unit Tests (20+ tests)**
- ✅ Initialization with/without torch.npu
- ✅ All collective operations (all_reduce, all_gather, etc.)
- ✅ Point-to-point operations (send, recv)
- ✅ Edge cases and error handling

**Integration Tests (8+ tests)**
- ✅ Real NPU device operations
- ✅ Multi-device distributed communication
- ✅ Memory management

### To Be Implemented

- 🔲 **NPUPlatform** - Device management, platform operations
- 🔲 **Attention Backends** - Attention mechanisms, MLA
- 🔲 **NPU Worker & Runner** - Model execution, batch processing
- 🔲 **Sampler** - Sampling strategies

## Troubleshooting

**"NPU hardware not available"**
→ Integration tests are skipped automatically. This is normal on systems without NPU.

**"pytest: command not found"**
→ Run: `pip install -e ".[test]"`

**"ModuleNotFoundError: No module named 'omni_npu'"**
→ Run: `pip install -e .`

## For CI/CD

**Without NPU:**
```bash
pytest tests/unit/ -v --cov=omni_npu
```

**With NPU:**
```bash
pytest tests/ -v --cov=omni_npu
```

## Documentation

- `README.md` - Overview and usage
- `TESTING.md` - Detailed testing guide
- `TEST_SUMMARY.md` - Complete test suite summary
