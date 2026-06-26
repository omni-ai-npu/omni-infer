# Test Directory Structure

## Overview

The test directory structure mirrors the source code structure in `src/omni_npu/` to make it easy to find tests for specific components.

## Current Structure

```
tests/
├── __init__.py
├── README.md                    # Test documentation and usage
├── QUICKSTART.md               # Quick reference guide
├── TESTING.md                  # Detailed testing guide
├── STRUCTURE.md                # This file
│
├── unit/                       # Unit tests (no NPU hardware required)
│   ├── __init__.py
│   └── distributed/
│       ├── __init__.py
│       └── test_communicator.py    # Tests for src/omni_npu/distributed/communicator.py
│
└── integration/                # Integration tests (requires NPU hardware)
    ├── __init__.py
    └── distributed/
        ├── __init__.py
        └── test_communicator.py    # Tests for src/omni_npu/distributed/communicator.py
```

## Source Code Structure (for reference)

```
src/omni_npu/
├── __init__.py
├── platform.py
├── vllm_plugin.py
│
├── distributed/
│   ├── __init__.py
│   └── communicator.py         # ← Tested by tests/*/distributed/test_communicator.py
│
├── attention/
│   └── backends/
│       ├── __init__.py
│       ├── attention.py
│       ├── attention_dummy_builder.py
│       └── mla.py
│
└── v1/
    ├── __init__.py
    ├── sample/
    │   └── sampler.py
    └── worker/
        ├── __init__.py
        ├── npu_model_runner.py
        └── npu_worker.py
```

## Adding New Tests

When adding tests for a new component, follow this structure:

### For a new source file: `src/omni_npu/foo/bar.py`

Create corresponding test files:

1. **Unit tests**: `tests/unit/foo/test_bar.py`
   ```bash
   mkdir -p tests/unit/foo
   touch tests/unit/foo/__init__.py
   touch tests/unit/foo/test_bar.py
   ```

2. **Integration tests** (if needed): `tests/integration/foo/test_bar.py`
   ```bash
   mkdir -p tests/integration/foo
   touch tests/integration/foo/__init__.py
   touch tests/integration/foo/test_bar.py
   ```

## Test File Naming Convention

- **Source file**: `src/omni_npu/distributed/communicator.py`
- **Unit test**: `tests/unit/distributed/test_communicator.py`
- **Integration test**: `tests/integration/distributed/test_communicator.py`

Pattern: `test_<source_filename>.py`

## Future Test Structure

As more components are added, the structure will grow like this:

```
tests/
├── unit/
│   ├── distributed/
│   │   └── test_communicator.py
│   ├── attention/
│   │   └── backends/
│   │       ├── test_attention.py
│   │       └── test_mla.py
│   ├── v1/
│   │   ├── sample/
│   │   │   └── test_sampler.py
│   │   └── worker/
│   │       ├── test_npu_model_runner.py
│   │       └── test_npu_worker.py
│   └── test_platform.py
│
└── integration/
    ├── distributed/
    │   └── test_communicator.py
    ├── attention/
    │   └── backends/
    │       └── test_attention.py
    └── v1/
        └── worker/
            └── test_npu_worker.py
```

## Benefits of This Structure

1. **Easy to find tests**: If you're working on `src/omni_npu/distributed/communicator.py`, you know tests are in `tests/*/distributed/test_communicator.py`

2. **Clear organization**: Unit and integration tests are separated but follow the same structure

3. **Scalable**: As the codebase grows, the test structure grows naturally

4. **IDE-friendly**: Most IDEs can easily navigate between source and test files

5. **Consistent**: All developers follow the same pattern

## Running Tests

```bash
# All unit tests
pytest tests/unit/ -v

# All integration tests
pytest tests/integration/ -v

# Specific component unit tests
pytest tests/unit/distributed/ -v

# Specific component integration tests
pytest tests/integration/distributed/ -v

# Specific test file
pytest tests/unit/distributed/test_communicator.py -v
```
