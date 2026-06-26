# omni-npu Test Suite

This directory contains comprehensive tests for the omni-npu package, separated into unit tests and integration tests.

**Current Status:** NPUCommunicator tests implemented and serving as template for other components.

## Test Structure

Tests are organized to mirror the source code structure in `src/omni_npu/`.

### Unit Tests (`tests/unit/`)
**Do NOT require NPU hardware** - use mocking to verify logic and API contracts.

```
tests/unit/
├── __init__.py
├── distributed/
│   ├── __init__.py
│   └── test_communicator.py    # ✅ NPUCommunicator tests
├── attention/                   # 🔲 TODO
│   └── backends/
├── v1/                          # 🔲 TODO
│   ├── sample/
│   └── worker/
└── test_platform.py             # 🔲 TODO: NPUPlatform tests
```

**Currently Implemented:**
- ✅ NPUCommunicator: Initialization, collective ops, point-to-point ops, edge cases
- Uses mocks to avoid requiring actual NPU hardware

**To Be Implemented:**
- 🔲 NPUPlatform: Device management, memory operations
- 🔲 Attention backends: Attention mechanisms, MLA
- 🔲 NPU Worker & Model Runner: Batch processing, model execution
- 🔲 Sampler: Sampling strategies

### Integration Tests (`tests/integration/`)
**REQUIRE NPU hardware** - verify end-to-end functionality with real devices.

```
tests/integration/
├── __init__.py
├── distributed/
│   ├── __init__.py
│   └── test_communicator.py    # ✅ NPUCommunicator integration tests
├── attention/                   # 🔲 TODO
│   └── backends/
└── v1/                          # 🔲 TODO
    └── worker/
```

**Currently Implemented:**
- ✅ NPUCommunicator: Device operations, multi-device communication (with torchrun)
- Automatically skipped if NPU hardware is not available

**To Be Implemented:**
- 🔲 Attention backends: End-to-end attention with real NPU
- 🔲 NPU Worker: End-to-end model inference workflows

## Running Tests

### Install test dependencies

```bash
pip install -e ".[test]"
```

### Run unit tests only (no NPU required)

```bash
pytest tests/unit/
```

### Run integration tests (requires NPU hardware)

```bash
pytest tests/integration/
```

### Run all tests

```bash
pytest tests/
```

### Run with coverage

```bash
pytest --cov=omni_npu --cov-report=html --cov-report=term tests/
```

### Run specific test file

```bash
pytest tests/unit/distributed/test_communicator.py
```

### Run specific test

```bash
pytest tests/unit/distributed/test_communicator.py::TestNPUCommunicatorUnit::test_init_with_torch_npu_available
```

### Run multi-device integration tests (requires 2+ NPUs)

```bash
torchrun --nproc_per_node=2 -m pytest tests/integration/distributed/test_communicator.py::TestNPUCommunicatorMultiDevice
```

## CI/CD Integration

For CI/CD pipelines without NPU hardware:
```bash
# Run only unit tests
pytest tests/unit/ -v
```

For systems with NPU hardware:
```bash
# Run all tests including integration
pytest tests/ -v
```

## Multi-Container UT (Multi-Docker)

This workflow runs UT across multiple containers in parallel, splits tests by duration,
collects per-container coverage/duration artifacts, and optionally merges reports.
By default, it starts 4 containers on an 8-NPU host, with 2 NPUs per container.

If you need to change container mapping, Ascend device allocation, split config, or
per-container test args, edit `tests/ut_config.sh`. The commands below remain the same.

### 1. Start containers

```bash
cd tests
bash run_docker.sh <docker_image>
```

### 2. Run tests in parallel

```bash
cd tests
bash concurrent_test_run_multi_docker.sh <omni-npu_root>
```

This will:
- Sync the repo into each container
- Run pytest with duration-based splits
- Write per-container logs to `tests/install_logs/`
- Write per-container durations to `tests/test_durations_from_dockers/`
- Copy per-container coverage files to `tests/coverage_from_dockers/`
- Merge coverage and durations files(optional) inside a container and copy reports back to host


### 3. Optional: check duration balance

```bash
python3 tests/ut_CI_check/ut_CI_check_durations_balance.py \
  --dir tests/test_durations_from_dockers
```

If containers are imbalanced, please update `tests/test_durations_v1.json` using the merged
`tests/test_durations_merged.json`.

### 4. Optional: parse logs and coverage

```bash
python3 tests/ut_CI_check/ut_CI_parse_logs.py \
  --log tests/install_logs/DT_1.log \
  --log tests/install_logs/DT_2.log \
  --log tests/install_logs/DT_3.log \
  --log tests/install_logs/DT_4.log \
  --merged-log tests/install_logs/merged_log.log \
  --known /path/to/known_fails_cases.txt

python3 tests/ut_CI_check/ut_CI_cover_rate_check.py \
  --report tests/coverage_from_dockers/coverage_report.txt \
  --min 60
```

### 4.1 Optional: incremental coverage (diff-cover)

Incremental coverage is currently supported only in the multi-container UT workflow.
The implementation lives in `tests/ut_CI_check/ut_diff_cov.py`, and it only checks
the **staged** changes in the working tree (uses `git diff --cached`).

If you want to run incremental coverage locally, follow the same idea as `tests/ut_CI_check/ut_diff_cov.py`:
1. get the diff file (if not provided)
2. run diff-cover

Example:

```bash
python3 tests/ut_CI_check/ut_diff_cov.py \
  --repo-root /path/to/omni-npu \
  --coverage-xml /path/to/coverage.xml \
  --old-prefix /workspace/omniinfer/components/omni-npu \
  --out-html /path/to/diffcov.html \
  --out-txt /path/to/diffcov.txt
```

### 5. Cleanup containers

```bash
cd tests
bash rm_docker.sh
```
## Notes

- **Unit tests** use mocking and can run anywhere
- **Integration tests** are automatically skipped if NPU is not available
- Multi-device tests require `torchrun` and multiple NPU devices
- Tests verify correct delegation to torch.distributed APIs
