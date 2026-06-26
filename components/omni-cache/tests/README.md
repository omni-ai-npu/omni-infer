# omni-cache Test Suite

This directory contains tests for the omni-cache package.

## Test Structure

```
tests/
├── unit_tests/      # Unit tests (mocked dependencies, fast)
├── module_tests/    # Module tests (component-level integration)
├── benchmarks/      # Performance benchmarks
└── run_tests.sh     # Test runner script
```

- **Unit tests**: Test individual functions/classes with mocked dependencies
- **Module tests**: Test module-level functionality with real components where possible

## Quick Start

```bash
# Install dependencies
pip install -e ".[test]"

# Run all tests with coverage
./run_tests.sh

# Run all tests without coverage
./run_tests.sh --no-cov
```

## Running Tests

### Using run_tests.sh

```bash
./run_tests.sh                            # Run all tests (with hugetlbfs setup)
./run_tests.sh --no-cov                   # Run all without coverage
./run_tests.sh --unit                     # Run unit tests with coverage
./run_tests.sh --no-cov --module          # Run module tests without coverage
./run_tests.sh --no-hugetlbfs --module    # Run module tests without hugetlbfs setup
./run_tests.sh -k "test_kv"           # Run tests matching pattern
./run_tests.sh -- -x                  # Stop on first failure
./run_tests.sh --help                 # Show help
```

### Using pytest directly

```bash
# Run all tests with coverage
pytest unit_tests/ module_tests/ --cov=omni_cache --cov-report=html -v

# Run unit tests only
pytest unit_tests/ -v

# Run module tests only
pytest module_tests/ -v

# Run specific test file
pytest unit_tests/core/test_kv_cache.py -v

# Run specific test
pytest unit_tests/core/test_kv_cache.py::TestKVCache::test_init -v
```

## Test Coverage

After running tests with coverage, a detailed HTML report is generated in `htmlcov/`.

### Coverage Report Files

```
htmlcov/
├── index.html              # Main coverage summary page
├── class_index.html        # Coverage by class
├── function_index.html     # Coverage by function
├── status.json             # Coverage data in JSON format
├── z_*.html                # Per-source-file coverage details
└── style_*.css, *.js       # Supporting assets
```

### Viewing Coverage Report

**VSCode (Recommended):**
1. Install the [Live Preview](https://marketplace.visualstudio.com/items?itemName=ms-vscode.live-server) extension
2. Open `htmlcov/index.html` in VSCode
3. Click "Show Preview" (or right-click → "Show Preview")
4. Navigate through source files to see line-by-line coverage

**Python HTTP Server:**
```bash
# Start a simple HTTP server in htmlcov directory
cd htmlcov && python -m http.server 8080
# Then open http://localhost:8080 in your browser
```

**Direct File:**
```bash
# Copy htmlcov/ to local machine and open in browser
scp -r server:/path/to/htmlcov ./
# Then open htmlcov/index.html in browser
```

### Coverage Report Features

- **index.html**: Shows overall coverage percentage and module breakdown
- **Per-file reports**: Click any file to see exactly which lines are covered (green) or missed (red)
- **Filtering**: Use the filter box to find specific files/modules
- **Hide covered**: Check "hide covered" to focus only on files needing more tests


## CI/CD Integration

```yaml
- name: Run tests
  run: |
    pip install -e ".[test]"
    ./run_tests.sh
```

## Writing New Tests

### Unit Tests

1. Create test file in `unit_tests/` following the source structure
2. Use mocking for vllm, torch_npu, triton, and llm_datadist dependencies (provided by conftest.py)
3. Test logic, not hardware functionality

Example:
```python
def test_my_feature(self, mock_vllm, mock_torch_npu):
    # Dependencies are mocked via conftest.py fixtures
    pass
```

### Module Tests

1. Create test file in `module_tests/` for component-level integration tests
2. Use real components where possible, mock only external dependencies
3. Test module interactions and data flow


## Test Dependencies

- `pytest`: Test framework
- `pytest-cov`: Coverage collection

Install with:
```bash
pip install -e ".[test]"
```

## Troubleshooting

### "pytest: command not found"
Run: `pip install -e ".[test]"`

### "ModuleNotFoundError: No module named 'omni_cache'"
Run: `pip install -e .`

### Import errors related to vllm/torch_npu/triton
These are mocked in conftest.py for unit tests. If you see import errors:
1. Check that conftest.py is in the tests directory
2. Ensure pytest is running from the correct directory
3. Verify PYTHONPATH includes the source directory