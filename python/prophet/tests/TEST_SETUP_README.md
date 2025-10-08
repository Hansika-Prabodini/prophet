# Setup.py Unit Tests

## Overview

This document describes the unit tests for the Stan model compilation functionality in `setup.py`. These tests were created to address the issue of missing unit tests for model compilation, particularly for edge cases that could lead to build failures at runtime.

## Test File

- **File**: `python/prophet/tests/test_setup.py`
- **Target**: Lines 162-173 of `setup.py` (model compilation code)

## Test Coverage

The test suite provides comprehensive coverage for:

### 1. TestBuildCmdStanModel
Tests the main `build_cmdstan_model` function:
- ✅ Successful model compilation
- ✅ Missing Stan file detection
- ✅ Compilation failure handling
- ✅ Missing executable after compilation
- ✅ CmdStan installation failures
- ✅ File copy operation failures
- ✅ Windows-specific compilation behavior
- ✅ Cleanup process and non-critical error handling

### 2. TestPruneCmdStan
Tests the `prune_cmdstan` function:
- ✅ Correct restructuring of cmdstan directory
- ✅ Binary file filtering
- ✅ TBB directory preservation
- ✅ Missing directory handling

### 3. TestInstallCmdStanDeps
Tests the `install_cmdstan_deps` function:
- ✅ Repackaging scenarios
- ✅ Installation failure detection
- ✅ Windows toolchain installation

### 4. TestMaybeInstallCmdStanToolchain
Tests the `maybe_install_cmdstan_toolchain` function:
- ✅ Already installed toolchain detection
- ✅ Successful installation
- ✅ Legacy cmdstanpy compatibility

### 5. TestHelperFunctions
Tests utility functions:
- ✅ `repackage_cmdstan` with various environment variables
- ✅ `get_backends_from_env` with different configurations

### 6. TestBuildModels
Tests the `build_models` function:
- ✅ CMDSTANPY backend support
- ✅ PYSTAN backend error handling

## Running the Tests

### Run all setup tests:
```bash
cd python
pytest prophet/tests/test_setup.py -v
```

### Run a specific test class:
```bash
pytest prophet/tests/test_setup.py::TestBuildCmdStanModel -v
```

### Run a specific test:
```bash
pytest prophet/tests/test_setup.py::TestBuildCmdStanModel::test_successful_model_compilation -v
```

### Run with coverage:
```bash
pytest prophet/tests/test_setup.py --cov=setup --cov-report=html -v
```

## Key Features

### Mocking
All tests use mocking to avoid:
- Actual Stan model compilation during tests
- Real CmdStan installation
- File system modifications
- Network operations

This ensures:
- Tests run quickly
- Tests are reliable and repeatable
- No side effects on the test environment

### Edge Case Coverage
The tests specifically address edge cases identified in the issue:
1. Missing Stan model file
2. Invalid Stan syntax (compilation failure)
3. Missing CmdStan dependencies
4. File permission issues
5. Path-related errors
6. Platform-specific behavior (Windows vs. Unix)
7. Cleanup failures

### Error Message Validation
Tests verify that appropriate error messages are raised using pytest's `match` parameter:
```python
with pytest.raises(RuntimeError, match="Failed to compile Stan model"):
    setup.build_cmdstan_model(target_dir)
```

## Integration with Existing Tests

The new test file:
- Follows the existing test structure in `prophet/tests/`
- Uses the same pytest conventions
- Integrates with the existing `conftest.py`
- Can be run as part of the full test suite

## Continuous Integration

These tests should be included in CI/CD pipelines to:
- Catch build failures early
- Ensure changes to setup.py don't break compilation
- Validate cross-platform compatibility

## Maintenance

When modifying `setup.py`, ensure:
1. Tests are updated to reflect changes
2. New functionality includes corresponding tests
3. Edge cases are considered and tested
4. Mocks are kept in sync with actual function signatures

## Test Results

All tests should pass with proper mocking. If tests fail:
1. Check that setup.py has not changed function signatures
2. Verify mock configurations match current implementation
3. Ensure all dependencies are properly mocked
4. Check for platform-specific issues

## Future Enhancements

Potential areas for additional testing:
- Performance testing for compilation speed
- Integration tests with actual Stan compilation (marked as slow)
- Memory usage validation during compilation
- Multi-threading/parallel compilation scenarios
