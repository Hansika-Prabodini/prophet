# MDAPE Function Bug Fix Test Documentation

## Overview
This document describes the test case added for the `mdape` function bug fix in `prophet/diagnostics.py`.

## Bug Description
The `mdape` function was passing `df['horizon']` (a pandas Series) instead of `df['horizon'].values` (a numpy array) to the `rolling_median_by_h` function. This was inconsistent with all other metric functions and could cause issues when DataFrames had non-default indices.

## Fix Applied
Changed line 668 in `python/prophet/diagnostics.py`:
```python
# Before
x=ape.values, h=df['horizon'], w=w, name='mdape'

# After
x=ape.values, h=df['horizon'].values, w=w, name='mdape'
```

## Test Case: `test_mdape_with_non_default_index`

### Purpose
Verify that the `mdape` function handles DataFrames with non-default indices correctly after the bug fix.

### Test Setup
- Creates a cross-validation style DataFrame with:
  - `y`: actual values (10.0 to 17.0)
  - `yhat`: predicted values 
  - `horizon`: time deltas (1 to 4 days)
  - `cutoff`: date range starting from 2020-01-01
- Sets a non-default index (RangeIndex starting at 100) to expose potential issues

### Test Cases
1. **Window sizes 1, 2, and 4**: Tests rolling median calculations
   - Verifies presence of 'horizon' and 'mdape' columns
   - Checks that all mdape values are finite and non-negative

2. **No rolling window (w=-1)**: Tests without rolling aggregation
   - Verifies output length matches input length
   - Checks for required columns

### Expected Behavior
- Function should work correctly regardless of DataFrame index
- All mdape values should be finite and non-negative
- Output should contain 'horizon' and 'mdape' columns

## Testing
Run the test with:
```bash
pytest python/prophet/tests/test_diagnostics.py::TestPerformanceMetrics::test_mdape_with_non_default_index -v
```
