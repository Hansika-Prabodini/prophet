# Bug Fix Summary: mdape Function Inconsistency

## Bug Description

**Location:** `python/prophet/diagnostics.py`, line 639

**Issue:** The `mdape` (Median Absolute Percent Error) function was passing a pandas Series (`df['horizon']`) to the `rolling_median_by_h` function, while all other performance metric functions correctly pass a numpy array (`df['horizon'].values`).

## Root Cause

In the `mdape` function, the call to `rolling_median_by_h` was:
```python
return rolling_median_by_h(
    x=ape.values, h=df['horizon'], w=w, name='mdape'  # BUG: df['horizon'] is a Series
)
```

While all other metric functions (mse, rmse, mae, mape, smape, coverage) correctly use:
```python
return rolling_mean_by_h(
    x=values.values, h=df['horizon'].values, w=w, name='metric'  # CORRECT: .values
)
```

## Why This Is a Problem

The `rolling_median_by_h` function internally uses array indexing operations that expect a numpy array. When a pandas Series with a non-default index is passed (e.g., after filtering or slicing a DataFrame), this can cause:

1. **Index confusion**: The Series retains its original DataFrame index, which may not be sequential (e.g., [5, 6, 7, ...] instead of [0, 1, 2, ...])
2. **Incorrect indexing**: At line 518 in `rolling_median_by_h`:
   ```python
   next_idx_to_add = np.array(h == h_i).argmax() - 1
   ```
   When `h` is a Series, the boolean comparison returns a Series with the original index, leading to potential indexing errors when trying to access `x[next_idx_to_add]` by position.

## The Fix

Changed line 639 in `python/prophet/diagnostics.py` from:
```python
return rolling_median_by_h(
    x=ape.values, h=df['horizon'], w=w, name='mdape'
)
```

To:
```python
return rolling_median_by_h(
    x=ape.values, h=df['horizon'].values, w=w, name='mdape'
)
```

This ensures consistency with all other performance metric functions and prevents issues with non-sequential DataFrame indices.

## Test Case

A comprehensive test was added in `python/prophet/tests/test_mdape_bug.py` that:

1. Creates cross-validation results with a non-default index (by filtering rows)
2. Calls `performance_metrics` with the 'mdape' metric
3. Verifies that:
   - No errors are raised
   - The mdape metric is computed successfully
   - The results are valid (positive values, no NaNs)

### Test Scenarios

1. **test_mdape_with_non_default_index**: Tests mdape with filtered DataFrame (non-sequential index starting at 5)
2. **test_mdape_consistency_with_other_metrics**: Tests that mdape works alongside other metrics with filtered data

## Impact

- **Before fix**: The mdape metric could fail or produce incorrect results when used with filtered or sliced cross-validation DataFrames
- **After fix**: The mdape metric behaves consistently with all other metrics and handles non-default indices correctly

## Files Changed

1. `python/prophet/diagnostics.py` (line 639) - Applied the fix
2. `python/prophet/tests/test_mdape_bug.py` - Added regression tests
