# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pandas as pd
import pytest

from prophet import Prophet, diagnostics


def test_mdape_with_custom_index():
    """
    Test that mdape function works correctly when df['horizon'] has a custom index.
    
    This test exposes a bug where mdape passes df['horizon'] instead of 
    df['horizon'].values to rolling_median_by_h, unlike all other metric functions.
    
    When df has a non-default index (e.g., after filtering or concatenation),
    passing a Series with custom index can cause issues in rolling_median_by_h
    because it expects numpy arrays.
    """
    # Create test data that would come from cross_validation
    # Simulate a dataframe with custom index (non-sequential)
    df = pd.DataFrame({
        'y': [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0],
        'yhat': [10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5],
        'horizon': pd.Timedelta('1 days'),
    })
    
    # Create a non-default index to simulate real cross-validation output
    df.index = [5, 6, 7, 8, 9, 15, 16, 17, 18, 19]
    
    # Test with w >= 0 (the code path that calls rolling_median_by_h)
    w = 3
    
    # This should work without errors
    result = diagnostics.mdape(df, w)
    
    # Verify the result has the expected structure
    assert 'horizon' in result.columns
    assert 'mdape' in result.columns
    assert len(result) > 0
    
    # Verify that the result matches what we'd expect from other metrics
    # All metrics should behave consistently
    result_mae = diagnostics.mae(df, w)
    
    # Both should have the same horizon values
    assert len(result) == len(result_mae)


def test_mdape_consistency_with_other_metrics():
    """
    Test that mdape handles Series vs array consistently with other metrics.
    
    This test verifies that mdape produces valid results and doesn't fail
    due to passing a pandas Series instead of numpy array to rolling_median_by_h.
    """
    # Create synthetic cross-validation-like data
    dates = pd.date_range('2020-01-01', periods=20, freq='D')
    cutoff = pd.Timestamp('2020-01-10')
    
    df = pd.DataFrame({
        'ds': dates,
        'y': np.linspace(10, 20, 20) + np.random.randn(20) * 0.5,
        'yhat': np.linspace(10, 20, 20) + np.random.randn(20) * 0.3,
        'cutoff': [cutoff] * 20,
    })
    
    df['horizon'] = df['ds'] - df['cutoff']
    
    # Filter to create non-sequential index
    df = df[df['ds'] > cutoff].reset_index(drop=True)
    
    # Further manipulate to create a challenging index scenario
    df.index = range(100, 100 + len(df))
    
    # Test mdape with different window sizes
    for w in [1, 3, 5]:
        result = diagnostics.mdape(df, w)
        
        # Should not raise an error and should return valid data
        assert result is not None
        assert len(result) > 0
        assert 'mdape' in result.columns
        assert not result['mdape'].isna().all()


def test_mdape_matches_expected_behavior():
    """
    Test that mdape calculates the correct values.
    
    This verifies the fix doesn't change the mathematical correctness,
    only fixes the data type issue.
    """
    # Simple test case with known values
    df = pd.DataFrame({
        'y': [10.0, 20.0, 30.0, 40.0],
        'yhat': [11.0, 22.0, 33.0, 44.0],
        'horizon': pd.Timedelta('1 days'),
    })
    
    # Test with w < 0 (direct calculation, no rolling)
    result = diagnostics.mdape(df, -1)
    
    # Calculate expected MDAPE manually
    expected_ape = np.abs((df['y'] - df['yhat']) / df['y'])
    
    assert len(result) == len(df)
    assert np.allclose(result['mdape'].values, expected_ape.values)
