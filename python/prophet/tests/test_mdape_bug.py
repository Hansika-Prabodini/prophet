# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pandas as pd
import pytest

from prophet import diagnostics


class TestMdapeBug:
    """Test for the mdape bug where df['horizon'] was passed instead of df['horizon'].values"""
    
    def test_mdape_with_non_default_index(self):
        """Test that mdape works correctly with a DataFrame that has a non-default index.
        
        The bug occurs when a pandas Series is passed to rolling_median_by_h instead of
        a numpy array. This manifests when the DataFrame has a non-sequential index.
        """
        # Create a sample cross-validation results DataFrame
        # Simulate what comes out of cross_validation
        np.random.seed(42)
        n = 50
        
        # Create data with multiple horizons
        horizons = pd.to_timedelta([1, 2, 3, 4, 5] * 10, unit='D')
        cutoffs = pd.date_range('2020-01-01', periods=n, freq='D')
        
        df_cv = pd.DataFrame({
            'ds': cutoffs + horizons,
            'yhat': np.random.randn(n) * 10 + 100,
            'y': np.random.randn(n) * 10 + 100,
            'cutoff': cutoffs,
            'yhat_lower': np.random.randn(n) * 10 + 90,
            'yhat_upper': np.random.randn(n) * 10 + 110,
        })
        
        # Filter the dataframe to create a non-sequential index
        # This simulates what might happen in real usage
        df_cv_filtered = df_cv.iloc[5:].copy()  # Skip first 5 rows, index now starts at 5
        
        # Calculate performance metrics including mdape
        # This should not raise an error and should produce correct results
        try:
            df_p = diagnostics.performance_metrics(
                df_cv_filtered,
                metrics=['mae', 'mdape', 'mse'],
                rolling_window=0.1
            )
            
            # Check that results are reasonable
            assert 'mdape' in df_p.columns
            assert 'mae' in df_p.columns
            assert 'mse' in df_p.columns
            assert len(df_p) > 0
            assert not df_p['mdape'].isna().all()
            
            # The mdape values should be positive percentages
            assert (df_p['mdape'] >= 0).all()
            
        except (IndexError, KeyError, ValueError) as e:
            # If this fails, it's likely due to the bug
            pytest.fail(f"mdape failed with non-default index: {e}")
    
    def test_mdape_consistency_with_other_metrics(self):
        """Test that mdape behaves consistently with other metrics when handling indices."""
        np.random.seed(42)
        n = 30
        
        # Create simple cross-validation results
        horizons = pd.to_timedelta([1, 2, 3] * 10, unit='D')
        cutoffs = pd.date_range('2020-01-01', periods=n, freq='D')
        
        df_cv = pd.DataFrame({
            'ds': cutoffs + horizons,
            'yhat': np.random.randn(n) * 5 + 50,
            'y': np.random.randn(n) * 5 + 50,
            'cutoff': cutoffs,
            'yhat_lower': np.random.randn(n) * 5 + 45,
            'yhat_upper': np.random.randn(n) * 5 + 55,
        })
        
        # Test with filtered dataframe (non-sequential index)
        df_cv_filtered = df_cv.iloc[3:].copy()
        
        # All metrics should work without errors
        df_all_metrics = diagnostics.performance_metrics(
            df_cv_filtered,
            rolling_window=0.2
        )
        
        # Check that mdape was computed successfully
        assert 'mdape' in df_all_metrics.columns
        assert not df_all_metrics['mdape'].isna().all()
        assert len(df_all_metrics) > 0
