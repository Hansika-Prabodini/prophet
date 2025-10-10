# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pandas as pd
import pytest

from prophet import diagnostics


class TestRollingMeanByH:
    """Test for rolling_mean_by_h bug fix."""
    
    def test_rolling_mean_by_h_correct_accumulator_subtraction(self):
        """
        Test that rolling_mean_by_h correctly computes rolling means.
        
        The bug was that the function was using trailing_i to index into xs and ns
        arrays when subtracting from accumulators, but trailing_i is the output array
        index, not the input array index. This caused incorrect values to be
        subtracted from the rolling window.
        
        This test creates a simple scenario where we can calculate the expected
        rolling mean manually and verify the function produces the correct result.
        """
        # Create test data with distinct horizons
        # Horizon 1: values [10, 10] (mean = 10, count = 2)
        # Horizon 2: values [20, 20, 20] (mean = 20, count = 3)
        # Horizon 3: values [30, 30] (mean = 30, count = 2)
        x = np.array([10, 10, 20, 20, 20, 30, 30], dtype=float)
        h = np.array([1, 1, 2, 2, 2, 3, 3])
        w = 3  # Window size of 3 samples
        name = 'test_metric'
        
        result = diagnostics.rolling_mean_by_h(x, h, w, name)
        
        # Expected results:
        # Starting from right (horizon 3):
        # - At h=3: accumulator has 2 samples (30, 30), n_sum=2 < w=3, continue
        # - At h=2: accumulator has 5 samples (20,20,20,30,30), n_sum=5 >= w=3
        #   Output for h=3: mean of (20,20,30,30,30) with weighted removal = (20*5 - 2*20/3) / 3
        #   Actually the algorithm computes a weighted mean
        #   Actually let's recalculate:
        #   When i points to h=2: xs[2]=60 (sum), ns[2]=3
        #   x_sum = 60 + 60 = 120, n_sum = 3 + 2 = 5
        #   excess_n = 5 - 3 = 2
        #   excess_x = 2 * 60 / 3 = 40
        #   res_x = (120 - 40) / 3 = 80 / 3 = 26.666...
        #   Then subtract: x_sum -= xs[trailing_i], n_sum -= ns[trailing_i]
        #   With the bug: trailing_i starts at 2 (same as i), so subtracts xs[2]=60, ns[2]=3
        #   x_sum = 120 - 60 = 60, n_sum = 5 - 3 = 2
        
        # Let me create a simpler test case
        # Each horizon has exactly 1 sample for clarity
        x_simple = np.array([10., 20., 30., 40.])
        h_simple = np.array([1, 2, 3, 4])
        w_simple = 2
        
        result_simple = diagnostics.rolling_mean_by_h(x_simple, h_simple, w_simple, 'test')
        
        # Expected: starting from right
        # i=3 (h=4): x_sum=40, n_sum=1, not enough
        # i=2 (h=3): x_sum=40+30=70, n_sum=2, enough!
        #   excess_n = 2-2 = 0, excess_x = 0
        #   res[2] = 70/2 = 35  (mean of h=3 and h=4)
        #   Now subtract - WITH BUG: xs[trailing_i=2]=30, ns[2]=1
        #   x_sum = 70-30=40, n_sum=2-1=1
        # i=1 (h=2): x_sum=40+20=60, n_sum=1+1=2
        #   excess_n = 0, excess_x = 0
        #   res[1] = 60/2 = 30  (mean of h=2 and... wait this is wrong!)
        #   The bug causes wrong accumulator state
        
        # With CORRECT code (should subtract xs[i] and ns[i]):
        # i=3: x_sum=40, n_sum=1
        # i=2: x_sum=70, n_sum=2, output res[2]=35
        #   Subtract xs[2]=30, ns[2]=1 -> x_sum=40, n_sum=1
        # i=1: x_sum=40+20=60, n_sum=2, output res[1]=30
        #   Subtract xs[1]=20, ns[1]=1 -> x_sum=40, n_sum=1
        # i=0: x_sum=40+10=50, n_sum=2, output res[0]=25
        
        # So expected horizons: [2, 3, 4] with means [30, 35, wait...]
        # Actually the algorithm outputs right-aligned means
        # Let me recalculate more carefully
        
        # Expected output should be horizons [2, 3, 4] with values [30, 35, and one more]
        # Actually let me just check that the result makes sense
        assert len(result_simple) == 3  # Should have 3 outputs for w=2 with 4 horizons
        assert result_simple['horizon'].tolist() == [2, 3, 4]
        
        # The correct mean for horizon 4 (rightmost) should be mean of h=3,4 = (30+40)/2 = 35
        np.testing.assert_almost_equal(result_simple[result_simple['horizon'] == 4]['test'].values[0], 35.0)
        
        # The correct mean for horizon 3 should be mean of h=2,3 = (20+30)/2 = 25  
        np.testing.assert_almost_equal(result_simple[result_simple['horizon'] == 3]['test'].values[0], 25.0)
        
        # The correct mean for horizon 2 should be mean of h=1,2 = (10+20)/2 = 15
        np.testing.assert_almost_equal(result_simple[result_simple['horizon'] == 2]['test'].values[0], 15.0)
