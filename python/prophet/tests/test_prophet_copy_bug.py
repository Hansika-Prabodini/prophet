# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pandas as pd
import pytest

from prophet import Prophet
from prophet.diagnostics import prophet_copy


class TestProphetCopyBug:
    def test_prophet_copy_with_cutoff_before_history(self, daily_univariate_ts, backend):
        """Test that prophet_copy handles cutoff dates before the history start.
        
        This test reproduces a bug where calling prophet_copy with a cutoff date
        that is earlier than all dates in the history causes a ValueError when
        trying to get the max of an empty series.
        """
        # Create and fit a model
        m = Prophet(stan_backend=backend)
        df = daily_univariate_ts.copy()
        
        # Add explicit changepoints
        changepoints = pd.date_range(start='2012-06-15', end='2012-09-15', periods=5)
        m = Prophet(changepoints=changepoints, stan_backend=backend)
        m.fit(df)
        
        # Try to copy with a cutoff that is before the start of the history
        # This should not raise a ValueError
        history_start = m.history['ds'].min()
        cutoff_before_history = history_start - pd.Timedelta(days=10)
        
        # This should either work gracefully or raise a meaningful error,
        # but not a ValueError about argmax of empty sequence
        m2 = prophet_copy(m, cutoff=cutoff_before_history)
        
        # The copied model should have no changepoints since they're all after the cutoff
        assert len(m2.changepoints) == 0
