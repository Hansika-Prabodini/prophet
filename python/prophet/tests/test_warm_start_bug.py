# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pytest

from prophet import Prophet
from prophet.utilities import warm_start_params


class TestParameterReshapingBug:
    def test_parameter_reshaping_1d_input(self, daily_univariate_ts, backend):
        """Test for IndexError bug in parameter reshaping when input parameters are 1D.
        
        This test addresses a bug in the CmdStanPyBackend.sampling method where:
        - Line 166: if s[1] == 1: assumes all parameters have at least 2 dimensions
        - If a parameter comes back as 1D from stan_to_dict_numpy, s[1] would cause IndexError
        - The fix adds len(s) >= 2 check before accessing s[1]
        """
        # Fit a model with MCMC sampling 
        m = Prophet(mcmc_samples=100, stan_backend=backend)
        # Use smaller dataset to potentially trigger edge cases in parameter shapes
        small_data = daily_univariate_ts.iloc[:50]  
        
        # Before the fix, this could potentially fail with IndexError 
        # if parameters come back as 1D arrays
        m.fit(small_data, show_progress=False)
        
        # Verify that all parameters are properly shaped and accessible
        for pname in ['k', 'm', 'sigma_obs', 'delta', 'beta']:
            assert pname in m.params
            param_shape = m.params[pname].shape
            # Parameters should either be 1D or 2D, not 0D
            assert len(param_shape) >= 1
        
        # Verify that warm_start_params works without errors
        params = warm_start_params(m)
        
        # All expected parameters should be present
        expected_params = ['k', 'm', 'sigma_obs', 'delta', 'beta']
        for pname in expected_params:
            assert pname in params
            
        # Scalar parameters should be actual scalars (not arrays)
        for scalar_param in ['k', 'm', 'sigma_obs']:
            assert isinstance(params[scalar_param], (int, float, np.number))
            
        # Array parameters should be 1D arrays
        for array_param in ['delta', 'beta']:
            assert isinstance(params[array_param], np.ndarray)
            assert params[array_param].ndim == 1
