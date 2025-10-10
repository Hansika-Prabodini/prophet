# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, print_function
from abc import abstractmethod, ABC
from dataclasses import dataclass
from typing import Sequence, Tuple, Union
from collections import OrderedDict
from enum import Enum
import importlib_resources
import pathlib
import platform

import logging
logger = logging.getLogger('prophet.models')

PLATFORM = "win" if platform.platform().startswith("Win") else "unix"

class TrendIndicator(Enum):
    LINEAR = 0
    LOGISTIC = 1
    FLAT = 2

@dataclass
class ModelInputData:
    T: int
    S: int
    K: int
    tau: float
    trend_indicator: int
    y: Sequence[float]  # length T
    t: Sequence[float]  # length T
    cap: Sequence[float]  # length T
    t_change: Sequence[float]  # length S
    s_a: Sequence[int]  # length K
    s_m: Sequence[int]  # length K
    X: Sequence[Sequence[float]]  # shape (T, K)
    sigmas: Sequence[float]  # length K

@dataclass
class ModelParams:
    k: float
    m: float
    delta: Sequence[float]  # length S
    beta: Sequence[float]  # length K
    sigma_obs: float


class IStanBackend(ABC):
    def __init__(self):
        self.model = self.load_model()
        self.stan_fit = None
        self.newton_fallback = True

    def set_options(self, **kwargs):
        """
        Specify model options as kwargs.
         * newton_fallback [bool]: whether to fallback to Newton if L-BFGS fails
        """
        for k, v in kwargs.items():
            if k == 'newton_fallback':
                self.newton_fallback = v
            else:
                raise ValueError(f'Unknown option {k}')
    
    def cleanup(self):
        """Clean up temporary files created during model fitting."""
        pass


    @staticmethod
    @abstractmethod
    def get_type():
        pass

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def fit(self, stan_init, stan_data, **kwargs) -> dict:
        pass

    @abstractmethod
    def sampling(self, stan_init, stan_data, samples, **kwargs) -> dict:
        pass


class CmdStanPyBackend(IStanBackend):
    CMDSTAN_VERSION = "2.33.1"
    def __init__(self):
        import cmdstanpy
        # this must be set before super.__init__() for load_model to work on Windows
        local_cmdstan = importlib_resources.files("prophet") / "stan_model" / f"cmdstan-{self.CMDSTAN_VERSION}"
        if local_cmdstan.exists():
            cmdstanpy.set_cmdstan_path(str(local_cmdstan))
        super().__init__()

    @staticmethod
    def get_type():
        return StanBackendEnum.CMDSTANPY.name

    def load_model(self):
        import cmdstanpy
        model_file = importlib_resources.files("prophet") / "stan_model" / "prophet_model.bin"
        return cmdstanpy.CmdStanModel(exe_file=str(model_file))

    def fit(self, stan_init, stan_data, **kwargs) -> dict:
        """Fit the model using optimization.
        
        Args:
            stan_init: Initial parameter values
            stan_data: Input data for the model
            **kwargs: Additional arguments passed to the optimizer
            
        Returns:
            Dictionary of optimized parameters
        """
        # Handle custom initialization
        if 'inits' not in kwargs and 'init' in kwargs:
            stan_init = self.sanitize_custom_inits(stan_init, kwargs['init'])
            del kwargs['init']

        inits_list, data_list = self.prepare_data(stan_init, stan_data)
        
        # Choose algorithm based on data size
        small_data_threshold = 100
        default_iterations = 10000
        
        args = dict(
            data=data_list,
            inits=inits_list,
            algorithm='Newton' if data_list['T'] < small_data_threshold else 'LBFGS',
            iter=default_iterations,
        )
        args.update(kwargs)

        try:
            self.stan_fit = self.model.optimize(**args)
        except RuntimeError as e:
            # Fall back on Newton if L-BFGS fails
            if not self.newton_fallback or args['algorithm'] == 'Newton':
                raise RuntimeError(
                    f"Optimization failed with algorithm {args['algorithm']}"
                ) from e
            logger.warning(
                'Optimization terminated abnormally. Falling back to Newton algorithm.'
            )
            args['algorithm'] = 'Newton'
            self.stan_fit = self.model.optimize(**args)
            
        params = self.stan_to_dict_numpy(
            self.stan_fit.column_names, self.stan_fit.optimized_params_np
        )
        for par in params:
            params[par] = params[par].reshape((1, -1))
        return params

    def sampling(self, stan_init, stan_data, samples, **kwargs) -> dict:
        """Generate posterior samples using MCMC.
        
        Args:
            stan_init: Initial parameter values
            stan_data: Input data for the model
            samples: Total number of samples to generate (across all chains)
            **kwargs: Additional arguments passed to the sampler
            
        Returns:
            Dictionary of parameter samples
        """
        # Handle custom initialization
        if 'inits' not in kwargs and 'init' in kwargs:
            stan_init = self.sanitize_custom_inits(stan_init, kwargs['init'])
            del kwargs['init']

        inits_list, data_list = self.prepare_data(stan_init, stan_data)
        
        default_chains = 4
        args = dict(
            data=data_list,
            inits=inits_list,
        )
        if 'chains' not in kwargs:
            kwargs['chains'] = default_chains
        chains = kwargs['chains']
        
        # Split samples between warmup and sampling across chains
        # Ensure that the total number of post-warmup draws across all chains is approximately `samples`
        # and that both warmup and sampling have at least one iteration.
        per_chain_half = max(1, samples // (2 * chains))
        kwargs['iter_sampling'] = per_chain_half
        if 'iter_warmup' not in kwargs:
            kwargs['iter_warmup'] = per_chain_half
        args.update(kwargs)

        self.stan_fit = self.model.sample(**args)
        res = self.stan_fit.draws()
        (n_samples, n_chains, n_columns) = res.shape
        res = res.reshape((n_samples * n_chains, n_columns))
        params = self.stan_to_dict_numpy(self.stan_fit.column_names, res)

        # Reshape parameters to match expected output format
        for par in params:
            s = params[par].shape
            if s[1] == 1:
                params[par] = params[par].reshape((s[0],))

            if par in ['delta', 'beta'] and len(s) < 2:
                params[par] = params[par].reshape((-1, 1))

        return params

    def cleanup(self):
        """Clean up temporary files created during model fitting."""
        import cmdstanpy
        
        if hasattr(self, "stan_fit") and self.stan_fit is not None:
            fit_result: Union[cmdstanpy.CmdStanMLE, cmdstanpy.CmdStanMCMC] = self.stan_fit
            to_remove = (
                fit_result.runset.csv_files + 
                fit_result.runset.diagnostic_files + 
                fit_result.runset.stdout_files + 
                fit_result.runset.profile_files
            )
            for fpath in to_remove:
                file_path = pathlib.Path(fpath)
                if file_path.is_file():
                    try:
                        file_path.unlink()
                    except OSError as e:
                        logger.warning(f'Failed to remove temporary file {fpath}: {e}')
                
    @staticmethod
    def sanitize_custom_inits(default_inits, custom_inits):
        """Validate that custom inits have the correct type and shape, otherwise use defaults.
        
        Args:
            default_inits: Dictionary of default initialization values
            custom_inits: Dictionary of custom initialization values to validate
            
        Returns:
            Dictionary of sanitized initialization values
        """
        sanitized = {}
        for param in ['k', 'm', 'sigma_obs']:
            try:
                sanitized[param] = float(custom_inits.get(param))
            except (TypeError, ValueError, KeyError) as e:
                logger.debug(f'Using default value for {param}: {e}')
                sanitized[param] = default_inits[param]
        for param in ['delta', 'beta']:
            try:
                if default_inits[param].shape == custom_inits[param].shape:
                    sanitized[param] = custom_inits[param]
                else:
                    logger.warning(
                        f'Shape mismatch for {param}: expected {default_inits[param].shape}, '
                        f'got {custom_inits[param].shape}. Using default.'
                    )
                    sanitized[param] = default_inits[param]
            except (AttributeError, KeyError):
                logger.debug(f'Using default value for {param}')
                sanitized[param] = default_inits[param]
        return sanitized

    @staticmethod
    def prepare_data(init, data) -> Tuple[dict, dict]:
        """Convert numpy arrays to lists that can be read by cmdstanpy.
        
        Args:
            init: Dictionary of initial parameter values
            data: Dictionary of input data
            
        Returns:
            Tuple of (initialization dict, data dict) formatted for cmdstanpy
        """
        X_val = data['X'].to_numpy() if hasattr(data['X'], 'to_numpy') else data['X']
        cmdstanpy_data = {
            'T': data['T'],
            'S': data['S'],
            'K': data['K'],
            'tau': data['tau'],
            'trend_indicator': data['trend_indicator'],
            'y': data['y'].tolist(),
            't': data['t'].tolist(),
            'cap': data['cap'].tolist(),
            't_change': data['t_change'].tolist(),
            's_a': data['s_a'].tolist(),
            's_m': data['s_m'].tolist(),
            'X': X_val.tolist(),
            'sigmas': data['sigmas'].tolist()
        }

        cmdstanpy_init = {
            'k': init['k'],
            'm': init['m'],
            'delta': init['delta'].tolist(),
            'beta': init['beta'].tolist(),
            'sigma_obs': init['sigma_obs']
        }
        return (cmdstanpy_init, cmdstanpy_data)

    @staticmethod
    def stan_to_dict_numpy(column_names: Tuple[str, ...], data: 'np.array') -> OrderedDict:
        """Convert Stan output columns to dictionary of numpy arrays.
        
        Args:
            column_names: Tuple of column names from Stan output
            data: Numpy array of parameter values
            
        Returns:
            OrderedDict mapping parameter names to numpy arrays
            
        Raises:
            RuntimeError: If duplicate column names are found
        """
        import numpy as np

        output = OrderedDict()

        prev = None
        start = 0
        end = 0
        two_dims = len(data.shape) > 1
        
        for cname in column_names:
            # Parse parameter name from column name
            parsed = cname.split(".") if "." in cname else cname.split("[")
            curr = parsed[0]
            if prev is None:
                prev = curr

            if curr != prev:
                if prev in output:
                    raise RuntimeError(
                        f"Found repeated column name: {prev}"
                    )
                if two_dims:
                    output[prev] = np.array(data[:, start:end])
                else:
                    output[prev] = np.array(data[start:end])
                prev = curr
                start = end
            end += 1
            
        # Add final parameter
        if prev in output:
            raise RuntimeError(
                f"Found repeated column name: {prev}"
            )
        if two_dims:
            output[prev] = np.array(data[:, start:end])
        else:
            output[prev] = np.array(data[start:end])
        return output




class StanBackendEnum(Enum):
    CMDSTANPY = CmdStanPyBackend

    @staticmethod
    def get_backend_class(name: str) -> IStanBackend:
        try:
            return StanBackendEnum[name].value
        except KeyError as e:
            raise ValueError(f"Unknown stan backend: {name}") from e