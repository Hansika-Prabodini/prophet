# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Custom exception classes for Prophet."""


class ProphetError(Exception):
    """Base exception class for all Prophet-specific errors."""
    pass


class ModelNotFittedError(ProphetError):
    """
    Exception raised when an operation requires a fitted model but the model hasn't been fit yet.
    
    This error is raised when calling methods like predict() or make_future_dataframe()
    on a Prophet instance that hasn't had fit() called on it yet.
    """
    pass


class DataValidationError(ProphetError):
    """
    Exception raised when input data fails validation.
    
    This includes issues like:
    - Missing required columns ('ds', 'y')
    - Invalid data types
    - NaN or infinite values in required columns
    - Dataframe structure issues
    """
    pass


class ModelConfigurationError(ProphetError):
    """
    Exception raised when model configuration or parameters are invalid.
    
    This includes issues like:
    - Invalid parameter values (e.g., growth not in ['linear', 'logistic', 'flat'])
    - Invalid changepoint_range values
    - Invalid seasonality mode
    - Conflicting parameter combinations
    """
    pass


class ModelAlreadyFittedError(ProphetError):
    """
    Exception raised when attempting to fit a model that has already been fitted.
    
    Prophet models can only be fit once. To fit a new model, create a new Prophet instance.
    """
    pass
