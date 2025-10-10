# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Custom exception classes for the Prophet forecasting library.

This module defines a hierarchy of exception classes that provide clear,
specific error messages for different failure scenarios in Prophet. Using
these exceptions helps distinguish between different types of errors and
enables more precise error handling in application code.

Exception Hierarchy:
    Exception
        └── ProphetError (base class for all Prophet exceptions)
            ├── ModelNotFittedError
            ├── ModelAlreadyFittedError
            ├── DataValidationError
            ├── ConfigurationError
            └── ProphetInternalError
"""

from __future__ import absolute_import, division, print_function


class ProphetError(Exception):
    """
    Base exception class for all Prophet-specific errors.
    
    This serves as the base class for all custom exceptions in the Prophet
    library. Applications can catch this exception to handle any Prophet-specific
    error, or catch more specific subclasses for targeted error handling.
    
    Parameters
    ----------
    message : str
        A descriptive error message explaining what went wrong.
    
    Examples
    --------
    Catching all Prophet errors:
    
    >>> from prophet import Prophet
    >>> from prophet.exceptions import ProphetError
    >>> try:
    ...     m = Prophet()
    ...     # Some Prophet operation
    ... except ProphetError as e:
    ...     print(f"Prophet error occurred: {e}")
    
    Raising a generic Prophet error:
    
    >>> raise ProphetError("An unexpected error occurred in Prophet")
    
    Notes
    -----
    In most cases, you should raise or catch a more specific subclass rather
    than this base exception. Use this class when no other specific exception
    type is appropriate, or when you want to catch all Prophet-related errors.
    """
    
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class ModelNotFittedError(ProphetError):
    """
    Exception raised when attempting to use a model that hasn't been fitted.
    
    This exception is raised when calling methods that require a fitted model
    (e.g., predict, plot_components) before the fit() method has been called.
    A model is considered "fitted" after successfully calling fit() with
    training data.
    
    Parameters
    ----------
    message : str, optional
        A custom error message. If not provided, a default message is used.
    
    Examples
    --------
    Attempting to predict before fitting:
    
    >>> from prophet import Prophet
    >>> from prophet.exceptions import ModelNotFittedError
    >>> m = Prophet()
    >>> future = m.make_future_dataframe(periods=365)
    Traceback (most recent call last):
        ...
    ModelNotFittedError: Model has not been fitted. Call fit() with training data before using predict().
    
    Proper usage with error handling:
    
    >>> from prophet import Prophet
    >>> from prophet.exceptions import ModelNotFittedError
    >>> import pandas as pd
    >>> 
    >>> m = Prophet()
    >>> try:
    ...     forecast = m.predict(future_df)
    ... except ModelNotFittedError:
    ...     print("Model needs to be fitted first")
    ...     m.fit(training_data)
    ...     forecast = m.predict(future_df)
    
    Checking if model is fitted before prediction:
    
    >>> from prophet import Prophet
    >>> m = Prophet()
    >>> if m.history is not None:
    ...     forecast = m.predict(future)
    ... else:
    ...     raise ModelNotFittedError("Please fit the model before prediction")
    
    Notes
    -----
    Methods that typically raise this exception include:
    - predict()
    - make_future_dataframe() (when periods not specified)
    - plot_components()
    - Any method that depends on fitted model parameters
    
    See Also
    --------
    ModelAlreadyFittedError : Raised when trying to fit an already fitted model
    """
    
    def __init__(self, message=None):
        if message is None:
            message = (
                "Model has not been fitted. Call fit() with training data "
                "before using predict() or other methods that require a fitted model."
            )
        super().__init__(message)


class ModelAlreadyFittedError(ProphetError):
    """
    Exception raised when attempting to fit a model that has already been fitted.
    
    This exception is raised when calling fit() on a Prophet model that has
    already been fitted. Prophet models are designed to be fitted once. If you
    need to refit with new data or different parameters, create a new Prophet
    instance.
    
    Parameters
    ----------
    message : str, optional
        A custom error message. If not provided, a default message is used.
    
    Examples
    --------
    Attempting to fit an already fitted model:
    
    >>> from prophet import Prophet
    >>> from prophet.exceptions import ModelAlreadyFittedError
    >>> import pandas as pd
    >>> 
    >>> df = pd.DataFrame({'ds': pd.date_range('2020-01-01', periods=100),
    ...                    'y': range(100)})
    >>> m = Prophet()
    >>> m.fit(df)
    >>> m.fit(df)  # Attempting to fit again
    Traceback (most recent call last):
        ...
    ModelAlreadyFittedError: Model has already been fitted. Create a new Prophet instance to fit with different data or parameters.
    
    Proper pattern for refitting with new data:
    
    >>> from prophet import Prophet
    >>> from prophet.exceptions import ModelAlreadyFittedError
    >>> 
    >>> # First model
    >>> m1 = Prophet()
    >>> m1.fit(old_data)
    >>> 
    >>> # For new data, create a new instance
    >>> m2 = Prophet()
    >>> m2.fit(new_data)
    
    Handling refitting scenarios:
    
    >>> from prophet import Prophet
    >>> from prophet.exceptions import ModelAlreadyFittedError
    >>> 
    >>> try:
    ...     m.fit(new_data)
    ... except ModelAlreadyFittedError:
    ...     print("Model already fitted. Creating new instance...")
    ...     m = Prophet()
    ...     m.fit(new_data)
    
    Notes
    -----
    This design choice ensures that:
    1. Model state remains consistent and predictable
    2. Users explicitly create new instances for new experiments
    3. Confusion about incremental vs. full refitting is avoided
    
    If you need to update a model with new data while retaining learned
    parameters, consider using cross-validation or creating an ensemble
    of models instead.
    
    See Also
    --------
    ModelNotFittedError : Raised when using an unfitted model
    """
    
    def __init__(self, message=None):
        if message is None:
            message = (
                "Model has already been fitted. Create a new Prophet instance "
                "to fit with different data or parameters."
            )
        super().__init__(message)


class DataValidationError(ProphetError, ValueError):
    """
    Exception raised when input data fails validation checks.
    
    This exception is raised when the input data does not meet Prophet's
    requirements. Common scenarios include missing required columns, NaN values,
    incorrect data types, insufficient data points, or invalid date formats.
    
    Parameters
    ----------
    message : str
        A descriptive error message explaining the validation failure.
    
    Examples
    --------
    Missing required column 'ds':
    
    >>> from prophet import Prophet
    >>> from prophet.exceptions import DataValidationError
    >>> import pandas as pd
    >>> 
    >>> df = pd.DataFrame({'date': pd.date_range('2020-01-01', periods=100),
    ...                    'y': range(100)})
    >>> m = Prophet()
    >>> m.fit(df)
    Traceback (most recent call last):
        ...
    DataValidationError: DataFrame must have columns 'ds' and 'y' with the dates and values respectively.
    
    Handling NaN values:
    
    >>> from prophet import Prophet
    >>> from prophet.exceptions import DataValidationError
    >>> import pandas as pd
    >>> import numpy as np
    >>> 
    >>> df = pd.DataFrame({
    ...     'ds': pd.date_range('2020-01-01', periods=100),
    ...     'y': [np.nan if i % 10 == 0 else i for i in range(100)]
    ... })
    >>> try:
    ...     m = Prophet()
    ...     m.fit(df)
    ... except DataValidationError as e:
    ...     print(f"Data validation failed: {e}")
    ...     # Clean the data
    ...     df = df.dropna()
    ...     m.fit(df)
    
    Invalid date format:
    
    >>> from prophet.exceptions import DataValidationError
    >>> df = pd.DataFrame({'ds': ['not-a-date', '2020-01-02'],
    ...                    'y': [1, 2]})
    >>> try:
    ...     m = Prophet()
    ...     m.fit(df)
    ... except DataValidationError as e:
    ...     print(f"Invalid date format: {e}")
    
    Validating custom data:
    
    >>> from prophet.exceptions import DataValidationError
    >>> 
    >>> def validate_data(df):
    ...     if len(df) < 2:
    ...         raise DataValidationError("DataFrame must contain at least 2 rows")
    ...     if df['ds'].isnull().any():
    ...         raise DataValidationError("Found NaN in column 'ds'")
    ...     if not pd.api.types.is_numeric_dtype(df['y']):
    ...         raise DataValidationError("Column 'y' must contain numeric values")
    
    Notes
    -----
    Common data validation failures include:
    - Missing 'ds' or 'y' columns
    - NaN or infinite values in 'y' column
    - Non-datetime values in 'ds' column
    - Insufficient number of data points (< 2 rows)
    - Invalid holiday DataFrame format
    - Cap/floor values for logistic growth not provided or invalid
    - Duplicate timestamps in 'ds' column
    - Regressor columns missing from prediction DataFrame
    
    This exception inherits from both ProphetError and ValueError to maintain
    backward compatibility with existing code that may catch ValueError.
    
    See Also
    --------
    ConfigurationError : For parameter/configuration validation errors
    """
    
    def __init__(self, message):
        super().__init__(message)


class ConfigurationError(ProphetError, ValueError):
    """
    Exception raised when model parameters or configuration are invalid.
    
    This exception is raised when Prophet is initialized or configured with
    invalid parameters, such as incorrect growth types, invalid prior scales,
    conflicting seasonality settings, or out-of-range parameter values.
    
    Parameters
    ----------
    message : str
        A descriptive error message explaining the configuration error.
    
    Examples
    --------
    Invalid growth parameter:
    
    >>> from prophet import Prophet
    >>> from prophet.exceptions import ConfigurationError
    >>> 
    >>> try:
    ...     m = Prophet(growth='exponential')
    ... except ConfigurationError as e:
    ...     print(f"Invalid configuration: {e}")
    Traceback (most recent call last):
        ...
    ConfigurationError: Parameter "growth" should be "linear", "logistic" or "flat".
    
    Invalid changepoint_range:
    
    >>> from prophet import Prophet
    >>> from prophet.exceptions import ConfigurationError
    >>> 
    >>> try:
    ...     m = Prophet(changepoint_range=1.5)
    ... except ConfigurationError as e:
    ...     print(f"Configuration error: {e}")
    Traceback (most recent call last):
        ...
    ConfigurationError: Parameter "changepoint_range" must be in [0, 1]
    
    Invalid seasonality configuration:
    
    >>> from prophet import Prophet
    >>> from prophet.exceptions import ConfigurationError
    >>> 
    >>> m = Prophet()
    >>> try:
    ...     m.add_seasonality(name='monthly', period=30.5, fourier_order=-1)
    ... except ConfigurationError as e:
    ...     print(f"Invalid seasonality: {e}")
    
    Handling configuration errors gracefully:
    
    >>> from prophet import Prophet
    >>> from prophet.exceptions import ConfigurationError
    >>> 
    >>> def create_prophet_model(config):
    ...     try:
    ...         return Prophet(
    ...             growth=config.get('growth', 'linear'),
    ...             changepoint_prior_scale=config.get('changepoint_prior_scale', 0.05),
    ...             seasonality_mode=config.get('seasonality_mode', 'additive')
    ...         )
    ...     except ConfigurationError as e:
    ...         print(f"Invalid configuration: {e}")
    ...         # Return model with default parameters
    ...         return Prophet()
    
    Reserved name validation:
    
    >>> from prophet import Prophet
    >>> from prophet.exceptions import ConfigurationError
    >>> 
    >>> m = Prophet()
    >>> try:
    ...     m.add_regressor('trend')  # 'trend' is a reserved name
    ... except ConfigurationError as e:
    ...     print(f"Cannot use reserved name: {e}")
    
    Notes
    -----
    Common configuration errors include:
    - Invalid growth type (must be 'linear', 'logistic', or 'flat')
    - changepoint_range not in [0, 1]
    - Invalid seasonality_mode (must be 'additive' or 'multiplicative')
    - Invalid holidays_mode (must be 'additive' or 'multiplicative')
    - Negative or zero Fourier order for seasonality
    - Using reserved names for regressors or seasonalities
    - Invalid prior scale values (must be positive)
    - Conflicting parameter combinations
    - Invalid scaling method (must be 'absmax' or 'minmax')
    
    This exception inherits from both ProphetError and ValueError to maintain
    backward compatibility with existing code that may catch ValueError.
    
    See Also
    --------
    DataValidationError : For input data validation errors
    """
    
    def __init__(self, message):
        super().__init__(message)


class ProphetInternalError(ProphetError, RuntimeError):
    """
    Exception raised when Prophet encounters an unexpected internal state.
    
    This exception indicates a bug or unexpected condition within Prophet's
    internal logic. It should not be raised for user errors (data or
    configuration issues), but rather for situations that suggest a problem
    with Prophet's implementation.
    
    Parameters
    ----------
    message : str
        A descriptive error message explaining the internal error.
    
    Examples
    --------
    Internal state inconsistency:
    
    >>> from prophet.exceptions import ProphetInternalError
    >>> 
    >>> # This would be raised internally by Prophet
    >>> if component_cols['additive_terms'].sum() + component_cols['multiplicative_terms'].sum() > len(df):
    ...     raise ProphetInternalError('A bug occurred in seasonal components.')
    
    Unexpected matrix dimensions:
    
    >>> from prophet.exceptions import ProphetInternalError
    >>> 
    >>> if X.shape[1] != expected_features:
    ...     raise ProphetInternalError(
    ...         f"Feature matrix has unexpected shape. Expected {expected_features} "
    ...         f"features but got {X.shape[1]}. This indicates an internal bug."
    ...     )
    
    Catching internal errors for debugging:
    
    >>> from prophet import Prophet
    >>> from prophet.exceptions import ProphetInternalError
    >>> import logging
    >>> 
    >>> try:
    ...     m = Prophet()
    ...     m.fit(df)
    ...     forecast = m.predict(future)
    ... except ProphetInternalError as e:
    ...     logging.error(f"Prophet internal error: {e}")
    ...     logging.error("This may be a bug. Please report with minimal example.")
    ...     raise
    
    Component validation failure:
    
    >>> from prophet.exceptions import ProphetInternalError
    >>> 
    >>> if not component_cols.equals(expected_component_cols):
    ...     raise ProphetInternalError(
    ...         'A bug occurred in constructing regressors. Component columns '
    ...         'do not match expected structure.'
    ...     )
    
    Notes
    -----
    This exception indicates potential bugs in Prophet and should be reported
    to the development team with:
    1. A minimal reproducible example
    2. Prophet version information
    3. Input data characteristics (if possible to share)
    4. Full stack trace
    
    Common scenarios that might raise this exception:
    - Inconsistent internal matrix dimensions
    - Component construction failures
    - Regressor column mismatches
    - Stan backend interface errors
    - Unexpected optimization results
    - State corruption after serialization
    
    This exception inherits from both ProphetError and RuntimeError to
    distinguish it from user-facing errors and to maintain compatibility
    with code that catches RuntimeError for unexpected conditions.
    
    If you encounter this exception:
    1. Check if you're using the latest version of Prophet
    2. Verify your installation is not corrupted
    3. Report the issue with a reproducible example
    4. As a workaround, try simplifying your model configuration
    
    See Also
    --------
    ProphetError : Base exception class for all Prophet errors
    """
    
    def __init__(self, message):
        super().__init__(message)


# For backward compatibility, export all exception classes
__all__ = [
    'ProphetError',
    'ModelNotFittedError',
    'ModelAlreadyFittedError',
    'DataValidationError',
    'ConfigurationError',
    'ProphetInternalError',
]
