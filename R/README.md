# Prophet: R Package

Prophet is a procedure for forecasting time series data based on an additive model where non-linear trends are fit with yearly, weekly, and daily seasonality, plus holiday effects. It works best with time series that have strong seasonal effects and several seasons of historical data. Prophet is robust to missing data and shifts in the trend, and typically handles outliers well.

This directory contains the R implementation of Prophet.

## Installation

### From CRAN

⚠️ **The CRAN version of prophet is fairly outdated. To get the latest bug fixes and updated country holiday data, we suggest installing the latest release from GitHub.**

```r
install.packages('prophet')
```

### Latest Release from GitHub

```r
install.packages('remotes')
remotes::install_github('facebook/prophet@*release', subdir = 'R')
```

### Experimental Backend - cmdstanr

You can also choose an experimental alternative stan backend called `cmdstanr`. Once you've installed `prophet`, follow these instructions to use `cmdstanr` instead of `rstan` as the backend:

```r
# We recommend running this in a fresh R session or restarting your current session
install.packages(c("cmdstanr", "posterior"), repos = c("https://mc-stan.org/r-packages/", getOption("repos")))

# If you haven't installed cmdstan before, run:
cmdstanr::install_cmdstan()
# Otherwise, you can point cmdstanr to your cmdstan path:
cmdstanr::set_cmdstan_path(path = <your existing cmdstan>)

# Set the R_STAN_BACKEND environment variable
Sys.setenv(R_STAN_BACKEND = "CMDSTANR")
```

## System Requirements

- R (>= 3.4.0)
- GNU make
- C++11 compiler

### Windows

On Windows, R requires a compiler so you'll need to [follow the instructions](https://github.com/stan-dev/rstan/wiki/RStan-Getting-Started) provided by `rstan`. The key step is installing [Rtools](http://cran.r-project.org/bin/windows/Rtools/) before attempting to install the package.

If you have custom Stan compiler settings, install from source rather than the CRAN binary.

## Quick Start

```r
library(prophet)

# Load example data
df <- read.csv('https://raw.githubusercontent.com/facebook/prophet/main/examples/example_wp_log_peyton_manning.csv')

# Fit the model
m <- prophet(df)

# Make predictions
future <- make_future_dataframe(m, periods = 365)
forecast <- predict(m, future)

# Plot forecast
plot(m, forecast)

# Plot components
prophet_plot_components(m, forecast)
```

## Package Structure

```
R/
├── R/              # R source code
├── src/            # C++/Stan model code
├── man/            # Documentation files
├── tests/          # Unit tests
├── vignettes/      # Package vignettes
├── data-raw/       # Raw data for package datasets
└── inst/           # Additional package files
```

## Key Features

- **Automatic changepoint detection**: Detects trend changes in your data
- **Multiple seasonality**: Handles yearly, weekly, and daily patterns
- **Holiday effects**: Built-in support for holidays in many countries
- **Robust to missing data**: Handles gaps in time series gracefully
- **Custom seasonalities**: Add your own seasonal patterns
- **Uncertainty intervals**: Provides forecast uncertainty estimates
- **Cross-validation**: Built-in tools for model evaluation

## Documentation

- Full documentation: https://facebook.github.io/prophet/docs/quick_start.html#r-api
- Function reference: Use `?prophet` or `help(package='prophet')` in R
- Vignettes: Use `browseVignettes('prophet')` in R
- Issues: https://github.com/facebook/prophet/issues

## Development

To build and test the package locally:

```r
# Install development dependencies
install.packages(c('devtools', 'testthat', 'knitr', 'rmarkdown'))

# Build package
devtools::build()

# Run tests
devtools::test()

# Check package
devtools::check()
```

## Contributing

Contributions are welcome! Please see the [contributing guide](https://facebook.github.io/prophet/docs/contributing.html) for more information.

## License

This package is licensed under the MIT License. See the LICENSE file for details.

## Citation

If you use Prophet in your research, please cite:

Sean J. Taylor, Benjamin Letham (2018) Forecasting at scale. The American Statistician 72(1):37-45. https://peerj.com/preprints/3190.pdf

## Links

- Homepage: https://facebook.github.io/prophet/
- Python package: https://pypi.python.org/pypi/prophet/
- CRAN: https://cran.r-project.org/package=prophet
- Source code: https://github.com/facebook/prophet
