# Contributing to Prophet

Thank you for your interest in contributing to Prophet! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Reporting Bugs](#reporting-bugs)
- [Suggesting Enhancements](#suggesting-enhancements)
- [Pull Request Process](#pull-request-process)
- [Code Style Guidelines](#code-style-guidelines)
- [Testing](#testing)
- [Documentation](#documentation)

## Code of Conduct

This project adheres to the [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to the project maintainers.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR-USERNAME/prophet.git
   cd prophet
   ```
3. **Add the upstream repository**:
   ```bash
   git remote add upstream https://github.com/facebook/prophet.git
   ```
4. **Create a branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Setup

### Python Development

#### Prerequisites
- Python 3.7 or higher
- C++ compiler (gcc/g++ on Linux, Xcode on macOS, Rtools on Windows)
- At least 4GB RAM for building

#### Setup Instructions

```bash
cd python

# Install in editable mode with development dependencies
python -m pip install -e ".[dev, parallel]"

# Run tests to verify installation
pytest prophet/tests/
```

#### Using Docker for Development

```bash
# Build the Docker image
make build

# Run Python shell in Docker
make py-shell

# Run bash shell in Docker
make shell
```

### R Development

#### Prerequisites
- R (>= 3.4.0)
- Rtools (Windows) or appropriate compilers (Linux/macOS)
- RStudio (recommended)

#### Setup Instructions

```bash
cd R

# Install development dependencies
R -e 'install.packages(c("devtools", "testthat"))'
R -e 'devtools::install_deps(".", dependencies = TRUE)'

# Build and install the package
R CMD build .
R CMD INSTALL prophet_*.tar.gz

# Run tests
R -e 'library(prophet); devtools::test()'
```

## How to Contribute

### Types of Contributions

We welcome various types of contributions:

- **Bug fixes**: Fix issues reported in the issue tracker
- **New features**: Add new functionality (discuss in an issue first)
- **Documentation**: Improve docs, add examples, fix typos
- **Tests**: Add test coverage for existing features
- **Performance improvements**: Optimize code performance
- **Code quality**: Refactoring, code cleanup

### Contribution Workflow

1. **Check existing issues** to avoid duplicate work
2. **Create an issue** for significant changes to discuss your approach
3. **Write your code** following our style guidelines
4. **Add tests** for new functionality
5. **Update documentation** as needed
6. **Submit a pull request**

## Reporting Bugs

When reporting bugs, please include:

- **Clear title and description**
- **Prophet version** (`python -c "import prophet; print(prophet.__version__)"` or `packageVersion("prophet")`)
- **Operating system and version**
- **Python/R version**
- **Minimal reproducible example** with sample data
- **Expected behavior** vs actual behavior
- **Error messages** and stack traces

### Bug Report Template

```markdown
**Prophet Version:** 1.1.6
**OS:** Ubuntu 22.04
**Python/R Version:** Python 3.9

**Description:**
Brief description of the bug

**Reproducible Example:**
```python
import prophet
# Your minimal code example
```

**Expected Behavior:**
What you expected to happen

**Actual Behavior:**
What actually happened

**Error Message:**
```
Full error traceback
```
```

## Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion:

- **Use a clear and descriptive title**
- **Provide a detailed description** of the proposed functionality
- **Explain why this enhancement would be useful**
- **Include code examples** showing how the feature would work
- **Consider the scope**: Is this a common use case?

## Pull Request Process

### Before Submitting

1. **Update your branch** with the latest upstream changes:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Run the test suite** and ensure all tests pass:
   ```bash
   # Python
   cd python && pytest

   # R
   cd R && R -e 'devtools::test()'
   ```

3. **Check code style**:
   ```bash
   # Python: Use flake8, black, or similar
   cd python && flake8 prophet/

   # R: Use lintr
   R -e 'lintr::lint_package("R")'
   ```

4. **Update documentation** if you've changed APIs or added features

5. **Add tests** for new functionality

### Submitting the Pull Request

1. **Push your branch** to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Open a Pull Request** on GitHub with:
   - Clear title describing the change
   - Description linking to related issues
   - Summary of changes made
   - Any breaking changes highlighted

3. **Respond to review comments** promptly

4. **Keep your PR up to date** with the base branch

### Pull Request Template

```markdown
## Description
Brief description of changes

## Related Issues
Fixes #123

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Tests pass locally
- [ ] Added new tests for new functionality
- [ ] Updated documentation

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review of code completed
- [ ] Comments added for complex code
- [ ] Documentation updated
- [ ] No new warnings generated
```

## Code Style Guidelines

### Python

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guide
- Use meaningful variable and function names
- Maximum line length: 100 characters
- Use type hints where appropriate
- Write docstrings for all public functions and classes

#### Python Example

```python
def make_future_dataframe(
    self,
    periods: int,
    freq: str = 'D',
    include_history: bool = True
) -> pd.DataFrame:
    """
    Create a dataframe with future dates for forecasting.

    Parameters
    ----------
    periods : int
        Number of future periods to forecast.
    freq : str, default 'D'
        Frequency of predictions (e.g., 'D' for daily, 'H' for hourly).
    include_history : bool, default True
        Whether to include historical dates in the output.

    Returns
    -------
    pd.DataFrame
        Dataframe with 'ds' column containing dates.
    """
    # Implementation
    pass
```

### R

- Follow [Tidyverse Style Guide](https://style.tidyverse.org/)
- Use snake_case for function and variable names
- Add roxygen2 documentation for all exported functions
- Maximum line length: 80 characters

#### R Example

```r
#' Create future dataframe
#'
#' @param m Prophet model object
#' @param periods Number of periods to forecast
#' @param freq Frequency of predictions
#' @param include_history Include historical dates
#'
#' @return Dataframe with future dates
#' @export
make_future_dataframe <- function(m, periods, freq = 'day', include_history = TRUE) {
  # Implementation
}
```

### Stan

- Use clear variable names
- Comment complex mathematical operations
- Follow existing code structure

## Testing

### Python Testing

```bash
cd python

# Run all tests
pytest

# Run specific test file
pytest prophet/tests/test_forecaster.py

# Run with coverage
pytest --cov=prophet --cov-report=html

# Run specific test
pytest prophet/tests/test_forecaster.py::TestForecaster::test_fit
```

### R Testing

```bash
cd R

# Run all tests
R -e 'devtools::test()'

# Run specific test file
R -e 'testthat::test_file("tests/testthat/test-prophet.R")'

# Check package
R CMD check prophet_*.tar.gz
```

### Writing Tests

- Write tests for all new functionality
- Include edge cases and error conditions
- Use descriptive test names
- Keep tests focused and independent
- Mock external dependencies when appropriate

#### Python Test Example

```python
def test_make_future_dataframe():
    """Test future dataframe generation."""
    df = pd.DataFrame({
        'ds': pd.date_range('2020-01-01', periods=100),
        'y': range(100)
    })
    m = Prophet()
    m.fit(df)
    
    future = m.make_future_dataframe(periods=30)
    
    assert len(future) == 130
    assert future['ds'].max() > df['ds'].max()
```

## Documentation

### Documentation Types

1. **Code Documentation**
   - Docstrings for all public APIs
   - Comments for complex logic
   - Type hints (Python)

2. **User Documentation**
   - Update docs/ folder for new features
   - Add examples for new functionality
   - Update README.md if needed

3. **Examples**
   - Add Jupyter notebooks for significant features
   - Include example datasets in examples/ folder

### Building Documentation Locally

```bash
# Python documentation uses Sphinx
cd docs
make html

# View documentation
open _build/html/index.html
```

## Development Best Practices

- **Keep commits atomic**: One logical change per commit
- **Write clear commit messages**: Describe what and why, not how
- **Keep PRs focused**: One feature/fix per pull request
- **Update tests**: Maintain or improve code coverage
- **Profile performance**: Use profiling tools for optimization work
- **Check backward compatibility**: Avoid breaking existing APIs

## Getting Help

- **GitHub Issues**: Search existing issues or create a new one
- **Documentation**: Check https://facebook.github.io/prophet/
- **Stack Overflow**: Tag questions with `prophet`
- **Discussions**: Use GitHub Discussions for questions

## Recognition

Contributors are recognized in:
- GitHub contributors page
- Release notes for significant contributions
- Special mentions for major features

Thank you for contributing to Prophet! 🎉
