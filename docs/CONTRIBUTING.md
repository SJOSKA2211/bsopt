# Contributing to BSOPT Platform

Thank you for considering contributing to the BSOPT Platform! This document provides guidelines for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Process](#development-process)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Community Guidelines](#community-guidelines)

## Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inclusive environment for all contributors, regardless of background or identity.

### Our Standards

**Positive Behavior**:
- Using welcoming and inclusive language
- Respecting differing viewpoints and experiences
- Gracefully accepting constructive criticism
- Focusing on what is best for the community
- Showing empathy towards other community members

**Unacceptable Behavior**:
- Trolling, insulting/derogatory comments, and personal attacks
- Public or private harassment
- Publishing others' private information without permission
- Other conduct which could reasonably be considered inappropriate

## How Can I Contribute?

### Reporting Bugs

**Before Submitting**:
- Check existing issues to avoid duplicates
- Verify the bug exists in the latest version
- Collect relevant information (version, OS, error messages)

**Bug Report Template**:
```markdown
**Description**: Brief description of the bug

**Steps to Reproduce**:
1. Step 1
2. Step 2
3. ...

**Expected Behavior**: What should happen

**Actual Behavior**: What actually happens

**Environment**:
- OS: Ubuntu 22.04
- Python: 3.11.5
- Version: 2.1.0

**Additional Context**: Screenshots, error logs, etc.
```

### Suggesting Enhancements

**Enhancement Proposal Template**:
```markdown
**Problem**: What problem does this solve?

**Proposed Solution**: How would it work?

**Alternatives Considered**: What else did you think about?

**Use Cases**: When would this be useful?
```

### Contributing Code

**Good First Issues**:
- Look for issues labeled `good-first-issue`
- Documentation improvements
- Test coverage increases
- Bug fixes

**Areas of Contribution**:
1. **Pricing Engines**: New methods, optimizations
2. **API Development**: New endpoints, improvements
3. **Testing**: Unit, integration, e2e tests
4. **Documentation**: User guides, API docs
5. **Frontend**: React components, visualizations
6. **ML Models**: New models, feature engineering
7. **Infrastructure**: Docker, Kubernetes, CI/CD

## Development Process

### 1. Set Up Development Environment

```bash
# Fork and clone repository
git clone https://github.com/YOUR_USERNAME/bsopt-platform.git
cd bsopt-platform

# Add upstream remote
git remote add upstream https://github.com/yourusername/bsopt-platform.git

# Set up environment
./setup.sh
docker-compose up -d
```

### 2. Create Feature Branch

```bash
# Update main branch
git checkout main
git pull upstream main

# Create feature branch
git checkout -b feature/your-feature-name
```

**Branch Naming Conventions**:
- `feature/description` - New features
- `fix/description` - Bug fixes
- `docs/description` - Documentation
- `refactor/description` - Code refactoring
- `test/description` - Testing improvements

### 3. Make Changes

- Write clean, readable code
- Follow coding standards (see below)
- Add tests for new functionality
- Update documentation as needed
- Commit frequently with clear messages

### 4. Test Your Changes

```bash
# Run all tests
pytest

# Run specific tests
pytest tests/unit/test_your_module.py

# Check coverage
pytest --cov=src --cov-report=term

# Format code
black src/ tests/
isort src/ tests/

# Type checking
mypy src/

# Linting
flake8 src/ tests/
```

### 5. Commit Changes

**Commit Message Format**:
```
type(scope): brief description (50 chars max)

Detailed explanation of the change (if needed).
Wrap at 72 characters.

- Bullet points for multiple changes
- Reference related issues

Closes #123
```

**Types**: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`, `perf`

**Examples**:
```bash
git commit -m "feat(pricing): add Black-Scholes-Merton model

Implement analytical pricing for European options with
continuous dividend yield.

- Add BSParameters dataclass
- Implement price_call and price_put methods
- Add Greeks calculation
- Include validation and tests

Closes #42"
```

### 6. Push to Your Fork

```bash
git push origin feature/your-feature-name
```

### 7. Create Pull Request

Go to GitHub and create a pull request from your fork to the main repository.

## Pull Request Process

### PR Checklist

Before submitting, ensure:

- [ ] Code follows style guidelines
- [ ] All tests pass (`pytest`)
- [ ] New tests added for new functionality
- [ ] Documentation updated (if applicable)
- [ ] Commit messages follow format
- [ ] No merge conflicts with main branch
- [ ] PR description clearly explains changes

### PR Template

```markdown
## Description
Brief description of what this PR does.

## Type of Change
- [ ] Bug fix (non-breaking change fixing an issue)
- [ ] New feature (non-breaking change adding functionality)
- [ ] Breaking change (fix or feature causing existing functionality to change)
- [ ] Documentation update

## Related Issues
Closes #123

## Changes Made
- Change 1
- Change 2
- Change 3

## Testing
Describe how you tested your changes:
- Unit tests added
- Integration tests updated
- Manual testing performed

## Screenshots (if applicable)

## Checklist
- [ ] My code follows the style guidelines
- [ ] I have performed a self-review
- [ ] I have commented my code where necessary
- [ ] I have updated the documentation
- [ ] My changes generate no new warnings
- [ ] I have added tests that prove my fix is effective
- [ ] New and existing unit tests pass locally
- [ ] Any dependent changes have been merged
```

### Review Process

1. **Automated Checks**: CI/CD runs tests, linting, type checking
2. **Code Review**: Maintainers review your code
3. **Feedback**: Address review comments
4. **Approval**: At least one maintainer approval required
5. **Merge**: Maintainer merges your PR

**Review Timeline**:
- Small PRs (<100 lines): 1-2 days
- Medium PRs (100-500 lines): 3-5 days
- Large PRs (>500 lines): 1-2 weeks

**Tips for Faster Review**:
- Keep PRs focused and small
- Write clear descriptions
- Respond promptly to feedback
- Link related issues
- Add screenshots/examples

## Coding Standards

### Python Style Guide

Follow **PEP 8** with these modifications:

1. **Line Length**: 100 characters
2. **Imports**: Organized (stdlib, third-party, local)
3. **Type Hints**: Required for all functions
4. **Docstrings**: Google style, required for public functions

**Example**:

```python
from typing import Optional, Tuple
import numpy as np
from scipy.stats import norm

from src.utils.validators import validate_positive


def calculate_call_price(
    spot: float,
    strike: float,
    maturity: float,
    volatility: float,
    rate: float,
    dividend: float = 0.0
) -> Tuple[float, dict]:
    """
    Calculate European call option price using Black-Scholes.

    Args:
        spot: Current asset price (must be positive)
        strike: Strike price (must be positive)
        maturity: Time to expiration in years
        volatility: Annualized volatility
        rate: Risk-free rate (annualized)
        dividend: Dividend yield (annualized), defaults to 0.0

    Returns:
        Tuple containing (price, greeks_dict)

    Raises:
        ValueError: If parameters are invalid

    Example:
        >>> price, greeks = calculate_call_price(100, 100, 1.0, 0.25, 0.05)
        >>> print(f"Price: ${price:.2f}")
        Price: $10.45
    """
    validate_positive(spot, "spot")
    validate_positive(strike, "strike")

    # Implementation...
    return price, greeks
```

### Automated Formatting

```bash
# Format code (required before PR)
black src/ tests/
isort src/ tests/

# Verify formatting
black --check src/ tests/
isort --check-only src/ tests/
```

### Testing Requirements

- **Coverage**: New code must have >90% test coverage
- **Types**: Unit tests required, integration tests preferred
- **Quality**: Tests must be meaningful and test edge cases

```python
# Good test
def test_call_price_increases_with_volatility():
    """Call price increases as volatility increases."""
    params_low_vol = BSParameters(100, 100, 1.0, 0.2, 0.05)
    params_high_vol = BSParameters(100, 100, 1.0, 0.3, 0.05)

    price_low = BlackScholesEngine.price_call(params_low_vol)
    price_high = BlackScholesEngine.price_call(params_high_vol)

    assert price_high > price_low, "Price should increase with volatility"

# Bad test
def test_price():
    """Test pricing."""  # Vague docstring
    result = some_function()
    assert result  # What are we testing?
```

### Documentation Requirements

- **Docstrings**: All public functions, classes, modules
- **Comments**: Complex logic, non-obvious code
- **README**: Update if adding features
- **API Docs**: Update OpenAPI spec if changing API

## Community Guidelines

### Communication Channels

- **GitHub Issues**: Bug reports, feature requests
- **GitHub Discussions**: Questions, ideas
- **Pull Requests**: Code review discussions
- **Slack** (internal): #bsopt-dev channel
- **Email**: dev-team@bsopt.com

### Getting Help

**Before Asking**:
1. Check documentation
2. Search existing issues
3. Review closed PRs
4. Try debugging yourself

**When Asking**:
- Be specific
- Include context (OS, version, error messages)
- Show what you've tried
- Provide minimal reproducible example

**Good Question**:
```
I'm implementing a new pricing method but getting this error:
[paste error with traceback]

Environment: Ubuntu 22.04, Python 3.11.5

What I've tried:
1. Checked existing methods for patterns
2. Verified parameters are valid
3. Ran tests (pytest output attached)

File: src/pricing/my_method.py, line 42
```

**Bad Question**:
```
Code doesn't work. Help???
```

### Recognition

We appreciate all contributions! Contributors are recognized in:
- CHANGELOG.md
- GitHub contributors page
- Release notes
- Annual contributor spotlight

## Development Workflow Summary

```bash
# 1. Set up
git clone <your-fork>
cd bsopt-platform
./setup.sh

# 2. Create branch
git checkout -b feature/your-feature

# 3. Develop
# ... make changes ...
pytest  # Test frequently

# 4. Format & lint
black src/ tests/
isort src/ tests/
mypy src/
flake8 src/ tests/

# 5. Commit
git add .
git commit -m "feat(scope): description"

# 6. Push
git push origin feature/your-feature

# 7. Create PR on GitHub

# 8. Address review feedback
git add .
git commit -m "fix: address review comments"
git push origin feature/your-feature

# 9. Celebrate merge! 🎉
```

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Questions?

- **General**: GitHub Discussions
- **Bugs**: GitHub Issues
- **Security**: security@bsopt.com (private)
- **Other**: dev-team@bsopt.com

---

**Thank you for contributing to BSOPT Platform!**

Together, we're building the future of quantitative finance tools.
