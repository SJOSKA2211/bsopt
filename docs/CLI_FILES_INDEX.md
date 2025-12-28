# Black-Scholes CLI - File Index

Complete index of all CLI-related files in the project.

## Core Implementation Files

### Main CLI Files

| File | Location | Lines | Purpose |
|------|----------|-------|---------|
| `cli.py` | `/home/kamau/comparison/cli.py` | 1,035 | Original CLI with basic commands |
| `cli_complete.py` | `/home/kamau/comparison/cli_complete.py` | 900+ | Complete CLI with all features |

**Recommendation**: Use `cli_complete.py` as the primary CLI entry point.

### CLI Support Modules

| File | Location | Lines | Purpose |
|------|----------|-------|---------|
| `src/cli/__init__.py` | `/home/kamau/comparison/src/cli/__init__.py` | 10 | Package initialization |
| `src/cli/auth.py` | `/home/kamau/comparison/src/cli/auth.py` | 350+ | Authentication manager |
| `src/cli/config.py` | `/home/kamau/comparison/src/cli/config.py` | 300+ | Configuration manager |
| `src/cli/portfolio.py` | `/home/kamau/comparison/src/cli/portfolio.py` | 350+ | Portfolio manager |

## Setup & Installation

| File | Location | Purpose |
|------|----------|---------|
| `setup.py` | `/home/kamau/comparison/setup.py` | Package setup for pip install |
| `setup_cli.py` | `/home/kamau/comparison/setup_cli.py` | Interactive setup script |
| `requirements_cli.txt` | `/home/kamau/comparison/requirements_cli.txt` | CLI-specific dependencies |
| `requirements.txt` | `/home/kamau/comparison/requirements.txt` | All project dependencies |

## Documentation

| File | Location | Pages | Content |
|------|----------|-------|---------|
| `CLI_README.md` | `/home/kamau/comparison/CLI_README.md` | ~12 | Main CLI documentation |
| `CLI_DOCUMENTATION.md` | `/home/kamau/comparison/CLI_DOCUMENTATION.md` | ~20 | Complete command reference |
| `CLI_QUICKSTART.md` | `/home/kamau/comparison/CLI_QUICKSTART.md` | ~6 | 5-minute getting started |
| `CLI_IMPLEMENTATION_SUMMARY.md` | `/home/kamau/comparison/CLI_IMPLEMENTATION_SUMMARY.md` | ~15 | Implementation details |
| `CLI_FILES_INDEX.md` | `/home/kamau/comparison/CLI_FILES_INDEX.md` | 1 | This file |

## Testing

| File | Location | Purpose |
|------|----------|---------|
| `test_cli.sh` | `/home/kamau/comparison/test_cli.sh` | Automated test suite |

## Supporting Files

### Pricing Engines (used by CLI)

| File | Location | Purpose |
|------|----------|---------|
| `src/pricing/black_scholes.py` | `/home/kamau/comparison/src/pricing/black_scholes.py` | Black-Scholes analytical |
| `src/pricing/finite_difference.py` | `/home/kamau/comparison/src/pricing/finite_difference.py` | Crank-Nicolson FDM |
| `src/pricing/monte_carlo.py` | `/home/kamau/comparison/src/pricing/monte_carlo.py` | Monte Carlo simulator |

### API & Configuration

| File | Location | Purpose |
|------|----------|---------|
| `src/api/main.py` | `/home/kamau/comparison/src/api/main.py` | FastAPI server |
| `src/config.py` | `/home/kamau/comparison/src/config.py` | Application config |

## File Tree Structure

```
/home/kamau/comparison/
│
├── CLI Entry Points
│   ├── cli.py                          # Basic CLI (1,035 lines)
│   └── cli_complete.py                 # Complete CLI (900+ lines) ⭐ USE THIS
│
├── Setup & Installation
│   ├── setup.py                        # Package setup
│   ├── setup_cli.py                    # Interactive installer
│   ├── requirements.txt                # All dependencies
│   └── requirements_cli.txt            # CLI dependencies
│
├── Documentation
│   ├── CLI_README.md                   # Main docs
│   ├── CLI_DOCUMENTATION.md            # Command reference
│   ├── CLI_QUICKSTART.md               # Quick start
│   ├── CLI_IMPLEMENTATION_SUMMARY.md   # Implementation details
│   └── CLI_FILES_INDEX.md              # This file
│
├── Testing
│   └── test_cli.sh                     # Test suite
│
└── Source Code
    └── src/
        ├── cli/
        │   ├── __init__.py
        │   ├── auth.py                 # Authentication
        │   ├── config.py               # Configuration
        │   └── portfolio.py            # Portfolio management
        │
        ├── pricing/
        │   ├── black_scholes.py        # BS engine
        │   ├── finite_difference.py    # FDM solver
        │   └── monte_carlo.py          # MC simulator
        │
        ├── api/
        │   └── main.py                 # FastAPI server
        │
        └── config.py                   # App config
```

## Installation Instructions

### Option 1: Quick Install

```bash
cd /home/kamau/comparison
python setup_cli.py install
```

### Option 2: Development Mode

```bash
cd /home/kamau/comparison
pip install -e .
```

### Option 3: Manual

```bash
cd /home/kamau/comparison
pip install -r requirements_cli.txt
chmod +x cli_complete.py
ln -s $(pwd)/cli_complete.py /usr/local/bin/bsopt
```

## Usage Examples

### Basic Commands

```bash
# Price an option
bsopt price call --spot 100 --strike 100 --maturity 1.0 --vol 0.2 --rate 0.05

# Calculate Greeks
bsopt greeks --spot 100 --strike 100 --maturity 1.0 --vol 0.2 --rate 0.05

# View help
bsopt --help
bsopt price --help
```

### Authentication

```bash
# Login
bsopt auth login

# Check status
bsopt auth whoami

# Logout
bsopt auth logout
```

### Portfolio

```bash
# List positions
bsopt portfolio list

# Add position
bsopt portfolio add --symbol AAPL --option-type call --quantity 10 \
  --strike 150 --maturity 0.5 --vol 0.3 --rate 0.05 \
  --entry-price 5.50 --spot 148.50

# View P&L
bsopt portfolio pnl
```

### Configuration

```bash
# View config
bsopt config list

# Set value
bsopt config set pricing.default_method mc

# Reset
bsopt config reset
```

## Testing the Installation

Run the test suite:

```bash
cd /home/kamau/comparison
bash test_cli.sh
```

Expected output:
```
========================================
Black-Scholes CLI Test Suite
========================================

CLI found!

----------------------------------------
1. Basic Commands
----------------------------------------
Test 1: Version check
✓ PASSED
Test 2: Help display
✓ PASSED

...

========================================
Test Summary
========================================
Total tests:  25
Passed:       25
Failed:       0

✓ All tests passed!
```

## Configuration Files

### User Configuration

Location: `~/.config/bsopt/config.yml`

```yaml
api:
  base_url: http://localhost:8000

pricing:
  default_method: bs
  mc_paths: 100000

output:
  format: table
  color: true
```

### Project Configuration

Location: `/home/kamau/comparison/.bsopt.yml`

```yaml
pricing:
  default_method: fdm
```

### Credentials Storage

- Primary: System keyring
- Fallback: `~/.config/bsopt/credentials.enc` (encrypted)

### Portfolio Data

Location: `~/.config/bsopt/portfolio.json`

## Dependencies

### Core Dependencies

```
click>=8.1.7          # CLI framework
rich>=13.7.0          # Terminal formatting
pandas>=2.1.0         # Data processing
numpy>=1.24.0         # Numerical computing
scipy>=1.11.0         # Scientific computing
```

### Authentication

```
keyring>=24.3.0       # Secure credential storage
cryptography>=41.0.0  # Encryption
```

### Configuration

```
pyyaml>=6.0.1         # YAML parsing
```

### API

```
fastapi>=0.104.0      # API framework
uvicorn>=0.24.0       # ASGI server
requests>=2.31.0      # HTTP client
```

See `requirements_cli.txt` for complete list.

## File Sizes

| File | Size | Purpose |
|------|------|---------|
| `cli.py` | 38 KB | Basic CLI |
| `cli_complete.py` | 28 KB | Complete CLI |
| `src/cli/auth.py` | ~15 KB | Auth manager |
| `src/cli/config.py` | ~12 KB | Config manager |
| `src/cli/portfolio.py` | ~14 KB | Portfolio manager |
| **Total CLI Code** | **~107 KB** | |
| **Documentation** | **~60 KB** | |
| **Grand Total** | **~167 KB** | |

## Line Counts

| Component | Lines |
|-----------|-------|
| CLI Entry Points | 1,935 |
| CLI Modules | 1,000 |
| Documentation | 1,500 |
| Tests | 200 |
| **Total** | **4,635+** |

## Commands Summary

### Total Commands: 24

#### Pricing (7)
- price, greeks, implied-vol, compare, vol-surface, batch, backtest

#### Portfolio (4)
- portfolio list, portfolio add, portfolio remove, portfolio pnl

#### System (8)
- auth login, auth logout, auth whoami, config list, config get, config set, config reset, serve

#### Data (5)
- init-db, fetch-data, train-model, vol-surface, batch

## Next Steps

1. **Install**: `python setup_cli.py install`
2. **Test**: `bash test_cli.sh`
3. **Read**: `CLI_QUICKSTART.md`
4. **Use**: `bsopt --help`

## Support

- **Documentation**: See all `CLI_*.md` files
- **Issues**: Check `CLI_IMPLEMENTATION_SUMMARY.md`
- **Examples**: See `CLI_QUICKSTART.md`

---

**Last Updated**: 2024-12-13
**Version**: 2.1.0
**Status**: Complete ✅
