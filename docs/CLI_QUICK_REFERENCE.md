# Black-Scholes CLI - Quick Reference Card

Version 2.1.0 | One-page command cheat sheet

## Installation

```bash
cd /home/kamau/comparison
python setup_cli.py install
bsopt --version
```

## Basic Commands

```bash
# Get help
bsopt --help
bsopt price --help

# Version
bsopt --version
```

## Pricing

```bash
# Price call option
bsopt price call --spot 100 --strike 100 --maturity 1.0 --vol 0.2 --rate 0.05

# Price put option
bsopt price put --spot 100 --strike 105 --maturity 0.5 --vol 0.25 --rate 0.03

# With dividend
bsopt price call --spot 100 --strike 100 --maturity 1.0 --vol 0.2 --rate 0.05 --dividend 0.02

# Compare methods
bsopt price call --spot 100 --strike 100 --maturity 1.0 --vol 0.2 --rate 0.05 --method all

# Specific method
bsopt price call --spot 100 --strike 100 --maturity 1.0 --vol 0.2 --rate 0.05 --method fdm

# JSON output
bsopt price call --spot 100 --strike 100 --maturity 1.0 --vol 0.2 --rate 0.05 --output json
```

## Greeks

```bash
# Calculate Greeks for call
bsopt greeks --spot 100 --strike 100 --maturity 1.0 --vol 0.2 --rate 0.05

# Calculate Greeks for put
bsopt greeks --spot 100 --strike 100 --maturity 1.0 --vol 0.2 --rate 0.05 --option-type put
```

## Authentication

```bash
# Login
bsopt auth login

# Check status
bsopt auth whoami

# Logout
bsopt auth logout
```

## Portfolio

```bash
# List positions
bsopt portfolio list

# Add position
bsopt portfolio add \
  --symbol AAPL \
  --option-type call \
  --quantity 10 \
  --strike 150 \
  --maturity 0.5 \
  --vol 0.3 \
  --rate 0.05 \
  --entry-price 5.50 \
  --spot 148.50

# Remove position
bsopt portfolio remove <position_id>

# View P&L
bsopt portfolio pnl
```

## Configuration

```bash
# View all config
bsopt config list

# Get value
bsopt config get pricing.default_method

# Set value
bsopt config set pricing.default_method mc

# Reset to defaults
bsopt config reset
```

## Batch Processing

```bash
# Batch price from CSV
bsopt batch --input options.csv --output results.csv

# With Greeks
bsopt batch --input options.csv --output results.csv --compute-greeks

# Different method
bsopt batch --input options.csv --output results.csv --method fdm
```

## Server

```bash
# Start server
bsopt serve

# With auto-reload
bsopt serve --reload

# Custom port
bsopt serve --port 8080

# Production mode
bsopt serve --workers 4 --host 0.0.0.0 --port 8000
```

## Data & Analysis

```bash
# Fetch market data
bsopt fetch-data --symbol SPY --days 30

# Calculate implied volatility
bsopt implied-vol --market-price 10.50 --spot 100 --strike 100 --maturity 1.0 --rate 0.05

# Volatility surface
bsopt vol-surface --symbol SPY

# Backtest
bsopt backtest --strategy delta_neutral --start 2023-01-01 --end 2023-12-31

# Train model
bsopt train-model --algorithm xgboost --data historical.csv --output models/
```

## Common Options

| Option | Description |
|--------|-------------|
| `--spot` | Current underlying price |
| `--strike` | Strike price |
| `--maturity` | Time to maturity (years) |
| `--vol` | Annualized volatility |
| `--rate` | Risk-free interest rate |
| `--dividend` | Dividend yield (default: 0.0) |
| `--option-type` | call or put (default: call) |
| `--method` | bs, fdm, mc, all (default: bs) |
| `--output` | table, json, csv (default: table) |

## Methods

| Method | Speed | Accuracy | Use For |
|--------|-------|----------|---------|
| `bs` | Fastest | Exact | European options |
| `fdm` | Medium | High | American options |
| `mc` | Slowest | Stochastic | Complex payoffs |

## Configuration Files

**User Config**: `~/.config/bsopt/config.yml`

```yaml
pricing:
  default_method: bs
output:
  format: table
  color: true
```

**Project Config**: `./.bsopt.yml`

```yaml
pricing:
  default_method: fdm
```

**Environment Variables**:

```bash
export BS_API_BASE_URL=http://localhost:8000
export BS_PRICING_DEFAULT_METHOD=mc
export BS_OUTPUT_FORMAT=json
```

## Shell Aliases

Add to `~/.bashrc` or `~/.zshrc`:

```bash
alias bsp='bsopt price'
alias bsg='bsopt greeks'
alias bspf='bsopt portfolio'
alias bsc='bsopt config'
```

## Batch CSV Format

**Input** (`options.csv`):

```csv
symbol,spot,strike,maturity,volatility,rate,dividend,option_type
AAPL,150.00,145.00,0.25,0.30,0.05,0.015,call
GOOGL,2800.00,2750.00,0.5,0.25,0.05,0.0,put
```

**Output** (`results.csv`):

```csv
symbol,price,delta,gamma,vega,theta,rho,method,computation_time_ms
AAPL,9.23,0.59,0.019,0.379,-0.014,0.495,black_scholes,0.45
```

## Error Handling

```bash
# Check installation
which bsopt

# Reinstall
pip install -e . --force-reinstall

# Debug mode
export BS_DEBUG=true
bsopt price call --spot 100 --strike 100 --maturity 1.0 --vol 0.2 --rate 0.05

# View logs
tail -f ~/.config/bsopt/logs/bsopt.log
```

## Testing

```bash
# Run test suite
bash test_cli.sh

# Test single command
bsopt price call --spot 100 --strike 100 --maturity 1.0 --vol 0.2 --rate 0.05
```

## File Locations

| File | Location |
|------|----------|
| CLI | `/home/kamau/comparison/cli_complete.py` |
| Config | `~/.config/bsopt/config.yml` |
| Credentials | `~/.config/bsopt/credentials.enc` |
| Portfolio | `~/.config/bsopt/portfolio.json` |
| Logs | `~/.config/bsopt/logs/bsopt.log` |

## Documentation

| File | Purpose |
|------|---------|
| `CLI_README.md` | Main documentation |
| `CLI_DOCUMENTATION.md` | Complete reference |
| `CLI_QUICKSTART.md` | Getting started |
| `CLI_QUICK_REFERENCE.md` | This file |

## Performance Tips

1. Use `bs` method for European options (fastest)
2. Enable caching: `bsopt config set cache.enabled true`
3. Batch process multiple options together
4. Use `fdm` for American options
5. Reduce MC paths for testing: `bsopt config set pricing.mc_paths 10000`

## Pipeline Integration

```bash
# Extract price as JSON
bsopt price call --spot 100 --strike 100 --maturity 1.0 --vol 0.2 --rate 0.05 \
  --output json | jq '.price'

# Extract delta
bsopt greeks --spot 100 --strike 100 --maturity 1.0 --vol 0.2 --rate 0.05 \
  --output json | jq '.greeks.delta'

# Batch and filter
bsopt batch --input options.csv --output results.csv
awk -F, '$2 > 10 { print $0 }' results.csv
```

## Support

- Docs: See all `CLI_*.md` files
- Test: `bash test_cli.sh`
- Help: `bsopt --help`
- Issues: Check `CLI_IMPLEMENTATION_SUMMARY.md`

---

**Quick Start**: `python setup_cli.py install && bsopt --help`

**Status**: Production Ready ✅ | **Version**: 2.1.0
