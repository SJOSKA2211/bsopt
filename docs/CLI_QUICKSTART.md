# Black-Scholes CLI - Quick Start Guide

Get started with the Black-Scholes Option Pricing CLI in 5 minutes.

## Installation

### Step 1: Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install CLI
pip install -e .

# Or use the setup script
python setup_cli.py install
```

### Step 2: Verify Installation

```bash
bsopt --version
# Output: bsopt, version 2.1.0

bsopt --help
# Shows all available commands
```

## Basic Usage

### 1. Price a Call Option

```bash
bsopt price call \
  --spot 100 \
  --strike 100 \
  --maturity 1.0 \
  --vol 0.2 \
  --rate 0.05
```

**Output:**
```
Option Price: $10.4506

Greeks:
  Delta:  0.6368
  Gamma:  0.0184
  Vega:   0.3970
  Theta: -0.0151
  Rho:    0.5320

Computation Time: 0.42ms
```

### 2. Calculate Greeks

```bash
bsopt greeks \
  --spot 100 \
  --strike 100 \
  --maturity 1.0 \
  --vol 0.2 \
  --rate 0.05 \
  --option-type call
```

### 3. Compare Methods

```bash
bsopt price call \
  --spot 100 \
  --strike 100 \
  --maturity 1.0 \
  --vol 0.2 \
  --rate 0.05 \
  --method all
```

**Output:**
```
Method Comparison
┌─────────────────┬──────────┬─────────────┐
│ Method          │ Price    │ Time (ms)   │
├─────────────────┼──────────┼─────────────┤
│ Black-Scholes   │ $10.4506 │ 0.42        │
│ Crank-Nicolson  │ $10.4512 │ 15.23       │
│ Monte Carlo     │ $10.4287 │ 1,234.56    │
└─────────────────┴──────────┴─────────────┘
```

### 4. Batch Processing

Create `options.csv`:
```csv
symbol,spot,strike,maturity,volatility,rate,dividend,option_type
AAPL,150.00,145.00,0.25,0.30,0.05,0.015,call
GOOGL,2800.00,2750.00,0.5,0.25,0.05,0.0,put
MSFT,380.00,375.00,0.33,0.28,0.05,0.01,call
```

Run batch pricing:
```bash
bsopt batch \
  --input options.csv \
  --output results.csv \
  --method bs \
  --compute-greeks
```

### 5. Portfolio Management

```bash
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

# List positions
bsopt portfolio list

# View P&L
bsopt portfolio pnl
```

### 6. Configuration

```bash
# Set default method
bsopt config set pricing.default_method mc

# Set output format
bsopt config set output.format json

# View all config
bsopt config list
```

## Common Workflows

### Workflow 1: Daily Option Analysis

```bash
# 1. Fetch current market data
bsopt fetch-data --symbol AAPL --days 30

# 2. Price options
bsopt price call --spot 150 --strike 155 --maturity 0.5 --vol 0.3 --rate 0.05

# 3. Calculate Greeks
bsopt greeks --spot 150 --strike 155 --maturity 0.5 --vol 0.3 --rate 0.05

# 4. Add to portfolio
bsopt portfolio add --symbol AAPL --option-type call --quantity 10 \
  --strike 155 --maturity 0.5 --vol 0.3 --rate 0.05 \
  --entry-price 6.25 --spot 150
```

### Workflow 2: Risk Analysis

```bash
# 1. View portfolio
bsopt portfolio list

# 2. Calculate portfolio Greeks
bsopt portfolio pnl

# 3. Generate vol surface
bsopt vol-surface --symbol SPY --output spy_vol.png
```

### Workflow 3: Strategy Backtesting

```bash
# 1. Initialize database
bsopt init-db --seed

# 2. Run backtest
bsopt backtest \
  --strategy delta_neutral \
  --start 2023-01-01 \
  --end 2023-12-31 \
  --capital 100000 \
  --output backtest_results.json

# 3. Analyze results
cat backtest_results.json | jq '.metrics'
```

## Advanced Features

### Authentication (if using API)

```bash
# Login
bsopt auth login

# Check status
bsopt auth whoami

# Logout
bsopt auth logout
```

### Server Mode

```bash
# Start development server
bsopt serve --reload

# Start production server
bsopt serve --workers 4 --host 0.0.0.0 --port 8000
```

### JSON Output for Pipelines

```bash
# Get price as JSON
bsopt price call --spot 100 --strike 100 --maturity 1.0 --vol 0.2 --rate 0.05 \
  --output json | jq '.price'

# Extract delta
bsopt greeks --spot 100 --strike 100 --maturity 1.0 --vol 0.2 --rate 0.05 \
  --output json | jq '.greeks.delta'
```

## Configuration Examples

### User Config (~/.config/bsopt/config.yml)

```yaml
api:
  base_url: http://localhost:8000
  timeout: 30

pricing:
  default_method: bs
  mc_paths: 100000
  fdm_spots: 200

output:
  format: table
  color: true
  precision: 4
```

### Environment Variables

```bash
export BS_API_BASE_URL=http://api.example.com
export BS_PRICING_DEFAULT_METHOD=mc
export BS_OUTPUT_FORMAT=json
export BS_OUTPUT_PRECISION=6
```

## Shell Integration

### Bash Completion

Add to `~/.bashrc`:

```bash
eval "$(_BSOPT_COMPLETE=bash_source bsopt)"
```

### Aliases

Add to `~/.bashrc` or `~/.zshrc`:

```bash
alias bsp='bsopt price'
alias bsg='bsopt greeks'
alias bspf='bsopt portfolio'
alias bsc='bsopt config'
```

Then use:
```bash
bsp call --spot 100 --strike 100 --maturity 1.0 --vol 0.2 --rate 0.05
```

## Troubleshooting

### Issue: Command not found

```bash
# Ensure installation completed
pip list | grep bsopt

# Reinstall
pip install -e . --force-reinstall
```

### Issue: Import errors

```bash
# Install all dependencies
pip install -r requirements_cli.txt
```

### Issue: Permission denied

```bash
# Make CLI executable
chmod +x cli_complete.py

# Or use python directly
python cli_complete.py --help
```

### Issue: Authentication failed

```bash
# Clear stored credentials
bsopt auth logout

# Login again
bsopt auth login
```

## Next Steps

1. Read full documentation: `CLI_DOCUMENTATION.md`
2. Explore examples: `examples/` directory
3. View API docs: Run `bsopt serve` and visit http://localhost:8000/docs
4. Join community: GitHub Discussions

## Support

- Documentation: `./docs/`
- Examples: `./examples/`
- Issues: GitHub Issues
- Email: support@bsopt.com

---

Happy pricing! 🚀📈
