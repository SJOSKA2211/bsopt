# Black-Scholes CLI Documentation

Comprehensive command-line interface for the Black-Scholes Option Pricing Platform.

## Installation

### Quick Install

```bash
# Install CLI globally or in virtual environment
python setup_cli.py install

# Or for development (editable mode)
python setup_cli.py develop
```

### Manual Install

```bash
pip install -e .
```

### Verify Installation

```bash
bsopt --version
bsopt --help
```

## Quick Start

### 1. Login (if using API)

```bash
bsopt auth login
# Enter email and password when prompted
```

### 2. Price an Option

```bash
# Price a call option
bsopt price call --spot 100 --strike 100 --maturity 1.0 --vol 0.2 --rate 0.05

# Price a put option
bsopt price put --spot 100 --strike 105 --maturity 0.5 --vol 0.25 --rate 0.03
```

### 3. Calculate Greeks

```bash
bsopt greeks --spot 100 --strike 100 --maturity 1.0 --vol 0.2 --rate 0.05
```

### 4. Manage Portfolio

```bash
# List portfolio
bsopt portfolio list

# Add position
bsopt portfolio add --symbol AAPL --option-type call --quantity 10 \
    --strike 150 --maturity 0.5 --vol 0.3 --rate 0.05 \
    --entry-price 5.50 --spot 148.50

# View P&L
bsopt portfolio pnl
```

## Command Reference

### Authentication Commands

#### `bsopt auth login`

Authenticate with the platform.

```bash
bsopt auth login
# Or specify email directly
bsopt auth login --email user@example.com
```

**Storage**: Tokens are stored securely in:
- macOS: Keychain
- Windows: Credential Manager
- Linux: Secret Service or encrypted file at `~/.config/bsopt/credentials.enc`

#### `bsopt auth logout`

Clear stored authentication tokens.

```bash
bsopt auth logout
```

#### `bsopt auth whoami`

Display current user information.

```bash
bsopt auth whoami
```

### Pricing Commands

#### `bsopt price [call|put]`

Price options using multiple methods.

**Required Options**:
- `--spot`: Current underlying price
- `--strike`: Strike price
- `--maturity`: Time to maturity in years
- `--vol`: Annualized volatility
- `--rate`: Risk-free interest rate

**Optional**:
- `--dividend`: Dividend yield (default: 0.0)
- `--method`: Pricing method (bs, fdm, mc, all) (default: bs)
- `--output`: Output format (table, json, csv) (default: table)

**Examples**:

```bash
# Black-Scholes pricing
bsopt price call --spot 100 --strike 100 --maturity 1.0 --vol 0.2 --rate 0.05

# Finite Difference Method
bsopt price call --spot 100 --strike 100 --maturity 1.0 --vol 0.2 --rate 0.05 --method fdm

# Monte Carlo
bsopt price call --spot 100 --strike 100 --maturity 1.0 --vol 0.2 --rate 0.05 --method mc

# Compare all methods
bsopt price call --spot 100 --strike 100 --maturity 1.0 --vol 0.2 --rate 0.05 --method all

# JSON output
bsopt price call --spot 100 --strike 100 --maturity 1.0 --vol 0.2 --rate 0.05 --output json
```

**Pricing Methods**:

| Method | Description | Speed | Accuracy |
|--------|-------------|-------|----------|
| `bs` | Black-Scholes analytical | Fastest | Exact for European |
| `fdm` | Finite Difference (Crank-Nicolson) | Medium | High |
| `mc` | Monte Carlo simulation | Slowest | Stochastic |
| `all` | Compare all methods | Slow | Comprehensive |

#### `bsopt greeks`

Calculate option Greeks (sensitivity measures).

**Options**: Same as `price` command

**Example**:

```bash
bsopt greeks --spot 100 --strike 100 --maturity 1.0 --vol 0.2 --rate 0.05 \
    --option-type call --method bs
```

**Output**:

| Greek | Description | Interpretation |
|-------|-------------|----------------|
| Delta | ∂V/∂S | Change in option price per $1 move in underlying |
| Gamma | ∂²V/∂S² | Change in delta per $1 move in underlying |
| Vega | ∂V/∂σ | Change in option price per 1% volatility change |
| Theta | ∂V/∂t | Daily time decay |
| Rho | ∂V/∂r | Change in option price per 1% rate change |

#### `bsopt implied-vol`

Calculate implied volatility from market price.

**Options**:
- `--market-price`: Observed market price (required)
- `--spot`: Current spot price (required)
- `--strike`: Strike price (required)
- `--maturity`: Time to maturity (required)
- `--rate`: Risk-free rate (required)
- `--dividend`: Dividend yield (default: 0.0)
- `--option-type`: call or put (default: call)

**Example**:

```bash
bsopt implied-vol --market-price 10.50 --spot 100 --strike 100 \
    --maturity 1.0 --rate 0.05
```

### Portfolio Commands

#### `bsopt portfolio list`

Display all portfolio positions.

```bash
bsopt portfolio list
```

**Output**: Table showing positions with:
- Position ID
- Symbol
- Option type (CALL/PUT)
- Quantity
- Strike price
- Entry price
- Current price
- P&L

#### `bsopt portfolio add`

Add new position to portfolio.

**Required Options**:
- `--symbol`: Underlying symbol
- `--option-type`: call or put
- `--quantity`: Number of contracts (negative for short)
- `--strike`: Strike price
- `--maturity`: Time to maturity
- `--vol`: Implied volatility
- `--rate`: Risk-free rate
- `--entry-price`: Entry price per contract
- `--spot`: Current underlying price

**Optional**:
- `--dividend`: Dividend yield (default: 0.0)

**Example**:

```bash
bsopt portfolio add --symbol AAPL --option-type call --quantity 10 \
    --strike 150 --maturity 0.5 --vol 0.3 --rate 0.05 \
    --entry-price 5.50 --spot 148.50
```

#### `bsopt portfolio remove`

Remove position from portfolio.

```bash
bsopt portfolio remove <position_id>
```

**Example**:

```bash
bsopt portfolio remove abc12345
```

#### `bsopt portfolio pnl`

Display detailed P&L report with portfolio Greeks.

```bash
bsopt portfolio pnl
```

**Output**:
- Total portfolio value
- Cost basis
- Realized/unrealized P&L
- P&L percentage
- Portfolio Greeks (delta, gamma, vega, theta, rho)

### Batch Processing

#### `bsopt batch`

Price multiple options from CSV file.

**Options**:
- `--input`: Input CSV file (required)
- `--output`: Output CSV file (required)
- `--method`: Pricing method (default: bs)
- `--compute-greeks`: Include Greeks in output (flag)

**Input CSV Format**:

```csv
symbol,spot,strike,maturity,volatility,rate,dividend,option_type
AAPL,150.00,145.00,0.25,0.30,0.05,0.015,call
GOOGL,2800.00,2750.00,0.5,0.25,0.05,0.0,put
MSFT,380.00,375.00,0.33,0.28,0.05,0.01,call
```

**Output CSV Format**:

```csv
symbol,price,delta,gamma,vega,theta,rho,method,computation_time_ms
AAPL,9.23,0.59,0.019,0.379,-0.014,0.495,black_scholes,0.45
GOOGL,12.45,-0.42,0.015,0.325,-0.011,0.387,black_scholes,0.38
...
```

**Example**:

```bash
bsopt batch --input options.csv --output results.csv --method bs --compute-greeks
```

### Configuration

#### `bsopt config list`

Display current configuration.

```bash
bsopt config list
```

#### `bsopt config get`

Get specific configuration value.

```bash
bsopt config get api.base_url
bsopt config get pricing.default_method
```

#### `bsopt config set`

Set configuration value.

**Options**:
- `--scope`: user or project (default: user)

```bash
bsopt config set api.base_url http://api.example.com
bsopt config set pricing.default_method mc
bsopt config set output.format json --scope project
```

#### `bsopt config reset`

Reset configuration to defaults.

```bash
bsopt config reset
bsopt config reset --scope project
```

### Server Management

#### `bsopt serve`

Start FastAPI server.

**Options**:
- `--host`: Host to bind (default: 0.0.0.0)
- `--port`: Port to bind (default: 8000)
- `--reload`: Enable auto-reload (flag)
- `--workers`: Number of worker processes (default: 1)

**Examples**:

```bash
# Development mode with auto-reload
bsopt serve --reload

# Production with multiple workers
bsopt serve --workers 4 --host 0.0.0.0 --port 8000
```

#### `bsopt init-db`

Initialize database schema.

**Options**:
- `--seed`: Seed with sample data (flag)
- `--force`: Force recreate (drops existing) (flag)

```bash
bsopt init-db
bsopt init-db --seed
bsopt init-db --force --seed
```

### Data Management

#### `bsopt fetch-data`

Download market data from providers.

**Options**:
- `--symbol`: Stock symbol (required)
- `--days`: Number of days (default: 30)
- `--provider`: Data provider (yahoo, alphavantage, polygon) (default: yahoo)
- `--output`: Output file path (optional)

**Examples**:

```bash
bsopt fetch-data --symbol SPY --days 30
bsopt fetch-data --symbol AAPL --days 90 --provider yahoo --output aapl_data.csv
```

#### `bsopt vol-surface`

Generate volatility surface visualization.

**Options**:
- `--symbol`: Underlying symbol (required)
- `--date`: Date (YYYY-MM-DD) (optional, default: today)
- `--output`: Output image path (optional)

```bash
bsopt vol-surface --symbol SPY
bsopt vol-surface --symbol AAPL --date 2024-01-15 --output aapl_vol.png
```

### Backtesting

#### `bsopt backtest`

Run strategy backtesting.

**Options**:
- `--strategy`: Strategy name (required)
  - `delta_neutral`: Delta-neutral hedging
  - `iron_condor`: Iron condor spreads
  - `straddle`: Long straddle
- `--start`: Start date YYYY-MM-DD (required)
- `--end`: End date YYYY-MM-DD (required)
- `--capital`: Initial capital (default: 100000)
- `--output`: Output report path (optional)

**Example**:

```bash
bsopt backtest --strategy delta_neutral --start 2023-01-01 --end 2023-12-31 \
    --capital 100000 --output backtest_report.json
```

### Machine Learning

#### `bsopt train-model`

Train ML model for option pricing.

**Options**:
- `--algorithm`: Algorithm (xgboost, lightgbm, random_forest, neural_network) (default: xgboost)
- `--data`: Training data CSV (required)
- `--output`: Output model directory (required)
- `--test-split`: Test split ratio (default: 0.2)

**Example**:

```bash
bsopt train-model --algorithm xgboost --data historical_data.csv --output models/
```

## Configuration Files

### User Configuration

Location: `~/.config/bsopt/config.yml`

```yaml
api:
  base_url: http://localhost:8000
  timeout: 30
  verify_ssl: true

pricing:
  default_method: bs
  default_option_type: call
  mc_paths: 100000
  mc_steps: 252
  fdm_spots: 200

output:
  format: table  # table, json, csv
  color: true
  verbose: false
  precision: 4

cache:
  enabled: true
  ttl: 3600
```

### Project Configuration

Location: `./.bsopt.yml` (in project directory)

```yaml
api:
  base_url: http://production-api.example.com

pricing:
  default_method: mc
  mc_paths: 500000
```

### Environment Variables

All configuration can be overridden with environment variables:

```bash
export BS_API_BASE_URL=http://api.example.com
export BS_PRICING_DEFAULT_METHOD=mc
export BS_OUTPUT_FORMAT=json
```

Format: `BS_<SECTION>_<KEY>` in uppercase.

## Output Formats

### Table (Default)

Beautiful formatted tables with colors:

```
Option Price: $10.5234
Method: BS | Time: 0.45ms

┌─────────┬──────────┬─────────────────────┐
│ Greek   │ Value    │ Description         │
├─────────┼──────────┼─────────────────────┤
│ Delta   │  0.5869  │ ∂V/∂S              │
│ Gamma   │  0.0190  │ ∂²V/∂S²            │
└─────────┴──────────┴─────────────────────┘
```

### JSON

Machine-readable JSON output:

```bash
bsopt price call --spot 100 --strike 100 --maturity 1.0 --vol 0.2 --rate 0.05 --output json
```

```json
{
  "price": 10.5234,
  "method": "bs",
  "option_type": "call",
  "computation_time_ms": 0.45,
  "greeks": {
    "delta": 0.5869,
    "gamma": 0.0190,
    "vega": 0.3790,
    "theta": -0.0139,
    "rho": 0.4946
  }
}
```

### CSV

For batch processing and spreadsheets:

```bash
bsopt batch --input options.csv --output results.csv
```

## Error Handling

The CLI provides clear error messages with suggestions:

```bash
$ bsopt price call --spot -100 --strike 100 --maturity 1.0 --vol 0.2 --rate 0.05
Error: Spot price must be positive, got -100

$ bsopt portfolio add --symbol AAPL --quantity 10
Error: Missing required options: --strike, --maturity, --vol, --rate, --entry-price, --spot
```

## Tips and Best Practices

### 1. Use Configuration Files

Set defaults in config file instead of passing same options repeatedly:

```yaml
# ~/.config/bsopt/config.yml
pricing:
  default_method: bs
output:
  format: json
  precision: 6
```

### 2. Shell Aliases

Add aliases to your `.bashrc` or `.zshrc`:

```bash
alias bsp='bsopt price'
alias bsg='bsopt greeks'
alias bspf='bsopt portfolio'
```

### 3. Batch Processing

For multiple options, use batch mode instead of loops:

```bash
# Good: Single batch command
bsopt batch --input options.csv --output results.csv

# Avoid: Multiple single commands in loop
for option in options.txt; do bsopt price ...; done
```

### 4. Environment-Specific Config

Use project config files for different environments:

```bash
# Development
cd ~/projects/dev && bsopt serve --reload

# Production
cd ~/projects/prod && bsopt serve --workers 4
```

### 5. Pipeline Integration

Use JSON output for pipeline integration:

```bash
# Price option and extract delta
bsopt price call --spot 100 --strike 100 --maturity 1.0 --vol 0.2 --rate 0.05 \
  --output json | jq '.greeks.delta'

# Batch process and filter
bsopt batch --input options.csv --output results.csv
cat results.csv | awk -F, '$2 > 10 { print $0 }'
```

## Troubleshooting

### Authentication Issues

```bash
# Check current auth status
bsopt auth whoami

# Re-login if token expired
bsopt auth logout
bsopt auth login
```

### Configuration Issues

```bash
# View current config
bsopt config list

# Reset to defaults
bsopt config reset --confirm

# Check environment variables
env | grep BS_
```

### Performance Issues

```bash
# Use faster Black-Scholes method
bsopt config set pricing.default_method bs

# Reduce Monte Carlo paths for testing
bsopt config set pricing.mc_paths 10000

# Enable caching
bsopt config set cache.enabled true
```

## Support

- Documentation: `/home/kamau/comparison/docs/`
- Issues: Create an issue in the repository
- API Docs: Run `bsopt serve` and visit http://localhost:8000/docs

## License

Copyright (c) 2024 Black-Scholes Platform Team
