# Black-Scholes Advanced Option Pricing Platform v2.1

A comprehensive quantitative finance platform for option pricing, trading, and research with 32 core features spanning numerical methods, machine learning, live trading, and enterprise integrations.

## Features

### Core Pricing Engines
- **Black-Scholes Analytical Pricing**: European options with full Greeks
- **Finite Difference Methods**: Crank-Nicolson scheme for American options
- **Monte Carlo Simulation**: Variance reduction techniques (antithetic variates, control variates)
- **Lattice Models**: Binomial and trinomial trees
- **Exotic Options**: Asian, Barrier, Lookback, Digital options
- **Implied Volatility**: Newton-Raphson and Brent's method
- **Volatility Surface**: SVI calibration with no-arbitrage constraints

### Machine Learning
- **Price Prediction Models**: XGBoost, LightGBM, Random Forest, Neural Networks, LSTM
- **MLflow Integration**: Model registry, versioning, and deployment
- **Hyperparameter Tuning**: Automated optimization with Optuna
- **Feature Engineering**: Technical indicators, Greeks, market regime detection

### Trading & Integration
- **Broker Integration**: Interactive Brokers, Alpaca, Paper Trading
- **Order Management System**: Validation, risk checks, order routing
- **Real-time Market Data**: WebSocket streaming
- **Backtesting Framework**: Historical simulation with performance metrics
- **Trading Strategies**: Delta-neutral hedging, volatility arbitrage, spreads

### Frontend Dashboard
- **Interactive Pricing Calculator**: Real-time calculations with multiple methods
- **Visualizations**: Greeks charts, payoff diagrams, 3D volatility surfaces
- **Portfolio Management**: Position tracking, P&L, risk metrics
- **Market Data Dashboard**: Live option chains, heatmaps

### Infrastructure
- **FastAPI REST API**: High-performance async endpoints
- **GraphQL API**: Flexible querying with Strawberry
- **PostgreSQL + TimescaleDB**: Time-series optimization
- **Redis Caching**: Response caching and rate limiting
- **Docker Compose**: Local development environment
- **CI/CD Pipeline**: Automated testing and deployment

## Quick Start

### Prerequisites
- Docker and Docker Compose
- Python 3.11+ (for local development)
- Node.js 18+ (for frontend development)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/bsopt-platform
cd bsopt-platform
```

2. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

3. Start services with Docker Compose:
```bash
docker-compose up -d
```

4. Initialize the database:
```bash
python cli.py init-database
```

5. Access the application:
- Frontend: http://localhost:3000
- API: http://localhost:8000
- API Docs: http://localhost:8000/docs
- MLflow: http://localhost:5000

### CLI Usage

Price a single option:
```bash
python cli.py price --spot 100 --strike 100 --maturity 1.0 --volatility 0.2 --rate 0.05
```

Batch pricing from CSV:
```bash
python cli.py batch --input-file options.csv --output-file results.csv
```

Train ML model:
```bash
python cli.py train-model --model xgboost --data historical_data.csv
```

Run backtesting:
```bash
python cli.py backtest --strategy delta_neutral --start 2023-01-01 --end 2023-12-31
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Nginx (Reverse Proxy)                │
├──────────────────────┬──────────────────────────────────┤
│   React Frontend     │        FastAPI Backend           │
│   (Port 3000)        │        (Port 8000)               │
└──────────────────────┴──────────┬───────────────────────┘
                                  │
                 ┌────────────────┼────────────────┐
                 │                │                │
         ┌───────▼──────┐  ┌─────▼─────┐  ┌──────▼──────┐
         │ PostgreSQL   │  │   Redis   │  │  RabbitMQ   │
         │ TimescaleDB  │  │  Cache    │  │  Message    │
         └──────────────┘  └───────────┘  └─────┬───────┘
                                                 │
                                          ┌──────▼──────┐
                                          │   Celery    │
                                          │   Workers   │
                                          └─────────────┘
```

## Performance Targets

- **Black-Scholes**: 1M+ calculations/second (single core)
- **Crank-Nicolson**: 100x100 grid in <10ms
- **Monte Carlo**: 100K paths in <2 seconds
- **API Response**: p95 < 100ms
- **Frontend Load**: <2 seconds
- **Test Coverage**: >90%

## Development

### Running Tests
```bash
# All tests
pytest

# With coverage
pytest --cov=src --cov-report=html

# Specific test file
pytest tests/unit/test_black_scholes.py

# Performance benchmarks
pytest tests/performance/ --benchmark
```

### Code Quality
```bash
# Format code
black src/ tests/
isort src/ tests/

# Lint
flake8 src/ tests/

# Type checking
mypy src/
```

### Local Development Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"

# Start development server
python cli.py serve
```

## Documentation

- [API Documentation](docs/api/README.md)
- [User Guide](docs/user/getting-started.md)
- [Developer Guide](docs/developer/architecture.md)
- [Testing Guide](docs/developer/testing.md)
- [Deployment Guide](docs/developer/deployment.md)

## License

MIT License - see [LICENSE](LICENSE) file for details

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Support

- Issues: https://github.com/yourusername/bsopt-platform/issues
- Documentation: https://docs.bsopt.com
- Email: support@bsopt.com
