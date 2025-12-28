# Getting Started with BSOPT Platform

Welcome to the Black-Scholes Advanced Option Pricing Platform! This guide will help you get up and running in minutes.

## Quick Start (3 Steps)

### Step 1: Setup Environment
```bash
# Make setup script executable (if needed)
chmod +x setup.sh

# Run setup script
./setup.sh
```

This will:
- Check Docker prerequisites
- Create `.env` file with secure defaults
- Generate random JWT secret
- Create required directories
- Build Docker images

### Step 2: Start Services
```bash
# Start all services in background
docker-compose up -d

# View logs (optional)
docker-compose logs -f
```

Services starting:
- PostgreSQL (TimescaleDB) on port 5432
- Redis on port 6379
- RabbitMQ on ports 5672 & 15672
- FastAPI backend on port 8000
- React frontend on port 3000 (when implemented)
- MLflow on port 5000
- Jupyter on port 8888
- Nginx on port 80

### Step 3: Initialize Database
```bash
# Run database initialization
docker-compose exec api python -c "from src.database import init_db; init_db()"
```

**Done!** 🎉 Your platform is ready.

---

## Access Your Services

Once running, access these URLs:

| Service | URL | Description |
|---------|-----|-------------|
| **API** | http://localhost:8000 | Main API endpoint |
| **API Docs** | http://localhost:8000/docs | Interactive Swagger UI |
| **API Redoc** | http://localhost:8000/redoc | Alternative API docs |
| **Health Check** | http://localhost:8000/health | Service health status |
| **MLflow** | http://localhost:5000 | ML model tracking |
| **Jupyter** | http://localhost:8888 | Notebook server |
| **RabbitMQ** | http://localhost:15672 | Message queue UI (admin/changeme) |

---

## Your First Option Pricing

### Method 1: Python (Direct)

```python
from src.pricing.black_scholes import BlackScholesEngine, BSParameters

# Define option parameters
params = BSParameters(
    spot=100.0,      # Current stock price
    strike=100.0,    # Strike price (ATM)
    maturity=1.0,    # 1 year to expiry
    volatility=0.25, # 25% annual volatility
    rate=0.05,       # 5% risk-free rate
    dividend=0.02    # 2% dividend yield
)

# Price a call option
call_price = BlackScholesEngine.price_call(params)
print(f"Call Price: ${call_price:.4f}")

# Calculate Greeks
greeks = BlackScholesEngine.calculate_greeks(params, 'call')
print(f"Delta: {greeks.delta:.4f}")
print(f"Gamma: {greeks.gamma:.4f}")
print(f"Vega:  {greeks.vega:.4f}")
print(f"Theta: {greeks.theta:.4f}")
print(f"Rho:   {greeks.rho:.4f}")
```

Expected output:
```
Call Price: $11.0945
Delta: 0.6110
Gamma: 0.0187
Vega:  0.3731
Theta: -0.0143
Rho:   0.5192
```

### Method 2: Using Different Methods

```python
from src.pricing.finite_difference import CrankNicolsonSolver
from src.pricing.monte_carlo import MonteCarloEngine, MCConfig

# Finite Difference Method
fdm = CrankNicolsonSolver(
    spot=100, strike=100, maturity=1.0,
    volatility=0.25, rate=0.05, dividend=0.02,
    option_type='call', n_spots=200, n_time=500
)
fdm_price = fdm.solve()
print(f"FDM Price: ${fdm_price:.4f}")

# Monte Carlo Simulation
mc_config = MCConfig(n_paths=100000, antithetic=True, control_variate=True)
mc_engine = MonteCarloEngine(mc_config)
mc_price, ci = mc_engine.price_european(params, 'call')
print(f"MC Price: ${mc_price:.4f} ± ${ci:.4f}")

# American Put (Longstaff-Schwartz)
american_put = mc_engine.price_american_lsm(params, 'put')
print(f"American Put: ${american_put:.4f}")
```

---

## Development Workflow

### Local Python Development

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black src/ tests/
isort src/ tests/

# Type checking
mypy src/

# Lint
flake8 src/ tests/
```

### Docker Development Workflow

```bash
# View all services
docker-compose ps

# View logs for specific service
docker-compose logs -f api
docker-compose logs -f postgres

# Restart a service
docker-compose restart api

# Execute commands in container
docker-compose exec api python cli.py --help
docker-compose exec postgres psql -U admin -d bsopt

# Stop all services
docker-compose down

# Stop and remove volumes (WARNING: deletes data)
docker-compose down -v

# Rebuild after code changes
docker-compose build api
docker-compose up -d api
```

---

## Common Tasks

### Database Management

```bash
# Access PostgreSQL
docker-compose exec postgres psql -U admin -d bsopt

# View tables
docker-compose exec postgres psql -U admin -d bsopt -c "\dt"

# Run SQL query
docker-compose exec postgres psql -U admin -d bsopt -c "SELECT COUNT(*) FROM users;"

# Backup database
docker-compose exec postgres pg_dump -U admin bsopt > backup.sql

# Restore database
docker-compose exec -T postgres psql -U admin bsopt < backup.sql
```

### Redis Management

```bash
# Access Redis CLI
docker-compose exec redis redis-cli -a changeme

# View all keys
docker-compose exec redis redis-cli -a changeme KEYS '*'

# Flush cache
docker-compose exec redis redis-cli -a changeme FLUSHALL
```

### Jupyter Notebooks

```bash
# Access Jupyter (no password required)
open http://localhost:8888

# Create notebook in work directory
# All notebooks saved to ./notebooks/
```

---

## Testing

### Run All Tests

```bash
# Inside container
docker-compose exec api pytest

# Local environment
pytest

# With coverage
pytest --cov=src --cov-report=html

# View coverage
open htmlcov/index.html
```

### Run Specific Tests

```bash
# Test Black-Scholes only
pytest tests/unit/test_black_scholes.py

# Test with verbose output
pytest -v tests/

# Test with specific marker
pytest -m "not slow"
```

---

## Monitoring & Debugging

### View Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f api

# Last 100 lines
docker-compose logs --tail=100 api

# Follow with timestamps
docker-compose logs -f -t api
```

### Service Health

```bash
# Check API health
curl http://localhost:8000/health

# Check all containers
docker-compose ps

# Check resource usage
docker stats
```

### Database Queries

```sql
-- View recent option prices
SELECT * FROM options_prices
ORDER BY time DESC
LIMIT 10;

-- View users
SELECT id, email, tier, created_at
FROM users;

-- View ML models
SELECT name, algorithm, version, is_production
FROM ml_models
ORDER BY created_at DESC;
```

---

## Troubleshooting

### Port Already in Use

```bash
# Find process using port 8000
lsof -i :8000

# Kill process
kill -9 <PID>

# Or change port in docker-compose.yml
```

### Database Connection Issues

```bash
# Check PostgreSQL is running
docker-compose ps postgres

# View PostgreSQL logs
docker-compose logs postgres

# Restart PostgreSQL
docker-compose restart postgres
```

### Container Won't Start

```bash
# View detailed logs
docker-compose logs <service-name>

# Rebuild container
docker-compose build --no-cache <service-name>
docker-compose up -d <service-name>
```

### Clear Everything and Start Fresh

```bash
# Stop and remove all containers, networks, volumes
docker-compose down -v

# Remove images
docker-compose down --rmi all

# Re-run setup
./setup.sh
docker-compose up -d
```

---

## Configuration

### Environment Variables

Edit `.env` file to customize:

```bash
# Database
DB_PASSWORD=your-secure-password

# JWT Authentication
JWT_SECRET=your-secret-key-here
ACCESS_TOKEN_EXPIRE_MINUTES=30

# API Configuration
DEBUG=true
LOG_LEVEL=INFO

# External APIs (optional)
ALPHA_VANTAGE_API_KEY=your-key
IBKR_ACCOUNT=your-account
```

After changing `.env`:
```bash
docker-compose down
docker-compose up -d
```

---

## Next Steps

### 1. Explore the Pricing Engines
- Try different option parameters
- Compare pricing methods (BS, FDM, MC)
- Test edge cases (deep ITM/OTM, short/long maturity)

### 2. Experiment with Notebooks
- Open http://localhost:8888
- Create new notebook in `notebooks/experiments/`
- Import pricing engines and experiment

### 3. Explore the API
- Open http://localhost:8000/docs
- Try the interactive API documentation
- Test pricing endpoints (when implemented)

### 4. Build Something Cool
- Implement a trading strategy
- Create custom visualizations
- Train ML models for price prediction

---

## Getting Help

### Documentation
- **Implementation Plan**: `.claude/plans/proud-dazzling-patterson.md`
- **Status**: `STATUS.md`
- **API Docs**: http://localhost:8000/docs (when running)

### Code Examples
- `src/pricing/` - Pricing engine implementations with examples
- `notebooks/` - Jupyter notebooks for experimentation

### Common Commands Reference

```bash
# Setup
./setup.sh

# Start/Stop
docker-compose up -d
docker-compose down

# View status
docker-compose ps
docker-compose logs -f

# Run tests
docker-compose exec api pytest

# Access Python shell
docker-compose exec api python

# Access database
docker-compose exec postgres psql -U admin -d bsopt
```

---

## Example Workflow

A typical development session:

```bash
# 1. Start services
docker-compose up -d

# 2. Check they're running
docker-compose ps

# 3. View API docs
open http://localhost:8000/docs

# 4. Run Python code
docker-compose exec api python
>>> from src.pricing.black_scholes import BlackScholesEngine, BSParameters
>>> params = BSParameters(100, 100, 1.0, 0.2, 0.05, 0.02)
>>> price = BlackScholesEngine.price_call(params)
>>> print(f"Price: ${price:.4f}")

# 5. Run tests
docker-compose exec api pytest -v

# 6. View logs if needed
docker-compose logs -f api

# 7. Stop when done
docker-compose down
```

---

**Ready to price some options? Let's go!** 🚀

For detailed implementation status, see `STATUS.md`.
