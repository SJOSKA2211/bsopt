# API Tutorial: Complete Guide to the Black-Scholes Option Pricing API

## Table of Contents

1. [Quick Start](#quick-start)
2. [Authentication](#authentication)
3. [Core Pricing Endpoints](#core-pricing-endpoints)
4. [Advanced Pricing Methods](#advanced-pricing-methods)
5. [Portfolio Management](#portfolio-management)
6. [Machine Learning API](#machine-learning-api)
7. [WebSocket Real-Time Data](#websocket-real-time-data)
8. [Error Handling](#error-handling)
9. [Rate Limiting](#rate-limiting)
10. [Best Practices](#best-practices)

## Quick Start

### Base URL

```
Production: https://api.bsopt.com/api/v1
Staging:    https://staging-api.bsopt.com/api/v1
Local:      http://localhost:8000/api/v1
```

### Your First Request

Price a simple European call option:

```bash
curl -X POST "http://localhost:8000/api/v1/pricing/black-scholes" \
  -H "Content-Type: application/json" \
  -d '{
    "spot": 100.0,
    "strike": 100.0,
    "maturity": 1.0,
    "volatility": 0.20,
    "rate": 0.05,
    "dividend": 0.02,
    "option_type": "call"
  }'
```

**Response**:
```json
{
  "price": 10.8336,
  "greeks": {
    "delta": 0.5693,
    "gamma": 0.0188,
    "vega": 37.5844,
    "theta": -0.0172,
    "rho": 44.2891
  },
  "metadata": {
    "method": "black_scholes",
    "computation_time_ms": 1.23,
    "timestamp": "2025-12-13T10:30:00Z"
  }
}
```

## Authentication

### Registration

Create a new user account:

```bash
curl -X POST "http://localhost:8000/api/v1/auth/register" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "trader@example.com",
    "password": "SecurePassword123!",
    "full_name": "John Trader"
  }'
```

**Response**:
```json
{
  "id": "usr_1234567890",
  "email": "trader@example.com",
  "full_name": "John Trader",
  "created_at": "2025-12-13T10:00:00Z",
  "role": "trader"
}
```

### Login

Obtain JWT access token:

```bash
curl -X POST "http://localhost:8000/api/v1/auth/login" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=trader@example.com&password=SecurePassword123!"
```

**Response**:
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 900
}
```

### Using the Token

Include the access token in the `Authorization` header for all authenticated requests:

```bash
curl -X GET "http://localhost:8000/api/v1/portfolio/list" \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
```

### Token Refresh

When the access token expires (15 minutes), use the refresh token:

```bash
curl -X POST "http://localhost:8000/api/v1/auth/refresh" \
  -H "Content-Type: application/json" \
  -d '{
    "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
  }'
```

## Core Pricing Endpoints

### 1. Black-Scholes Pricing

Price European options using the Black-Scholes-Merton model.

**Endpoint**: `POST /api/v1/pricing/black-scholes`

**Request Body**:
```json
{
  "spot": 100.0,          // Current asset price (required, > 0)
  "strike": 105.0,        // Strike price (required, > 0)
  "maturity": 0.5,        // Time to maturity in years (required, > 0)
  "volatility": 0.25,     // Annualized volatility (required, > 0)
  "rate": 0.05,           // Risk-free rate (required)
  "dividend": 0.02,       // Dividend yield (optional, default: 0.0)
  "option_type": "call"   // "call" or "put" (required)
}
```

**Response**:
```json
{
  "price": 7.4853,
  "greeks": {
    "delta": 0.4721,
    "gamma": 0.0224,
    "vega": 27.9856,
    "theta": -0.0214,
    "rho": 21.6342
  },
  "intrinsic_value": 0.0,
  "time_value": 7.4853,
  "moneyness": 0.9524,
  "metadata": {
    "method": "black_scholes",
    "computation_time_ms": 0.87,
    "timestamp": "2025-12-13T10:30:00Z",
    "version": "2.1.0"
  }
}
```

**Python Example**:
```python
import requests

url = "http://localhost:8000/api/v1/pricing/black-scholes"
headers = {"Content-Type": "application/json"}

payload = {
    "spot": 100.0,
    "strike": 105.0,
    "maturity": 0.5,
    "volatility": 0.25,
    "rate": 0.05,
    "dividend": 0.02,
    "option_type": "call"
}

response = requests.post(url, json=payload, headers=headers)
result = response.json()

print(f"Option Price: ${result['price']:.4f}")
print(f"Delta: {result['greeks']['delta']:.4f}")
```

### 2. Batch Pricing

Price multiple options in a single request.

**Endpoint**: `POST /api/v1/pricing/batch`

**Request Body**:
```json
{
  "options": [
    {
      "spot": 100.0,
      "strike": 95.0,
      "maturity": 1.0,
      "volatility": 0.20,
      "rate": 0.05,
      "dividend": 0.0,
      "option_type": "call"
    },
    {
      "spot": 100.0,
      "strike": 100.0,
      "maturity": 1.0,
      "volatility": 0.20,
      "rate": 0.05,
      "dividend": 0.0,
      "option_type": "call"
    },
    {
      "spot": 100.0,
      "strike": 105.0,
      "maturity": 1.0,
      "volatility": 0.20,
      "rate": 0.05,
      "dividend": 0.0,
      "option_type": "call"
    }
  ],
  "method": "black_scholes"
}
```

**Response**:
```json
{
  "results": [
    {
      "price": 13.2701,
      "greeks": { "delta": 0.6736, "gamma": 0.0171, ... },
      "strike": 95.0
    },
    {
      "price": 10.4506,
      "greeks": { "delta": 0.5968, "gamma": 0.0193, ... },
      "strike": 100.0
    },
    {
      "price": 8.0216,
      "greeks": { "delta": 0.5127, "gamma": 0.0199, ... },
      "strike": 105.0
    }
  ],
  "summary": {
    "total_options": 3,
    "successful": 3,
    "failed": 0,
    "computation_time_ms": 2.34
  }
}
```

### 3. Implied Volatility

Calculate implied volatility from market price.

**Endpoint**: `POST /api/v1/pricing/implied-volatility`

**Request Body**:
```json
{
  "market_price": 10.45,
  "spot": 100.0,
  "strike": 100.0,
  "maturity": 1.0,
  "rate": 0.05,
  "dividend": 0.0,
  "option_type": "call",
  "method": "newton_raphson",  // or "brent"
  "initial_guess": 0.20,       // optional
  "tolerance": 1e-6,           // optional
  "max_iterations": 100        // optional
}
```

**Response**:
```json
{
  "implied_volatility": 0.1998,
  "iterations": 4,
  "error": 3.2e-7,
  "converged": true,
  "verification": {
    "calculated_price": 10.4499,
    "market_price": 10.4500,
    "price_difference": 0.0001
  },
  "metadata": {
    "method": "newton_raphson",
    "computation_time_ms": 0.42
  }
}
```

**JavaScript Example**:
```javascript
const axios = require('axios');

async function calculateImpliedVol() {
  const response = await axios.post(
    'http://localhost:8000/api/v1/pricing/implied-volatility',
    {
      market_price: 10.45,
      spot: 100.0,
      strike: 100.0,
      maturity: 1.0,
      rate: 0.05,
      dividend: 0.0,
      option_type: 'call',
      method: 'newton_raphson'
    }
  );

  console.log(`Implied Volatility: ${(response.data.implied_volatility * 100).toFixed(2)}%`);
}
```

## Advanced Pricing Methods

### 1. Monte Carlo Simulation

Price options using Monte Carlo path simulation.

**Endpoint**: `POST /api/v1/pricing/monte-carlo`

**Request Body**:
```json
{
  "spot": 100.0,
  "strike": 100.0,
  "maturity": 1.0,
  "volatility": 0.20,
  "rate": 0.05,
  "dividend": 0.0,
  "option_type": "call",
  "num_simulations": 100000,
  "num_steps": 252,
  "variance_reduction": "antithetic",  // or "control_variates", "none"
  "random_seed": 42                    // optional, for reproducibility
}
```

**Response**:
```json
{
  "price": 10.4623,
  "standard_error": 0.0324,
  "confidence_interval_95": [10.3988, 10.5258],
  "greeks": {
    "delta": 0.5971,
    "gamma": 0.0194,
    "vega": 38.7621
  },
  "simulation_details": {
    "num_simulations": 100000,
    "num_steps": 252,
    "variance_reduction": "antithetic",
    "paths_generated": 200000,
    "convergence_rate": 0.0032
  },
  "metadata": {
    "method": "monte_carlo",
    "computation_time_ms": 1847
  }
}
```

### 2. Finite Difference Methods (American Options)

Price American options using PDE solvers.

**Endpoint**: `POST /api/v1/pricing/finite-difference`

**Request Body**:
```json
{
  "spot": 100.0,
  "strike": 100.0,
  "maturity": 1.0,
  "volatility": 0.20,
  "rate": 0.05,
  "dividend": 0.0,
  "option_type": "put",
  "exercise_type": "american",
  "method": "crank_nicolson",  // or "implicit_euler", "explicit_euler"
  "grid_points_spot": 100,
  "grid_points_time": 100,
  "spot_max_multiplier": 3.0
}
```

**Response**:
```json
{
  "price": 5.8764,
  "early_exercise_premium": 0.3214,
  "european_price": 5.5550,
  "optimal_exercise_boundary": [
    {"time": 0.0, "spot": 100.0},
    {"time": 0.25, "spot": 97.3},
    {"time": 0.50, "spot": 94.8},
    {"time": 0.75, "spot": 92.5},
    {"time": 1.0, "spot": 90.2}
  ],
  "greeks": {
    "delta": -0.4234,
    "gamma": 0.0201
  },
  "grid_details": {
    "spot_points": 100,
    "time_points": 100,
    "spot_min": 0.0,
    "spot_max": 300.0,
    "time_step": 0.01
  },
  "metadata": {
    "method": "crank_nicolson",
    "computation_time_ms": 8.67
  }
}
```

### 3. Binomial Tree (Lattice Model)

Price options using binomial tree method.

**Endpoint**: `POST /api/v1/pricing/binomial`

**Request Body**:
```json
{
  "spot": 100.0,
  "strike": 100.0,
  "maturity": 1.0,
  "volatility": 0.20,
  "rate": 0.05,
  "dividend": 0.0,
  "option_type": "put",
  "exercise_type": "american",
  "num_steps": 500,
  "tree_type": "crr"  // Cox-Ross-Rubinstein
}
```

**Response**:
```json
{
  "price": 5.8734,
  "greeks": {
    "delta": -0.4230,
    "gamma": 0.0199
  },
  "convergence_analysis": {
    "steps_50": 5.9123,
    "steps_100": 5.8891,
    "steps_200": 5.8782,
    "steps_500": 5.8734,
    "converged": true
  },
  "metadata": {
    "method": "binomial_tree",
    "tree_type": "crr",
    "num_steps": 500,
    "computation_time_ms": 12.34
  }
}
```

### 4. Exotic Options

Price exotic option types.

**Endpoint**: `POST /api/v1/pricing/exotic`

**Asian Option Example**:
```json
{
  "spot": 100.0,
  "strike": 100.0,
  "maturity": 1.0,
  "volatility": 0.20,
  "rate": 0.05,
  "dividend": 0.0,
  "option_type": "call",
  "exotic_type": "asian",
  "averaging_type": "arithmetic",  // or "geometric"
  "num_observations": 252,
  "method": "monte_carlo",
  "num_simulations": 100000
}
```

**Barrier Option Example**:
```json
{
  "spot": 100.0,
  "strike": 100.0,
  "maturity": 1.0,
  "volatility": 0.20,
  "rate": 0.05,
  "dividend": 0.0,
  "option_type": "call",
  "exotic_type": "barrier",
  "barrier_type": "down_and_out",
  "barrier_level": 95.0,
  "rebate": 5.0,
  "method": "monte_carlo",
  "num_simulations": 100000
}
```

**Response**:
```json
{
  "price": 6.7234,
  "standard_error": 0.0289,
  "confidence_interval_95": [6.6668, 6.7800],
  "exotic_details": {
    "type": "barrier",
    "barrier_type": "down_and_out",
    "barrier_level": 95.0,
    "knock_out_probability": 0.1234
  },
  "metadata": {
    "method": "monte_carlo",
    "computation_time_ms": 2134
  }
}
```

## Portfolio Management

### 1. Create Portfolio

**Endpoint**: `POST /api/v1/portfolio/create`

**Authorization**: Required (Bearer token)

**Request Body**:
```json
{
  "name": "Tech Options Portfolio",
  "description": "Long-term tech sector options",
  "currency": "USD"
}
```

**Response**:
```json
{
  "id": "pf_9876543210",
  "name": "Tech Options Portfolio",
  "description": "Long-term tech sector options",
  "currency": "USD",
  "created_at": "2025-12-13T10:30:00Z",
  "total_value": 0.0,
  "positions_count": 0
}
```

### 2. Add Position

**Endpoint**: `POST /api/v1/portfolio/{portfolio_id}/positions`

**Request Body**:
```json
{
  "symbol": "AAPL",
  "option_type": "call",
  "strike": 150.0,
  "expiration_date": "2026-01-16",
  "quantity": 10,
  "entry_price": 12.50,
  "entry_date": "2025-12-13"
}
```

**Response**:
```json
{
  "position_id": "pos_1234567890",
  "symbol": "AAPL",
  "option_type": "call",
  "strike": 150.0,
  "expiration_date": "2026-01-16",
  "quantity": 10,
  "entry_price": 12.50,
  "current_price": 13.25,
  "unrealized_pnl": 75.00,
  "pnl_percentage": 6.0,
  "greeks": {
    "delta": 0.6234,
    "gamma": 0.0187,
    "theta": -0.0234,
    "vega": 42.5,
    "rho": 38.2
  },
  "days_to_expiration": 399
}
```

### 3. Get Portfolio Summary

**Endpoint**: `GET /api/v1/portfolio/{portfolio_id}`

**Response**:
```json
{
  "id": "pf_9876543210",
  "name": "Tech Options Portfolio",
  "total_value": 13250.00,
  "total_cost_basis": 12500.00,
  "unrealized_pnl": 750.00,
  "realized_pnl": 0.0,
  "pnl_percentage": 6.0,
  "positions": [
    {
      "position_id": "pos_1234567890",
      "symbol": "AAPL",
      "option_type": "call",
      "quantity": 10,
      "current_value": 13250.00,
      "pnl": 750.00
    }
  ],
  "portfolio_greeks": {
    "delta": 623.4,
    "gamma": 187.0,
    "theta": -234.0,
    "vega": 4250.0,
    "rho": 3820.0
  },
  "risk_metrics": {
    "max_loss": 12500.00,
    "probability_of_profit": 0.6234,
    "expected_return": 0.08,
    "sharpe_ratio": 1.23
  }
}
```

### 4. Calculate Portfolio Greeks

**Endpoint**: `GET /api/v1/portfolio/{portfolio_id}/greeks`

**Response**:
```json
{
  "portfolio_id": "pf_9876543210",
  "timestamp": "2025-12-13T10:30:00Z",
  "total_delta": 623.4,
  "total_gamma": 187.0,
  "total_vega": 4250.0,
  "total_theta": -234.0,
  "total_rho": 3820.0,
  "delta_exposure": {
    "long": 623.4,
    "short": 0.0,
    "net": 623.4
  },
  "position_breakdown": [
    {
      "symbol": "AAPL",
      "delta": 623.4,
      "gamma": 187.0,
      "contribution_to_portfolio_delta": 1.0
    }
  ]
}
```

## Machine Learning API

### 1. List Available Models

**Endpoint**: `GET /api/v1/ml/models`

**Response**:
```json
{
  "models": [
    {
      "id": "model_xgb_v1_2",
      "name": "XGBoost Price Predictor",
      "version": "1.2",
      "algorithm": "xgboost",
      "created_at": "2025-12-01T10:00:00Z",
      "metrics": {
        "rmse": 0.0234,
        "mae": 0.0187,
        "r2": 0.9876
      },
      "status": "production"
    },
    {
      "id": "model_lstm_v2_0",
      "name": "LSTM Volatility Forecaster",
      "version": "2.0",
      "algorithm": "lstm",
      "created_at": "2025-12-05T14:30:00Z",
      "metrics": {
        "rmse": 0.0198,
        "mae": 0.0156
      },
      "status": "production"
    }
  ]
}
```

### 2. Predict Option Price

**Endpoint**: `POST /api/v1/ml/predict`

**Request Body**:
```json
{
  "model_id": "model_xgb_v1_2",
  "features": {
    "spot": 100.0,
    "strike": 100.0,
    "maturity": 1.0,
    "volatility": 0.20,
    "rate": 0.05,
    "dividend": 0.0,
    "option_type": "call",
    "historical_volatility_30d": 0.22,
    "historical_volatility_60d": 0.21,
    "volume": 1000000,
    "open_interest": 50000
  }
}
```

**Response**:
```json
{
  "predicted_price": 10.4823,
  "confidence_interval_95": [10.3456, 10.6190],
  "feature_importance": {
    "volatility": 0.4523,
    "maturity": 0.2134,
    "moneyness": 0.1876,
    "rate": 0.0987,
    "others": 0.0480
  },
  "model_metadata": {
    "model_id": "model_xgb_v1_2",
    "version": "1.2",
    "trained_on": "2025-12-01T10:00:00Z",
    "training_data_size": 1000000
  },
  "computation_time_ms": 23.45
}
```

### 3. Train New Model

**Endpoint**: `POST /api/v1/ml/train`

**Request Body**:
```json
{
  "model_name": "Custom XGBoost",
  "algorithm": "xgboost",
  "training_data_path": "s3://bucket/training_data.csv",
  "hyperparameters": {
    "n_estimators": 1000,
    "max_depth": 8,
    "learning_rate": 0.01,
    "subsample": 0.8
  },
  "validation_split": 0.2,
  "random_seed": 42
}
```

**Response**:
```json
{
  "task_id": "task_training_123",
  "status": "queued",
  "estimated_time_minutes": 15,
  "status_url": "/api/v1/ml/train/task_training_123/status"
}
```

**Check Training Status**:
```bash
GET /api/v1/ml/train/task_training_123/status
```

**Response**:
```json
{
  "task_id": "task_training_123",
  "status": "completed",
  "progress_percentage": 100,
  "model_id": "model_xgb_custom_1_0",
  "metrics": {
    "rmse": 0.0245,
    "mae": 0.0192,
    "r2": 0.9854
  },
  "training_time_seconds": 876,
  "artifact_path": "mlruns/1/abc123def456/artifacts"
}
```

## WebSocket Real-Time Data

### Connect to WebSocket

**Endpoint**: `ws://localhost:8000/ws`

**Connection Example (JavaScript)**:
```javascript
const socket = new WebSocket('ws://localhost:8000/ws');

socket.onopen = () => {
  console.log('Connected to WebSocket');

  // Subscribe to market data
  socket.send(JSON.stringify({
    action: 'subscribe',
    channel: 'market_data',
    symbols: ['AAPL', 'MSFT', 'GOOGL']
  }));
};

socket.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Received:', data);
};

socket.onerror = (error) => {
  console.error('WebSocket error:', error);
};

socket.onclose = () => {
  console.log('Disconnected from WebSocket');
};
```

### Subscribe to Market Data

**Message**:
```json
{
  "action": "subscribe",
  "channel": "market_data",
  "symbols": ["AAPL", "MSFT"]
}
```

**Incoming Messages**:
```json
{
  "type": "market_data",
  "symbol": "AAPL",
  "timestamp": "2025-12-13T10:30:15.123Z",
  "bid": 149.85,
  "ask": 149.90,
  "last": 149.87,
  "volume": 1234567,
  "implied_volatility": 0.2134
}
```

### Subscribe to Portfolio Updates

**Message**:
```json
{
  "action": "subscribe",
  "channel": "portfolio",
  "portfolio_id": "pf_9876543210",
  "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
}
```

**Incoming Messages**:
```json
{
  "type": "portfolio_update",
  "portfolio_id": "pf_9876543210",
  "timestamp": "2025-12-13T10:30:15.123Z",
  "total_value": 13450.00,
  "unrealized_pnl": 950.00,
  "pnl_change": 200.00,
  "positions_updated": [
    {
      "position_id": "pos_1234567890",
      "current_price": 13.45,
      "pnl": 95.00
    }
  ]
}
```

## Error Handling

### Error Response Format

All errors follow a consistent format:

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid input parameters",
    "details": {
      "field": "volatility",
      "issue": "must be greater than 0",
      "received": -0.2
    },
    "timestamp": "2025-12-13T10:30:00Z",
    "request_id": "req_abc123def456"
  }
}
```

### Common Error Codes

| HTTP Status | Error Code | Description |
|------------|------------|-------------|
| 400 | `VALIDATION_ERROR` | Invalid request parameters |
| 401 | `UNAUTHORIZED` | Missing or invalid authentication token |
| 403 | `FORBIDDEN` | Insufficient permissions for operation |
| 404 | `NOT_FOUND` | Resource not found |
| 409 | `CONFLICT` | Resource already exists |
| 422 | `UNPROCESSABLE_ENTITY` | Request cannot be processed |
| 429 | `RATE_LIMIT_EXCEEDED` | Too many requests |
| 500 | `INTERNAL_SERVER_ERROR` | Server error |
| 503 | `SERVICE_UNAVAILABLE` | Service temporarily unavailable |

### Error Handling Examples

**Python**:
```python
import requests

try:
    response = requests.post(url, json=payload)
    response.raise_for_status()
    result = response.json()
except requests.exceptions.HTTPError as e:
    error_data = e.response.json()
    print(f"Error {error_data['error']['code']}: {error_data['error']['message']}")
except requests.exceptions.RequestException as e:
    print(f"Request failed: {e}")
```

**JavaScript**:
```javascript
try {
  const response = await axios.post(url, payload);
  const result = response.data;
} catch (error) {
  if (error.response) {
    const { code, message, details } = error.response.data.error;
    console.error(`Error ${code}: ${message}`, details);
  } else {
    console.error('Request failed:', error.message);
  }
}
```

## Rate Limiting

### Rate Limit Headers

Every response includes rate limit information:

```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 987
X-RateLimit-Reset: 1702468200
```

### Rate Limits by Endpoint

| Endpoint | Limit | Window |
|----------|-------|--------|
| `/api/v1/auth/login` | 5 | 15 minutes |
| `/api/v1/pricing/black-scholes` | 1,000 | 1 hour |
| `/api/v1/pricing/monte-carlo` | 100 | 1 hour |
| `/api/v1/portfolio/*` | 10,000 | 1 hour |
| `/api/v1/ml/*` | 500 | 1 hour |

### Rate Limit Exceeded Response

```json
{
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "Rate limit exceeded",
    "details": {
      "limit": 1000,
      "window_seconds": 3600,
      "retry_after_seconds": 1234
    },
    "timestamp": "2025-12-13T10:30:00Z"
  }
}
```

## Best Practices

### 1. Use Batch Endpoints

Instead of multiple individual requests:
```python
# BAD: Multiple requests
for strike in [95, 100, 105]:
    response = requests.post(url, json={"strike": strike, ...})

# GOOD: Single batch request
response = requests.post(batch_url, json={"options": [
    {"strike": 95, ...},
    {"strike": 100, ...},
    {"strike": 105, ...}
]})
```

### 2. Cache Results

Implement client-side caching for unchanged parameters:
```python
import hashlib
import json
from functools import lru_cache

@lru_cache(maxsize=1000)
def price_option(params_tuple):
    params = dict(params_tuple)
    response = requests.post(url, json=params)
    return response.json()

# Convert dict to hashable tuple
params_tuple = tuple(sorted(params.items()))
result = price_option(params_tuple)
```

### 3. Handle Errors Gracefully

Implement exponential backoff for retries:
```python
import time

def api_call_with_retry(url, payload, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:  # Rate limit
                wait_time = 2 ** attempt
                time.sleep(wait_time)
            else:
                raise
        except requests.exceptions.Timeout:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)
    raise Exception("Max retries exceeded")
```

### 4. Use Async for Multiple Requests

For multiple independent requests, use async:
```python
import asyncio
import httpx

async def price_option(client, params):
    response = await client.post(url, json=params)
    return response.json()

async def price_multiple_options(options_list):
    async with httpx.AsyncClient() as client:
        tasks = [price_option(client, params) for params in options_list]
        results = await asyncio.gather(*tasks)
    return results

# Usage
results = asyncio.run(price_multiple_options(options_list))
```

### 5. Validate Inputs Before API Call

Reduce API errors by validating inputs:
```python
def validate_pricing_params(params):
    assert params['spot'] > 0, "Spot must be positive"
    assert params['strike'] > 0, "Strike must be positive"
    assert params['maturity'] > 0, "Maturity must be positive"
    assert params['volatility'] > 0, "Volatility must be positive"
    assert params['option_type'] in ['call', 'put'], "Invalid option type"

validate_pricing_params(params)
response = requests.post(url, json=params)
```

---

**API Version**: v1
**Last Updated**: 2025-12-13
**Support**: api-support@bsopt.com
**Documentation**: https://docs.bsopt.com
