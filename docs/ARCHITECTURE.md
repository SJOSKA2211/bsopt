# System Architecture

## Table of Contents

1. [Overview](#overview)
2. [System Components](#system-components)
3. [Architecture Diagrams](#architecture-diagrams)
4. [Data Flow](#data-flow)
5. [Technology Stack](#technology-stack)
6. [Design Patterns](#design-patterns)
7. [Scalability Considerations](#scalability-considerations)
8. [Security Architecture](#security-architecture)

## Overview

The Black-Scholes Advanced Option Pricing Platform is a comprehensive quantitative finance system built on a microservices-inspired architecture. The platform combines analytical pricing engines, numerical solvers, machine learning models, and real-time trading capabilities into a unified system.

### Architectural Principles

1. **Separation of Concerns**: Clear boundaries between pricing logic, API layer, data persistence, and frontend
2. **Stateless Services**: API servers maintain no session state, enabling horizontal scaling
3. **Event-Driven Processing**: Asynchronous task processing for computationally intensive operations
4. **Data Locality**: Time-series data optimized for fast query performance
5. **Fail-Fast Validation**: Input validation at API boundaries before computation

### Design Goals

- **Performance**: Sub-100ms response time for 95th percentile of pricing requests
- **Accuracy**: Numerical precision within 0.01% of theoretical values
- **Scalability**: Handle 10,000+ concurrent pricing calculations
- **Reliability**: 99.9% uptime with automatic failover
- **Maintainability**: Modular codebase with comprehensive test coverage (>90%)

## System Components

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Client Applications                          │
│  (Web Browser, CLI, Mobile Apps, Third-Party Integrations)          │
└────────────┬────────────────────────────────────────────────────────┘
             │
             │ HTTPS/WSS
             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         Load Balancer (Nginx)                        │
│             (SSL Termination, Rate Limiting, Compression)            │
└────────┬─────────────────────────────────────────────┬──────────────┘
         │                                              │
         │ HTTP                                         │ HTTP
         ▼                                              ▼
┌──────────────────────┐                    ┌────────────────────────┐
│   React Frontend     │                    │   FastAPI Backend      │
│   (Port 3000)        │                    │   (Port 8000)          │
│                      │                    │                        │
│ - Dashboard UI       │                    │ - REST API             │
│ - Pricing Calculator │                    │ - GraphQL API          │
│ - Charts/Viz         │                    │ - WebSocket Server     │
│ - Portfolio Mgmt     │                    │ - Business Logic       │
└──────────────────────┘                    └───────┬────────────────┘
                                                    │
                               ┌────────────────────┼─────────────────┐
                               │                    │                 │
                               ▼                    ▼                 ▼
                    ┌──────────────────┐  ┌──────────────┐  ┌────────────────┐
                    │  PostgreSQL +    │  │    Redis     │  │   RabbitMQ     │
                    │  TimescaleDB     │  │   (Cache)    │  │  (Message Bus) │
                    │                  │  │              │  │                │
                    │ - User Data      │  │ - Sessions   │  │ - Task Queue   │
                    │ - Options        │  │ - API Cache  │  │ - Events       │
                    │ - Portfolios     │  │ - Rate Limit │  │                │
                    │ - Time-Series    │  │              │  │                │
                    └──────────────────┘  └──────────────┘  └────────┬───────┘
                                                                     │
                                                                     ▼
                                                          ┌────────────────────┐
                                                          │  Celery Workers    │
                                                          │                    │
                                                          │ - Batch Pricing    │
                                                          │ - ML Training      │
                                                          │ - Backtesting      │
                                                          │ - Data Pipeline    │
                                                          └────────────────────┘
```

### Component Descriptions

#### 1. Frontend Layer (React SPA)

**Purpose**: User-facing web application for interactive option pricing and portfolio management

**Key Features**:
- Real-time pricing calculator with multiple pricing methods
- Interactive Greeks visualization (delta, gamma, vega, theta, rho)
- 3D volatility surface rendering
- Portfolio dashboard with P&L tracking
- Live market data display with WebSocket updates
- Responsive design for desktop and mobile

**Technology Stack**:
- React 18 with TypeScript
- Redux Toolkit for state management
- Recharts/Victory for data visualization
- Material-UI for component library
- WebSocket client for real-time updates

**File Location**: `/frontend/`

#### 2. API Layer (FastAPI)

**Purpose**: High-performance REST and GraphQL API server

**Key Features**:
- Async request handling with `asyncio`
- Automatic OpenAPI documentation generation
- Request validation with Pydantic models
- JWT-based authentication
- Rate limiting and caching
- WebSocket support for real-time data

**Endpoints**:
- `/api/v1/pricing/*` - Option pricing endpoints
- `/api/v1/portfolio/*` - Portfolio management
- `/api/v1/ml/*` - Machine learning model operations
- `/api/v1/market-data/*` - Real-time market data
- `/graphql` - GraphQL query endpoint
- `/ws/*` - WebSocket connections

**File Location**: `/src/api/`

**Architecture Pattern**: Layered architecture with clear separation:
```
routes/ → schemas/ → services/ → pricing/ → database/
(HTTP)    (Valid)    (Logic)     (Calc)     (Persist)
```

#### 3. Pricing Engines

**Purpose**: Core mathematical and numerical computation modules

**Modules**:

**a) Black-Scholes Analytical Engine** (`/src/pricing/black_scholes.py`)
- Closed-form European option pricing
- Full Greeks calculation (delta, gamma, vega, theta, rho)
- Put-call parity verification
- Supports dividend yield adjustments
- Performance: 1M+ calculations/second (single core)

**b) Finite Difference Methods** (`/src/pricing/finite_difference.py`)
- Crank-Nicolson scheme for American options
- Explicit and Implicit Euler methods
- Adaptive grid spacing
- Early exercise boundary detection
- Performance: 100x100 grid in <10ms

**c) Monte Carlo Simulator** (`/src/pricing/monte_carlo.py`)
- Geometric Brownian Motion path generation
- Variance reduction techniques:
  - Antithetic variates
  - Control variates
  - Importance sampling
- Exotic option support
- Parallel execution with NumPy vectorization
- Performance: 100K paths in <2 seconds

**d) Lattice Models** (`/src/pricing/lattice.py`)
- Binomial tree (Cox-Ross-Rubinstein)
- Trinomial tree
- Adaptive time-stepping
- American option support
- Convergence analysis

**e) Exotic Options** (`/src/pricing/exotic.py`)
- Asian options (arithmetic and geometric averaging)
- Barrier options (knock-in, knock-out)
- Lookback options
- Digital/Binary options
- Quanto options

**f) Implied Volatility** (`/src/pricing/implied_vol.py`)
- Newton-Raphson method with vega
- Brent's method (root-finding)
- Bounded optimization
- Initial guess heuristics

**g) Volatility Surface** (`/src/pricing/vol_surface.py`)
- SVI (Stochastic Volatility Inspired) parameterization
- No-arbitrage constraints
- Calendar and butterfly arbitrage checks
- Surface interpolation

#### 4. Data Layer

**a) PostgreSQL + TimescaleDB**

**Purpose**: Primary data store with time-series optimization

**Schema**:
```sql
-- Users and authentication
users (id, email, password_hash, created_at, is_active)

-- Options data
options (
    id,
    underlying_symbol,
    strike,
    expiration_date,
    option_type,
    market_price,
    implied_volatility,
    timestamp
)

-- Portfolio positions
portfolios (id, user_id, name, created_at)
positions (
    id,
    portfolio_id,
    option_id,
    quantity,
    entry_price,
    current_price,
    pnl
)

-- Time-series market data (TimescaleDB hypertable)
market_data (
    timestamp,
    symbol,
    bid,
    ask,
    last,
    volume,
    open_interest
)

-- ML model metadata
ml_models (
    id,
    model_name,
    version,
    algorithm,
    hyperparameters,
    metrics,
    artifact_path,
    created_at
)
```

**Optimizations**:
- TimescaleDB for automatic partitioning of time-series data
- B-tree indexes on `(symbol, timestamp)`
- Composite indexes on `(user_id, portfolio_id)`
- Materialized views for common aggregations

**File Location**: `/src/database/models.py`

**b) Redis Cache**

**Purpose**: In-memory cache and session store

**Use Cases**:
- API response caching (5-minute TTL for pricing results)
- Session management (JWT token blacklist)
- Rate limiting counters (sliding window)
- Real-time market data snapshots
- Distributed locks for background jobs

**Data Structures**:
- Strings: Cached pricing results
- Hashes: User sessions
- Sorted Sets: Rate limit tracking
- Pub/Sub: Real-time event broadcasting

#### 5. Task Queue (Celery + RabbitMQ)

**Purpose**: Asynchronous task processing for long-running operations

**Task Types**:

**a) Batch Pricing**
- Input: CSV file with option parameters
- Processing: Parallel pricing using multiple workers
- Output: Results CSV with prices and Greeks
- Typical load: 10,000 options in <30 seconds

**b) ML Model Training**
- Input: Historical market data, features configuration
- Processing: Hyperparameter tuning with Optuna
- Output: Trained model artifact, metrics report
- Tracked with MLflow for versioning

**c) Backtesting**
- Input: Strategy configuration, date range
- Processing: Historical simulation with tick data
- Output: Performance metrics (Sharpe, max drawdown, etc.)

**d) Market Data Ingestion**
- Scheduled: Every 1 minute during market hours
- Fetches: Real-time quotes from broker APIs
- Stores: Time-series data in TimescaleDB

**Configuration**:
```python
# Celery configuration
broker_url = "amqp://rabbitmq:5672"
result_backend = "redis://redis:6379/1"
task_serializer = "json"
result_serializer = "json"
timezone = "UTC"
enable_utc = True
```

#### 6. Machine Learning Pipeline

**Purpose**: Predictive modeling for option pricing and volatility forecasting

**Components**:

**a) Feature Engineering**
- Technical indicators (RSI, MACD, Bollinger Bands)
- Greeks as features
- Market regime detection
- Volatility clustering metrics

**b) Model Registry (MLflow)**
- Model versioning and lineage
- Experiment tracking
- A/B testing support
- Model deployment and serving

**c) Supported Algorithms**
- XGBoost: Gradient boosting for price prediction
- LightGBM: Fast tree-based learning
- Random Forest: Ensemble methods
- Neural Networks: Deep learning with PyTorch
- LSTM: Sequential modeling for time-series

**d) Model Serving**
- REST API endpoint: `/api/v1/ml/predict`
- Input: Market features
- Output: Predicted price, confidence interval
- Latency: <50ms per prediction

## Architecture Diagrams

### Request Flow Diagram

```
┌──────────┐
│  Client  │
└────┬─────┘
     │ 1. HTTP POST /api/v1/pricing/black-scholes
     ▼
┌─────────────┐
│    Nginx    │ 2. SSL termination, rate limit check
└────┬────────┘
     │
     ▼
┌──────────────┐
│   FastAPI    │ 3. Route to pricing endpoint
└────┬─────────┘
     │
     ▼
┌──────────────┐
│ Pydantic     │ 4. Validate request body
│ Schema       │    (spot, strike, maturity, volatility, rate)
└────┬─────────┘
     │
     ▼
┌──────────────┐
│ Redis Cache  │ 5. Check cache (key: hash(params))
└────┬─────────┘
     │ Cache MISS
     ▼
┌──────────────┐
│ Black-Scholes│ 6. Calculate option price and Greeks
│ Engine       │    - Compute d1, d2
└────┬─────────┘    - Price call/put
     │              - Calculate sensitivities
     ▼
┌──────────────┐
│ Redis Cache  │ 7. Store result (TTL: 5 minutes)
└────┬─────────┘
     │
     ▼
┌──────────────┐
│ FastAPI      │ 8. Return JSON response
└────┬─────────┘
     │
     ▼
┌──────────┐
│  Client  │ 9. Display result
└──────────┘

Total latency: ~20-50ms (p95)
```

### Database Schema Diagram

```
┌─────────────────┐
│     users       │
├─────────────────┤
│ id (PK)         │
│ email           │
│ password_hash   │
│ created_at      │
└────────┬────────┘
         │ 1:N
         ▼
┌─────────────────┐
│  portfolios     │
├─────────────────┤
│ id (PK)         │
│ user_id (FK)    │
│ name            │
│ created_at      │
└────────┬────────┘
         │ 1:N
         ▼
┌─────────────────┐        ┌─────────────────┐
│   positions     │───────>│    options      │
├─────────────────┤  N:1   ├─────────────────┤
│ id (PK)         │        │ id (PK)         │
│ portfolio_id(FK)│        │ symbol          │
│ option_id (FK)  │        │ strike          │
│ quantity        │        │ expiration_date │
│ entry_price     │        │ option_type     │
│ current_pnl     │        │ market_price    │
└─────────────────┘        │ implied_vol     │
                           │ timestamp       │
                           └─────────────────┘
                                   │
                                   │ 1:N
                                   ▼
                           ┌─────────────────┐
                           │  market_data    │
                           │ (TimescaleDB)   │
                           ├─────────────────┤
                           │ timestamp (PK)  │
                           │ symbol (PK)     │
                           │ bid             │
                           │ ask             │
                           │ last            │
                           │ volume          │
                           └─────────────────┘
```

### Deployment Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Cloud Provider (AWS/GCP)                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌────────────────────────────────────────────────────────┐    │
│  │              Application Load Balancer                  │    │
│  │         (Auto-scaling, Health Checks, SSL)              │    │
│  └──────────┬──────────────────────────────┬──────────────┘    │
│             │                               │                    │
│  ┌──────────▼──────────┐       ┌───────────▼──────────┐        │
│  │  Frontend Servers   │       │   API Servers        │        │
│  │  (Nginx + React)    │       │   (FastAPI)          │        │
│  │  EC2 Auto-scaling   │       │   EC2 Auto-scaling   │        │
│  │  Min: 2, Max: 10    │       │   Min: 3, Max: 20    │        │
│  └─────────────────────┘       └──────────┬───────────┘        │
│                                            │                     │
│                        ┌───────────────────┼─────────────┐      │
│                        │                   │             │      │
│             ┌──────────▼─────┐  ┌─────────▼──────┐  ┌──▼─────┐│
│             │  RDS PostgreSQL│  │  ElastiCache   │  │  SQS   ││
│             │  Multi-AZ      │  │  Redis Cluster │  │ (Queue)││
│             │  Read Replicas │  │                │  │        ││
│             └────────────────┘  └────────────────┘  └────┬───┘│
│                                                           │     │
│                                              ┌────────────▼───┐│
│                                              │ Celery Workers ││
│                                              │ EC2 Spot Fleet ││
│                                              └────────────────┘│
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │              Monitoring & Logging                         │ │
│  │  - CloudWatch Metrics, Logs                              │ │
│  │  - Prometheus + Grafana                                  │ │
│  │  - Sentry (Error Tracking)                               │ │
│  └──────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## Data Flow

### Synchronous Pricing Request Flow

1. **Client Request**: User submits pricing parameters via web form or API call
2. **Load Balancer**: Nginx routes to available API server instance
3. **Authentication**: JWT token validated, user permissions checked
4. **Rate Limiting**: Redis checks request count for user (max: 1000/hour)
5. **Validation**: Pydantic schema validates input parameters
6. **Cache Lookup**: Check Redis for cached result using parameter hash
7. **Computation**: If cache miss, execute pricing engine calculation
8. **Cache Store**: Store result in Redis with 5-minute TTL
9. **Response**: Return JSON with price, Greeks, and metadata
10. **Logging**: Record request metrics (latency, method, user)

### Asynchronous Batch Processing Flow

1. **Upload**: User uploads CSV file with option parameters
2. **Validation**: File format and content validated
3. **Task Creation**: Celery task created with file reference
4. **Queue**: Task published to RabbitMQ queue
5. **Worker Pick-up**: Available Celery worker claims task
6. **Batch Processing**: Worker processes rows in parallel
7. **Progress Updates**: WebSocket updates sent to client
8. **Result Storage**: Output CSV stored in S3/object storage
9. **Notification**: User notified via email/WebSocket
10. **Cleanup**: Temporary files removed after 24 hours

### Real-Time Market Data Flow

1. **Broker WebSocket**: Connect to market data provider (IB, Alpaca)
2. **Message Handler**: Parse incoming tick data
3. **Validation**: Validate data quality (bid < ask, timestamp check)
4. **Database Insert**: Bulk insert to TimescaleDB hypertable
5. **Redis Update**: Update latest price snapshot in Redis
6. **Pub/Sub Broadcast**: Publish to Redis channel
7. **WebSocket Fanout**: Send updates to connected frontend clients
8. **Aggregation**: 1-minute OHLCV bars computed every minute

## Technology Stack

### Backend

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| API Framework | FastAPI | 0.104+ | Async REST API server |
| Language | Python | 3.11+ | Primary development language |
| Validation | Pydantic | 2.5+ | Data validation and serialization |
| Database | PostgreSQL | 15+ | Relational data storage |
| Time-Series | TimescaleDB | 2.13+ | High-performance time-series |
| Cache | Redis | 7.2+ | In-memory cache and session store |
| Task Queue | Celery | 5.3+ | Distributed task processing |
| Message Broker | RabbitMQ | 3.12+ | Message queue for Celery |
| Numerical | NumPy | 1.26+ | Array operations |
| Scientific | SciPy | 1.11+ | Statistical functions |
| ML | Scikit-learn | 1.3+ | Classical machine learning |
| Gradient Boosting | XGBoost | 2.0+ | Tree-based models |
| Deep Learning | PyTorch | 2.1+ | Neural networks |
| ML Ops | MLflow | 2.9+ | Model registry and tracking |

### Frontend

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| Framework | React | 18+ | UI library |
| Language | TypeScript | 5.0+ | Type-safe JavaScript |
| State Management | Redux Toolkit | 2.0+ | Global state |
| UI Components | Material-UI | 5.14+ | Component library |
| Charts | Recharts | 2.10+ | Data visualization |
| Build Tool | Vite | 5.0+ | Fast bundler |

### Infrastructure

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Containerization | Docker | Application packaging |
| Orchestration | Docker Compose / Kubernetes | Multi-container deployment |
| Reverse Proxy | Nginx | Load balancing, SSL termination |
| CI/CD | GitHub Actions | Automated testing and deployment |
| Monitoring | Prometheus + Grafana | Metrics and dashboards |
| Logging | ELK Stack | Centralized logging |
| Error Tracking | Sentry | Exception monitoring |

## Design Patterns

### 1. Repository Pattern

**Purpose**: Abstraction layer between business logic and data access

**Implementation**:
```python
# src/database/crud.py
class OptionRepository:
    def __init__(self, db: Session):
        self.db = db

    def get_option_by_id(self, option_id: int) -> Option:
        return self.db.query(Option).filter(Option.id == option_id).first()

    def create_option(self, option_data: OptionCreate) -> Option:
        option = Option(**option_data.dict())
        self.db.add(option)
        self.db.commit()
        return option
```

**Benefits**:
- Testable business logic (mock repository in tests)
- Database-agnostic service layer
- Centralized query logic

### 2. Service Layer Pattern

**Purpose**: Encapsulate business logic separate from API routes

**Implementation**:
```python
# src/services/pricing_service.py
class PricingService:
    def __init__(self, cache: Redis, db: Session):
        self.cache = cache
        self.db = db

    async def price_option(self, params: PricingRequest) -> PricingResult:
        # Check cache
        cached = await self.cache.get(params.cache_key())
        if cached:
            return cached

        # Calculate price
        price = BlackScholesEngine.price_call(params.to_bs_params())

        # Store in cache
        await self.cache.setex(params.cache_key(), 300, price)

        return PricingResult(price=price)
```

### 3. Factory Pattern

**Purpose**: Create pricing engine instances based on method type

**Implementation**:
```python
class PricingEngineFactory:
    @staticmethod
    def create_engine(method: str) -> PricingEngine:
        if method == "black_scholes":
            return BlackScholesEngine()
        elif method == "monte_carlo":
            return MonteCarloEngine()
        elif method == "finite_difference":
            return FiniteDifferenceEngine()
        else:
            raise ValueError(f"Unknown method: {method}")
```

### 4. Strategy Pattern

**Purpose**: Interchangeable algorithms for implied volatility calculation

**Implementation**:
```python
class ImpliedVolStrategy(ABC):
    @abstractmethod
    def calculate(self, market_price: float, params: BSParameters) -> float:
        pass

class NewtonRaphsonStrategy(ImpliedVolStrategy):
    def calculate(self, market_price: float, params: BSParameters) -> float:
        # Newton-Raphson implementation
        pass

class BrentStrategy(ImpliedVolStrategy):
    def calculate(self, market_price: float, params: BSParameters) -> float:
        # Brent's method implementation
        pass
```

### 5. Decorator Pattern

**Purpose**: Add cross-cutting concerns (caching, logging, timing)

**Implementation**:
```python
def cached(ttl: int = 300):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            cache_key = f"{func.__name__}:{hash(frozenset(kwargs.items()))}"
            cached_result = await redis.get(cache_key)
            if cached_result:
                return cached_result
            result = await func(*args, **kwargs)
            await redis.setex(cache_key, ttl, result)
            return result
        return wrapper
    return decorator

@cached(ttl=300)
async def price_option(params: PricingRequest):
    # Pricing logic
    pass
```

## Scalability Considerations

### Horizontal Scaling

**API Servers**:
- Stateless design enables unlimited horizontal scaling
- Auto-scaling based on CPU utilization (target: 70%)
- Health check endpoint: `/api/health`
- Graceful shutdown on SIGTERM (drain connections)

**Celery Workers**:
- Worker pool can scale independently from API servers
- Spot instances for cost optimization (batch jobs tolerant of interruption)
- Task routing to specialized workers (CPU-intensive vs I/O-intensive)

### Vertical Scaling

**Database**:
- PostgreSQL: Scale up to 96 vCPUs, 768 GB RAM
- Read replicas for read-heavy workloads
- Connection pooling with PgBouncer (max 100 connections per instance)

**Redis**:
- Cluster mode for >100 GB datasets
- Read replicas for cache reads
- Eviction policy: `allkeys-lru` for cache, `noeviction` for sessions

### Performance Optimizations

**Database**:
- Partitioning time-series tables by month
- Materialized views for common aggregations
- Index-only scans for hot queries
- `EXPLAIN ANALYZE` for query optimization

**Application**:
- NumPy vectorization for batch calculations
- Numba JIT compilation for hot loops
- Redis pipelining for bulk operations
- Connection pooling (database, cache, message queue)

**Caching Strategy**:
- L1 Cache: Application-level (in-memory dict with TTL)
- L2 Cache: Redis (distributed cache)
- Cache warming for predictable queries
- Cache invalidation on data updates

### Load Testing Results

```
Scenario: Black-Scholes pricing endpoint
Tool: Locust
Users: 1000 concurrent
Duration: 5 minutes

Results:
- Requests per second: 2,450
- Average response time: 38ms
- 95th percentile: 87ms
- 99th percentile: 142ms
- Failure rate: 0.02%

Bottleneck: Database connection pool (resolved by increasing pool size)
```

## Security Architecture

### Authentication

**JWT-based authentication**:
- Access token: 15-minute expiration
- Refresh token: 7-day expiration, stored in HTTP-only cookie
- Token blacklist in Redis for logout functionality

**Password security**:
- Bcrypt hashing with salt (cost factor: 12)
- Minimum password requirements: 8 chars, uppercase, lowercase, number
- Rate limiting on login endpoint (5 attempts per 15 minutes)

### Authorization

**Role-Based Access Control (RBAC)**:
```python
Roles:
- admin: Full system access
- trader: Trading and portfolio management
- analyst: Read-only access to pricing and analytics
- api_user: Programmatic API access
```

**Endpoint-level permissions**:
```python
@router.post("/portfolio/create", dependencies=[Depends(require_role("trader"))])
async def create_portfolio(...):
    pass
```

### API Security

**Rate Limiting**:
- Global: 10,000 requests/hour per IP
- Authenticated: 100,000 requests/hour per user
- Pricing endpoint: 1,000 requests/hour per user
- Implemented with Redis sliding window

**Input Validation**:
- Pydantic schemas for all request bodies
- SQL injection prevention (parameterized queries)
- XSS prevention (HTML escaping)
- CORS configuration (whitelist origins)

**Data Encryption**:
- TLS 1.3 for all external communication
- Encrypted database connections
- Secrets stored in environment variables (AWS Secrets Manager in production)
- PII data encrypted at rest (PostgreSQL pgcrypto)

### Infrastructure Security

**Network Security**:
- VPC with private subnets for databases
- Security groups limiting ingress to load balancer
- No direct internet access for backend services

**Monitoring**:
- Failed authentication attempts logged
- Anomaly detection for unusual API usage patterns
- Automated alerts for security events

**Compliance**:
- GDPR compliance for EU users
- SOC 2 Type II controls
- Regular security audits and penetration testing

---

**Document Version**: 2.2.0
**Last Updated**: 2025-12-28
**Maintainer**: Technical Architecture Team
**Review Cycle**: Quarterly
