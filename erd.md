```mermaid
erDiagram
    USERS ||--o{ PORTFOLIOS : has
    USERS ||--o{ OAUTH2_CLIENTS : has
    USERS ||--o{ API_KEYS : has
    USERS ||--o{ OAUTH2_AUTHORIZATION_CODES : has
    USERS ||--o{ OAUTH2_TOKENS : has
    USERS ||--o{ ORDERS : places
    USERS ||--o{ RATE_LIMITS : has
    ML_MODELS ||--o{ MODEL_PREDICTIONS : generates

    USERS {
        UUID id PK
        VARCHAR email UK
        VARCHAR hashed_password
        VARCHAR full_name
        VARCHAR tier
        DATETIME created_at
        DATETIME last_login
        BOOLEAN is_active
        BOOLEAN is_verified
        VARCHAR mfa_secret
        VARCHAR verification_token
    }

    API_KEYS {
        UUID id PK
        UUID user_id FK
        VARCHAR name
        VARCHAR key_hash UK
        VARCHAR key_prefix
        DATETIME created_at
        DATETIME expires_at
        DATETIME last_used_at
        BOOLEAN is_active
    }

    AUDIT_LOGS {
        DATETIME time PK
        TEXT method
        TEXT path
        INTEGER status_code
        TEXT user_id
        TEXT client_ip
        TEXT user_agent
        NUMERIC latency_ms
        JSONB metadata_json
    }
    
    REQUEST_LOGS {
        UUID id PK
        DATETIME time
        VARCHAR path
        VARCHAR method
        INTEGER status_code
        NUMERIC latency_ms
    }

    SECURITY_INCIDENTS {
        UUID id PK
        VARCHAR event_type
        VARCHAR severity
        TEXT description
        DATETIME created_at
        DATETIME resolved_at
    }

    OAUTH2_CLIENTS {
        UUID id PK
        VARCHAR client_id UK
        VARCHAR client_secret
        JSONB redirect_uris
        JSONB scopes
        JSONB grant_types
        JSONB response_types
        BOOLEAN is_confidential
        DATETIME created_at
        UUID user_id FK
    }

    OAUTH2_AUTHORIZATION_CODES {
        INTEGER id PK
        VARCHAR code UK
        VARCHAR client_id
        TEXT redirect_uri
        TEXT scope
        TEXT nonce
        INTEGER auth_time
        TEXT code_challenge
        VARCHAR code_challenge_method
        UUID user_id FK
    }

    OAUTH2_TOKENS {
        INTEGER id PK
        VARCHAR client_id
        VARCHAR token_type
        VARCHAR access_token UK
        VARCHAR refresh_token IX
        TEXT scope
        INTEGER issued_at
        INTEGER expires_in
        BOOLEAN revoked
        UUID user_id FK
    }

    OPTIONS_PRICES {
        DATETIME time PK
        VARCHAR symbol PK
        NUMERIC strike PK
        DATE expiry PK
        VARCHAR option_type PK
        NUMERIC bid
        NUMERIC ask
        NUMERIC last
        INTEGER volume
        INTEGER open_interest
        NUMERIC implied_volatility
    }

    PORTFOLIOS {
        UUID id PK
        UUID user_id FK UK
        VARCHAR name UK
        NUMERIC cash_balance
        DATETIME created_at
    }

    POSITIONS {
        UUID id PK
        UUID portfolio_id FK
        VARCHAR symbol
        NUMERIC quantity
        NUMERIC average_price
        NUMERIC entry_price
        NUMERIC current_price
        NUMERIC exit_price
        NUMERIC realized_pnl
        VARCHAR status
        DATETIME entry_date
        DATETIME exit_date
        NUMERIC strike
        DATE expiry
        VARCHAR option_type
    }

    ORDERS {
        UUID id PK
        UUID user_id FK
        UUID portfolio_id FK
        VARCHAR symbol
        VARCHAR side
        VARCHAR order_type
        VARCHAR status
        NUMERIC quantity
        NUMERIC price
        NUMERIC limit_price
        NUMERIC stop_price
        NUMERIC filled_quantity
        NUMERIC filled_price
        NUMERIC strike
        DATE expiry
        VARCHAR option_type
        VARCHAR broker
        VARCHAR broker_order_id
        DATETIME created_at
    }

    MARKET_TICKS {
        DATETIME time PK
        VARCHAR symbol PK
        NUMERIC price
        INTEGER volume
    }

    ML_MODELS {
        UUID id PK
        VARCHAR name
        VARCHAR version
        VARCHAR model_type
        VARCHAR algorithm
        VARCHAR artifact_uri
        VARCHAR model_artifact_url
        JSONB metrics
        JSONB hyperparameters
        JSONB training_metrics
        UUID created_by
        BOOLEAN is_production
        DATETIME created_at
    }

    MODEL_PREDICTIONS {
        UUID id PK
        UUID model_id FK
        VARCHAR symbol
        DATETIME prediction_time
        NUMERIC predicted_value
        NUMERIC confidence
    }

    RATE_LIMITS {
        UUID user_id PK FK
        VARCHAR endpoint PK
        DATETIME window_start PK
        INTEGER request_count
    }
```