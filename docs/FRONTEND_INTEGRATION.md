# Frontend-Backend Integration Guide

This document outlines the RESTful API integration architecture for the Black-Scholes platform.

## Architecture Overview

The system uses a **FastAPI** backend and a **React (Vite)** frontend. Communication is handled via **Axios** with centralized interceptors.

### API Base Configuration
- **Development**: `http://localhost:8000/api/v1` (Proxied via Vite `/api`)
- **Production**: Configured via `VITE_API_URL` environment variable.

## Authentication (JWT + MFA)

### Token Management
- **Access Token**: Stored in `localStorage`. Included in the `Authorization: Bearer <token>` header.
- **Refresh Token**: Stored in `localStorage`. Used automatically by the interceptor when a 401 error occurs.

### Automated Refresh Logic
The `apiClient.ts` interceptor handles token rotation:
1. Detects `401 Unauthorized` response.
2. Calls `/auth/refresh` with the stored refresh token.
3. Retries the original request with the new access token.

## Data Contracts & Type Safety

### TypeScript Interfaces
Types are centralized in `frontend/src/types/` and mirror the backend Pydantic models:
- `Pricing`: `PriceRequest`, `PriceResponse`, `GreeksRequest`, etc.
- `Auth`: `LoginRequest`, `LoginResponse`, `TokenResponse`.
- `User`: `UserResponse`, `UserStatsResponse`.

### Standard Response Envelope
All API responses follow the `DataResponse<T>` or `SuccessResponse` schema:
```typescript
interface DataResponse<T> {
  success: boolean;
  data: T;
  message?: string;
}
```

## Error Handling

### Centralized Interceptor
Errors are normalized in `apiClient.ts`. The frontend receives a standardized `ErrorResponse` object:
- **422 Validation Error**: Contains field-specific error details.
- **401/403 Auth Error**: Triggers logout or token refresh.
- **500 Server Error**: Returns a generic error message with a unique `request_id`.

### UI Hook (`useApi`)
Use the `useApi` hook to manage loading and error states in components:
```typescript
const { data, loading, error, execute } = useApi(priceOption);
```

## Programmatic Access (API Keys)

The platform supports secure API Keys for automated trading and high-frequency access.

### Generating Keys
Users can generate keys via the Dashboard (using `createAPIKey` from the `users` service). Keys are hashed on the backend and only shown once.

### Usage
Add the `X-API-Key` header to your requests:
```bash
curl -X POST "http://localhost:8000/api/v1/pricing/price" \
  -H "X-API-Key: bs_your_secret_key_here" \
  -H "Content-Type: application/json" \
  -d '{...}'
```

## Rate Limiting & Tiers

The API enforces tiered rate limits based on user status:

| Tier | Limit (Requests/Hour) |
| :--- | :--- |
| **Free** | 100 |
| **Pro** | 1,000 |
| **Enterprise** | 10,000 |

### Handling Rate Limits
- Responses include `X-RateLimit-Limit` and `X-RateLimit-Remaining` headers.
- If exceeded (429), the `apiClient` automatically extracts the `retryAfter` value.

## Resiliency Features

### Circuit Breakers
Critical pricing routes are protected by a Circuit Breaker. If the backend pricing engine experiences failures (threshold: 10), the circuit opens for 60 seconds, returning a `503 Service Unavailable` error to prevent cascading failures.

### Timeouts & Retries
- **Timeout**: 10 seconds for all requests.
- **Retries**: 2 retries for failed `GET` requests (network errors/timeouts).

## Monitoring & Observability

### Health Checks
- **Endpoint**: `/api/v1/health`
- **Hook**: `useSystemHealth` provides real-time status of the API, Database, and Cache.

### Metrics
The system is instrumented with Prometheus. Metrics are available at `/metrics` (restricted) and visualized via the **Grafana Dashboard** (Port 3001).

### System Status
Real-time circuit breaker states can be monitored via `/api/v1/system/status`.
