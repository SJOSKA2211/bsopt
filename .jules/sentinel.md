## 2026-01-30 - WebSocket Authentication Gap
**Vulnerability:** The `/ws/market-data` endpoint was exposed without any authentication, allowing unauthorized access to real-time market data.
**Learning:** WebSocket connections in browsers cannot send custom headers during the handshake, rendering standard `OAuth2PasswordBearer` dependencies ineffective for initial connection auth.
**Prevention:** Always implement a dedicated dependency for WebSocket authentication that accepts tokens via query parameters (or ticket-based auth) and validates them before accepting the connection.
