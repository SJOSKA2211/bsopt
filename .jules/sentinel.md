## 2026-02-05 - Missing Authorization on Sensitive Endpoints
**Vulnerability:** The `list_users` endpoint was exposed to all authenticated users, allowing non-admin users to list all users in the system.
**Learning:** Middleware (`JWTAuthenticationMiddleware`) handles authentication but does not enforce authorization (RBAC). Explicit dependency injection with `RoleChecker` is required for sensitive routes.
**Prevention:** Always verify if a route handling sensitive data has an explicit role check or permission dependency. Do not assume authentication middleware implies authorization.
