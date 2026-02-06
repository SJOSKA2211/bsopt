## 2026-02-06 - [Missing Authorization on User List]
**Vulnerability:** The `list_users` endpoint in `src/api/routes/users.py` was accessible to any authenticated user, allowing regular users to view all user profiles.
**Learning:** Endpoints that return lists of resources often lack explicit authorization checks, relying solely on authentication. Middleware handles authentication but not permission checking.
**Prevention:** Always verify if an endpoint should be restricted to specific roles (like "admin") and apply `RoleChecker` or similar authorization dependencies.
