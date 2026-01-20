## 2025-01-26 - [Input Validation for Dynamic Types]
**Vulnerability:** The `is_prime` function accepted floating-point numbers and returned incorrect results (e.g., `is_prime(2.5) -> True`) due to implicit type behavior. It also crashed with `TypeError` when passed strings, exposing implementation details via stack traces if unhandled.
**Learning:** In dynamically typed languages like Python, type assumptions should be explicitly enforced at boundaries, especially for mathematical utility functions that rely on integer properties.
**Prevention:** Use `isinstance()` checks to enforce type constraints on public API boundaries. Fail fast and securely with descriptive errors.
