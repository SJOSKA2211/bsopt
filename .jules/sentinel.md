## 2025-02-18 - MFA Setup Overwrite Vulnerability
**Vulnerability:** The `/mfa/setup` endpoint did not check if MFA was already enabled, allowing an authenticated attacker (e.g., via session hijacking) to overwrite the existing MFA secret and take over the account's 2FA.
**Learning:** Authentication flows involving state changes (like enabling MFA) must strictly validate the current state (is it already enabled?) to prevent logical bypasses.
**Prevention:** Enforce a "MFA must be disabled" check in the MFA setup handler.
