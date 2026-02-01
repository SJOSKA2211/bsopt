## 2026-02-18 - [MFA Overwrite Vulnerability]
**Vulnerability:** The `/auth/mfa/setup` endpoint allowed re-initialization of MFA secrets for users with MFA already enabled, permitting attackers with session access to lock out victims or bypass MFA.
**Learning:** Checking `is_mfa_enabled` was missing in the setup flow. Auth flows must strictly enforce state transitions (disabled -> enabled).
**Prevention:** Always verify the current state of a security control before allowing changes. Ensure MFA setup endpoints are gated by `if not user.is_mfa_enabled`.
