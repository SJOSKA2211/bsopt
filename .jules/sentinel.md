## 2026-01-26 - Prevent MFA Overwrite
**Vulnerability:** The `/mfa/setup` endpoint allowed users to re-initialize MFA even if it was already enabled, overwriting the existing secret.
**Learning:** Endpoints that modify security settings must check the current state (e.g., `is_mfa_enabled`) before proceeding. Relying on frontend to hide buttons is insufficient; the API must enforce the logic.
**Prevention:** Added a check `if user.is_mfa_enabled:` to raise `PermissionDeniedException` in `setup_mfa`. Users must strictly disable MFA (which requires verification) before setting it up again.
