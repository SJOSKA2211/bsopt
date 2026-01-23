## 2026-01-23 - [HIGH] MFA Overwrite Vulnerability
**Vulnerability:** The `/mfa/setup` endpoint allowed authenticated users to generate a new MFA secret even if MFA was already enabled, without verifying the existing MFA code or password.
**Learning:** Checking for "logged in" is not enough for sensitive operations. We must check the *state* of the user (e.g., `is_mfa_enabled`) and enforce state transitions properly. State changes that decrease security or replace credentials must require re-authentication or proof of ownership of the current credential.
**Prevention:** Always verify if a security feature is already active before allowing its reconfiguration. Implement "sudo mode" (re-prompt for password/MFA) for sensitive settings changes.
