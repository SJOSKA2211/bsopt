## 2026-02-04 - MFA Overwrite Vulnerability
**Vulnerability:** The `/mfa/setup` endpoint allowed re-initialization of MFA secrets even if MFA was already enabled, without requiring prior disablement.
**Learning:** Logic flows for sensitive state changes (like MFA) must explicitly check the current state to prevent accidental or malicious overwrites.
**Prevention:** Ensure state transition diagrams are verified: `Enabled -> Setup` should be invalid. Only `Disabled -> Setup` or `Enabled -> Disable` should be allowed.
