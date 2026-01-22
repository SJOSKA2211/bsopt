## 2026-01-22 - Broken Access Control: Enterprise Tier Escalation
**Vulnerability:** The `require_admin` dependency was simply checking if a user had the "enterprise" subscription tier. This meant any paying customer automatically gained full administrative access to the system (listing all users, viewing system stats).
**Learning:** Tying security roles directly to billing/subscription tiers creates dangerous privilege escalation paths. Business logic (subscription status) should be decoupled from Security logic (administrative privileges).
**Prevention:** Always use dedicated role flags (e.g., `is_admin` or a specific `admin` role) for privileged operations. Ensure "admin" roles cannot be self-assigned or purchased via standard checkout flows.
