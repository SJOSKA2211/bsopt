## 2026-01-31 - [IDOR in User Profile Access]
**Vulnerability:** IDOR in `get_user_by_id` allowed "enterprise" tier users to access any user profile.
**Learning:** "Enterprise" tier was implicitly treated as an admin/privileged role without proper scoping or explicit "admin" role check.
**Prevention:** Strictly enforce `current_user.id == target_id` for resource access unless explicit `admin` role is verified. Do not rely on subscription tiers for administrative privileges.
