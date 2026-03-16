# CI/CD Pipeline and Bot Remediation Plan

## Objective
Diagnose and fix the 0.0-second pipeline crashes and the linting/testing annotations caused by automated security and performance bots.

## 1. Fix the `main.yml` 0.0-Second Syntax Crash
**Issue:** The `.github/workflows/main.yml` file is failing to parse before execution. In the `continuous-training` job, the `python -c` script inside the `run: |` block scalar loses its indentation. In YAML, a block scalar string ends immediately if any line has less indentation than the first line. The `import os, sys` is at column 0.
**Action:** 
- Edit `.github/workflows/main.yml`.
- Add proper indentation (10 spaces) to the entire inline Python script starting from `import os, sys` to the `except` block.

## 2. Fix the "2 Annotations" (Security & Linting Job)
**Issue 1: Unused Variable (`F841`) in `auth.py`**
- **Location:** `services/api/routes/auth.py` inside the `login()` function.
- **Cause:** `user_id, email, tier, is_active = row` unpacks the `is_active` variable, but it is never subsequently used, triggering a Flake8/Ruff `F841` violation.
- **Action:** Change the unpack to `user_id, email, tier, _ = row`.

**Issue 2: Security Vulnerability & Broken Test Assertion**
- **Location:** `services/api/routes/auth.py`
- **Cause:** In `_send_password_reset_email()`, the reset token is explicitly logged (`logger.info(f"User reset link: {token}")`).
- **Action:** Remove the plaintext token logging in `auth.py`. 

## Next Steps
Once this plan is approved, I will sequentially implement the fixes and verify using the local linting/testing tools to guarantee the pipeline runs perfectly.