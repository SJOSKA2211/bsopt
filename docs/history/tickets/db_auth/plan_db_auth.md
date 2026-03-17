# Plan: Database Auth

## Steps
1.  **Create Schema**:
    -   `users` table with `password_hash`.
    -   `sessions` table.
    -   `oauth_accounts` table.
    -   `email_verification_tokens` table.

2.  **Create Functions**:
    -   `create_user`: Securely hash passwords.
    -   `verify_password`: Check credentials.
    -   `upsert_oauth_user`: Handle OAuth login/registration logic.

## Validation
-   Run the SQL scripts against a local Postgres instance (or CI).
-   Verify that functions return expected UUIDs.

