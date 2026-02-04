# Neon (Postgres) Setup Guide

The BSOpt platform has been upgraded to use a PostgreSQL database, specifically targeted for **Neon (Serverless Postgres)**.

## 1. Get Your Connection String
1.  Go to the Neon Console.
2.  Create a project.
3.  Copy the connection string (e.g., `postgres://user:password@ep-xyz.aws.neon.tech/neondb?sslmode=require`).

## 2. Configure Environment
Create or update your `.env` file in the project root:

```bash
DATABASE_URL="postgres://user:password@ep-xyz.aws.neon.tech/neondb?sslmode=require"
```

## 3. Initialize Schema
Run the schema creation script using `psql` or a database tool:

```bash
psql $DATABASE_URL -f src/database/schema.sql
```

## 4. Verify Connection
Run the verification script:
```bash
python3 src/database/verify.py
```

## 5. Troubleshooting
-   **Missing Driver**: Ensure `psycopg2-binary` is installed (`pip install psycopg2-binary`).
-   **SSL**: Neon requires SSL. Ensure `sslmode=require` is in the URL.
