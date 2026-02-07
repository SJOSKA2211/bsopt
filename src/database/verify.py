import sys

from sqlalchemy import create_engine, text

from src.config import get_settings


def verify_connection():
    print("🥒 BSOpt Database Verification Tool")
    print("-----------------------------------")

    try:
        settings = get_settings()
        db_url = settings.DATABASE_URL

        # Mask password for display
        safe_url = db_url
        if "@" in safe_url:
            prefix = safe_url.split("@")[0]
            suffix = safe_url.split("@")[1]
            if ":" in prefix and "//" in prefix:
                proto = prefix.split("://")[0]
                user = prefix.split("://")[1].split(":")[0]
                safe_url = f"{proto}://{user}:****@{suffix}"

        print(f"Target: {safe_url}")

        if "sqlite" in db_url:
            print(
                "⚠️  WARNING: Using SQLite. Neon/Postgres is recommended for production."
            )

        engine = create_engine(db_url)
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1")).scalar()
            version = (
                conn.execute(text("SELECT version()")).scalar()
                if "postgres" in db_url
                else "SQLite"
            )

        if result == 1:
            print("✅ Connection Successful!")
            print(f"   Backend: {version}")
        else:
            print("❌ Connection Failed: Unexpected result.")
            sys.exit(1)

    except Exception as e:
        print(f"❌ Connection Failed: {e}")
        sys.exit(1)


# Alias for backward compatibility with tests
verify_postgres_connection = verify_connection

if __name__ == "__main__":
    verify_connection()
