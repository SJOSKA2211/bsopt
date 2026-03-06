import sys

from sqlalchemy import create_engine, text

from src.config import get_settings


def verify_connection():
    print(" 🥒 BSOpt God-Mode Database Verification")
    print("---------------------------------------")

    try:
        settings = get_settings()
        db_url = settings.DATABASE_URL.replace("+asyncpg", "")

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

        engine = create_engine(db_url)
        with engine.connect() as conn:
            # 1. Basic Connectivity
            result = conn.execute(text("SELECT 1")).scalar()
            if result != 1:
                raise Exception("Unexpected result from SELECT 1")

            # 2. Engine Version
            version = conn.execute(text("SELECT version()")).scalar()
            print(f"✅ Backend: {version}")

            # 3. TimescaleDB Check
            try:
                ts_version = conn.execute(
                    text("SELECT extversion FROM pg_extension WHERE extname = 'timescaledb'")
                ).scalar()
                if ts_version:
                    print(f"✅ TimescaleDB: {ts_version}")
                else:
                    print("❌ TimescaleDB extension not found!")
            except Exception:
                print("❌ Failed to query TimescaleDB status")

            # 4. Hypertable Verification
            hypertables = conn.execute(
                text("SELECT count(*) FROM timescaledb_information.hypertables")
            ).scalar()
            print(f"✅ Hypertables: {hypertables} active")

            # 5. Compression Status
            compressed = conn.execute(
                text(
                    "SELECT count(*) FROM timescaledb_information.hypertables WHERE compression_enabled = true"
                )
            ).scalar()
            print(f"✅ Compression: {compressed} hypertables optimized")

            # 6. CAGG Status
            caggs = conn.execute(
                text("SELECT count(*) FROM timescaledb_information.continuous_aggregates")
            ).scalar()
            print(f"✅ Continuous Aggregates: {caggs} pressurized")

            # 7. pg_repack Check
            try:
                repack_version = conn.execute(
                    text("SELECT extversion FROM pg_extension WHERE extname = 'pg_repack'")
                ).scalar()
                if repack_version:
                    print(f"✅ pg_repack: {repack_version} ready for zero-downtime maintenance")
                else:
                    print("⚠️ pg_repack: Extension not installed (Maintenance may require LOCKS)")
            except Exception:
                print("⚠️ pg_repack: Status unknown")

            # 8. RLS Enforcement Check
            rls_count = conn.execute(text("SELECT count(*) FROM pg_policy")).scalar()
            if rls_count > 0:
                print(f"✅ Row Level Security: {rls_count} policies shielding user data")
            else:
                print("❌ Row Level Security: NO POLICIES FOUND! (Data isolation risk)")

            # 9. Scheduled Jobs Check
            jobs_count = conn.execute(
                text("SELECT count(*) FROM timescaledb_information.jobs")
            ).scalar()
            print(f"✅ Maintenance Automation: {jobs_count} background jobs scheduled")

        print("\n✨ Database is Solenya-tight. God Mode Active! 🥒")

    except Exception as e:
        print(f"❌ Verification Failed: {e}")
        sys.exit(1)


# Alias for backward compatibility
verify_postgres_connection = verify_connection

if __name__ == "__main__":
    verify_connection()
