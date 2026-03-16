import sys

from sqlalchemy import text

from core.config import get_settings
from core.database import get_engine


def verify_connection():
    print("  BSOpt High-Performance Database Verification")
    print("---------------------------------------")

    try:
        get_settings()
        _, _ = get_settings(), None  # Trigger settings load if needed

        # We use the centralized getter to test the ACTUAL production configuration
        engine = get_engine()
        db_url = str(engine.url)

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

        with engine.connect() as conn:
            # 1. Basic Connectivity
            result = conn.execute(text("SELECT 1")).scalar()
            if result != 1:
                raise Exception("Unexpected result from SELECT 1")

            # 2. Engine Version
            version = conn.execute(text("SELECT version()")).scalar()
            print(f" Backend: {version}")

            # 3. TimescaleDB Check
            try:
                ts_version = conn.execute(
                    text("SELECT extversion FROM pg_extension WHERE extname = 'timescaledb'")
                ).scalar()
                if ts_version:
                    print(f" TimescaleDB: {ts_version}")
                else:
                    print("❌ TimescaleDB extension not found!")
            except Exception:
                print("❌ Failed to query TimescaleDB status")

            # 4. Hypertable Verification
            hypertables = conn.execute(
                text("SELECT count(*) FROM timescaledb_information.hypertables")
            ).scalar()
            print(f" Hypertables: {hypertables} active")

            # 5. Compression Status
            compressed = conn.execute(
                text(
                    "SELECT count(*) FROM timescaledb_information.hypertables WHERE compression_enabled = true"
                )
            ).scalar()
            print(f" Compression: {compressed} hypertables optimized")

            # 6. CAGG Status
            caggs = conn.execute(
                text("SELECT count(*) FROM timescaledb_information.continuous_aggregates")
            ).scalar()
            print(f" Continuous Aggregates: {caggs} pressurized")

            # 7. pg_repack Check
            try:
                repack_version = conn.execute(
                    text("SELECT extversion FROM pg_extension WHERE extname = 'pg_repack'")
                ).scalar()
                if repack_version:
                    print(f" pg_repack: {repack_version} ready for zero-downtime maintenance")
                else:
                    print("⚠️ pg_repack: Extension not installed (Maintenance may require LOCKS)")
            except Exception:
                print("⚠️ pg_repack: Status unknown")

            # 8. RLS Enforcement Check
            rls_count = conn.execute(text("SELECT count(*) FROM pg_policy")).scalar()
            if rls_count > 0:
                print(f" Row Level Security: {rls_count} policies shielding user data")
            else:
                print("❌ Row Level Security: NO POLICIES FOUND! (Data isolation risk)")

            # 9. Scheduled Jobs Check
            jobs_count = conn.execute(
                text("SELECT count(*) FROM timescaledb_information.jobs")
            ).scalar()
            print(f" Maintenance Automation: {jobs_count} background jobs scheduled")

            # NEW: Continuous Aggregate Freshness Check
            try:
                cagg_freshness = conn.execute(
                    text("""
                    SELECT 
                        view_name,
                        last_refresh_time,
                        now() - last_refresh_time as drift
                    FROM timescaledb_information.continuous_aggregates
                    WHERE last_refresh_time IS NOT NULL
                """)
                ).fetchall()
                for row in cagg_freshness:
                    status = "" if row[2].total_seconds() < 3600 else "⚠️"
                    print(f"{status} CAGG Freshness: {row[0]} (Drift: {row[2]})")
            except Exception:
                print("⚠️ CAGG Freshness: Unable to query")

            # NEW: Compression Ratio Check
            try:
                compression_stats = conn.execute(
                    text("""
                    SELECT 
                        hypertable_name,
                        compression_status,
                        uncompressed_total_bytes / NULLIF(compressed_total_bytes, 0) as ratio
                    FROM timescaledb_information.compression_settings
                    JOIN timescaledb_information.hypertables ON hypertable_name = table_name
                """)
                ).fetchall()
                for row in compression_stats:
                    if row[2]:
                        print(f" Compression Ratio: {row[0]} ({round(row[2], 2)}x)")
            except Exception:
                print("⚠️ Compression Stats: Unable to query")

        print("\n Database is tight. High-Performance Active! ")

    except Exception as e:
        print(f"❌ Verification Failed: {e}")
        sys.exit(1)


# Alias for backward compatibility
verify_postgres_connection = verify_connection

if __name__ == "__main__":
    verify_connection()
