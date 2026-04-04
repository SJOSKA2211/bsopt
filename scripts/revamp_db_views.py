import structlog
from sqlalchemy import text

from src.database import db_manager

logger = structlog.get_logger()


def revamp_diagnostics():
    db_manager.initialize()
    engine = db_manager.engine

    view_sql = """
    DROP VIEW IF EXISTS db_health_overview CASCADE;
    CREATE OR REPLACE VIEW db_health_overview AS
    SELECT
        now() as check_time,
        (SELECT count(*) FROM pg_stat_activity) as total_backends,
        (SELECT count(*) FROM pg_stat_activity WHERE state = 'active') as active_backends,
        (SELECT count(*) FROM pg_stat_activity WHERE state = 'idle') as idle_backends,
        (SELECT count(*) FROM pg_stat_activity WHERE wait_event_type IS NOT NULL) as waiting_backends,
        (SELECT pg_size_pretty(pg_database_size(current_database()))) as db_size,
        (SELECT version()) as pg_version,
        (SELECT extversion FROM pg_extension WHERE extname = 'timescaledb') as timescale_version;
    """

    with engine.connect() as conn:
        try:
            conn.execute(text(view_sql))
            print("✅ Database diagnostics view 'db_health_overview' REVAMPED.")
        except Exception as e:
            print(f"❌ Failed to revamp diagnostics view: {e}")

        try:
            conn.execute(text("""
                CREATE OR REPLACE VIEW timescale_jobs_overview AS
                SELECT
                    j.job_id,
                    j.proc_name,
                    j.hypertable_name,
                    js.last_run_status,
                    js.last_run_duration,
                    js.next_start,
                    js.total_runs,
                    js.total_failures
                FROM timescaledb_information.jobs j
                LEFT JOIN timescaledb_information.job_stats js ON j.job_id = js.job_id;
            """))
            print("✅ Database diagnostics view 'timescale_jobs_overview' CREATED.")
        except Exception as e:
            print(f"❌ Failed to create timescale jobs view: {e}")

        try:
            conn.execute(text("""
                CREATE OR REPLACE VIEW timescale_health_overview AS
                SELECT
                    h.hypertable_name,
                    h.num_chunks,
                    h.compression_enabled,
                    pg_size_pretty(s.before_compression_total_bytes) as uncompressed_size,
                    pg_size_pretty(s.after_compression_total_bytes) as compressed_size,
                    CASE 
                        WHEN s.before_compression_total_bytes > 0 
                        THEN round(100.0 * (s.before_compression_total_bytes - s.after_compression_total_bytes) / s.before_compression_total_bytes, 2)
                        ELSE 0 
                    END as compression_ratio_pct
                FROM timescaledb_information.hypertables h
                CROSS JOIN LATERAL hypertable_compression_stats(h.hypertable_name::regclass) s;
            """))
            print("✅ Database diagnostics view 'timescale_health_overview' CREATED.")
        except Exception as e:
            print(f"❌ Failed to create timescale health view: {e}")

        conn.commit()


if __name__ == "__main__":
    revamp_diagnostics()
