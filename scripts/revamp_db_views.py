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
            conn.execute(text("""
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
            """))
            print("✅ Database diagnostics view 'db_health_overview' REVAMPED.")
        except Exception as e:
            print(f"❌ Failed to revamp diagnostics view: {e}")

        try:
            conn.execute(text("""
                CREATE OR REPLACE VIEW db_performance_stats AS
                WITH cache_stats AS (
                    SELECT 
                        sum(heap_blks_read) as heap_read,
                        sum(heap_blks_hit)  as heap_hit,
                        sum(idx_blks_read) as idx_read,
                        sum(idx_blks_hit)  as idx_hit
                    FROM pg_statio_user_tables
                )
                SELECT
                    round(100 * heap_hit / NULLIF(heap_hit + heap_read, 0), 2) as heap_cache_hit_ratio,
                    round(100 * idx_hit / NULLIF(idx_hit + idx_read, 0), 2) as index_cache_hit_ratio,
                    (SELECT count(*) FROM pg_locks) as lock_count,
                    (SELECT count(*) FROM pg_stat_activity WHERE wait_event_type = 'Lock') as blocked_queries;
            """))
            print("✅ Database performance view 'db_performance_stats' CREATED.")
        except Exception as e:
            print(f"❌ Failed to create performance stats view: {e}")

        try:
            conn.execute(text("""
                CREATE OR REPLACE VIEW index_efficiency_stats AS
                SELECT
                    relname as table_name,
                    indexrelname as index_name,
                    idx_scan as index_scans,
                    idx_tup_read as tuples_read,
                    idx_tup_fetch as tuples_fetched,
                    pg_size_pretty(pg_relation_size(indexrelid)) as index_size
                FROM pg_stat_user_indexes
                WHERE idx_scan > 0
                ORDER BY idx_scan DESC
                LIMIT 10;
            """))
            print("✅ Database diagnostics view 'index_efficiency_stats' CREATED.")
        except Exception as e:
            print(f"❌ Failed to create index efficiency stats view: {e}")

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
