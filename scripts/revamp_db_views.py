import structlog
from sqlalchemy import text

from src.database import db_manager

logger = structlog.get_logger()

def revamp_diagnostics():
    db_manager.initialize()
    engine = db_manager.engine
    
    view_sql = """
    CREATE OR REPLACE VIEW db_health_overview AS
    SELECT
        now() as check_time,
        (SELECT count(*) FROM pg_stat_activity) as active_backends,
        (SELECT count(*) FROM pg_stat_activity WHERE wait_event_type IS NOT NULL) as waiting_backends,
        (SELECT version()) as pg_version,
        (SELECT extversion FROM pg_extension WHERE extname = 'timescaledb') as timescale_version;
    """
    
    perf_view_sql = """
    CREATE OR REPLACE VIEW db_performance_stats AS
    SELECT
        round((100 * sum(heap_blks_hit) / (sum(heap_blks_hit) + sum(heap_blks_read) + 1))::numeric, 2) as heap_cache_hit_ratio,
        round((100 * sum(idx_blks_hit) / (sum(idx_blks_hit) + sum(idx_blks_read) + 1))::numeric, 2) as index_cache_hit_ratio,
        (SELECT count(*) FROM pg_stat_activity WHERE state = 'active' AND wait_event_type = 'Lock') as blocked_queries
    FROM pg_statio_user_tables;
    """
    
    with engine.connect() as conn:
        # We need to use autocommit for these view creations sometimes, 
        # but here we just use conn.execute and then commit if needed.
        # SQLAlchemy 2.0+ requires explicit commit for some setups, 
        # or it handles it if we are in a transaction.
        conn.execute(text(view_sql))
        conn.execute(text(perf_view_sql))
        # conn.commit() is not always available on Connection depending on how it was created
        print("✅ Database diagnostics views 'db_health_overview' and 'db_performance_stats' REVAMPED.")

if __name__ == "__main__":
    revamp_diagnostics()
