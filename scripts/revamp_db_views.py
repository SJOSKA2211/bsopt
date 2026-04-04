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
            conn.commit()
            print("✅ Database diagnostics view 'db_health_overview' REVAMPED.")
        except Exception as e:
            print(f"❌ Failed to revamp diagnostics view: {e}")


if __name__ == "__main__":
    revamp_diagnostics()
