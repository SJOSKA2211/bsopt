import asyncio
from sqlalchemy import text
from src.database import db_manager
import structlog

logger = structlog.get_logger(__name__)

# List of time-series tables and their time columns
HYPERTABLES = [
    ("market_ticks", "time", "1 day", "7 days"),
    ("options_prices", "time", "1 day", "30 days"),
    ("request_logs", "created_at", "1 day", "14 days"),
    ("audit_logs", "time", "1 week", "90 days"),
    ("data_audit_logs", "time", "1 week", "90 days"),
    ("email_logs", "created_at", "1 week", "30 days"),
    ("model_predictions", "timestamp", "1 day", "30 days"),
    ("rl_episodes", "created_at", "1 day", "30 days"),
    ("calibration_results", "created_at", "1 day", "30 days"),
    ("security_incidents", "detected_at", "1 week", "365 days")
]

async def revamp_fully():
    db_manager.initialize()
    engine = db_manager.engine
    
    # Phase 2 needs AUTOCOMMIT
    autocommit_engine = engine.execution_options(isolation_level="AUTOCOMMIT")
    
    with autocommit_engine.connect() as conn:
        print("\n STARTING GOD-MODE ENGINE REVAMP...")
        
        # 1. Core Extensions
        print("--- Phase 1: Core Extensions ---")
        try:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;"))
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_stat_statements;"))
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_prewarm;"))
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\";"))
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
            print(" Extensions active.")
        except Exception as e:
            print(f"   Phase 1 failed: {e}")

        # 2. Engine Runtime Tuning (ALTER SYSTEM)
        print("\n--- Phase 2: Engine Runtime Tuning ---")
        tuning_params = {
            "shared_buffers": "512MB", 
            "work_mem": "64MB",
            "maintenance_work_mem": "128MB",
            "random_page_cost": "1.1",
            "effective_io_concurrency": "200",
            "timescaledb.max_background_workers": "8",
            "max_parallel_workers_per_gather": "4",
            "jit": "on"
        }
        for param, value in tuning_params.items():
            try:
                conn.execute(text(f"ALTER SYSTEM SET {param} = '{value}';"))
                print(f"  SET {param} = {value}")
            except Exception as e:
                print(f"  ️ Failed to set {param}: {e}")
        
        conn.execute(text("SELECT pg_reload_conf();"))
        print(" Configuration reloaded.")

        # 3. Hypertable Optimization
        print("\n--- Phase 3: Hypertable Optimization ---")
        for table, time_col, chunk_interval, retention in HYPERTABLES:
            try:
                # Check if table exists
                res = conn.execute(text(f"SELECT count(*) FROM pg_tables WHERE tablename = '{table}'"))
                if res.scalar() == 0:
                    print(f"  ️ Skipping {table} (Table not found)")
                    continue
                
                # Make hypertable
                conn.execute(text(f"SELECT create_hypertable('{table}', '{time_col}', if_not_exists => TRUE);"))
                
                # Set chunk interval
                conn.execute(text(f"SELECT set_chunk_time_interval('{table}', INTERVAL '{chunk_interval}');"))
                
                # Enable compression
                conn.execute(text(f"ALTER TABLE {table} SET (timescaledb.compress, timescaledb.compress_orderby = '{time_col} DESC');"))
                
                # Add compression policy (wait 7 days before compressing)
                conn.execute(text(f"SELECT add_compression_policy('{table}', INTERVAL '7 days', if_not_exists => TRUE);"))
                
                # Add retention policy
                conn.execute(text(f"SELECT add_retention_policy('{table}', INTERVAL '{retention}', if_not_exists => TRUE);"))
                
                print(f"   {table} optimized (Chunk: {chunk_interval}, Retention: {retention})")
            except Exception as e:
                print(f"   Failed to optimize {table}: {e}")

        # 4. Multivariate Statistics
        print("\n--- Phase 4: Multivariate Statistics ---")
        stats_queries = [
            "CREATE STATISTICS IF NOT EXISTS s_options_prices ON symbol, strike, expiry FROM options_prices;",
            "CREATE STATISTICS IF NOT EXISTS s_market_ticks ON symbol, price FROM market_ticks;"
        ]
        for sql in stats_queries:
            try:
                conn.execute(text(sql))
                print(f"   {sql.split(' ')[4]} statistics created.")
            except Exception as e:
                print(f"  ️ Statistics error: {e}")

        # 5. Health Dashboard Views
        print("\n--- Phase 5: Health Dashboard ---")
        views_sql = [
            """
            CREATE OR REPLACE VIEW timescale_health_overview AS
            SELECT
                h.hypertable_name,
                h.num_chunks,
                h.compression_enabled,
                pg_size_pretty(COALESCE(s.before_compression_total_bytes, 0)) as uncompressed_size,
                pg_size_pretty(COALESCE(s.after_compression_total_bytes, 0)) as compressed_size,
                CASE 
                    WHEN s.before_compression_total_bytes > 0 
                    THEN round(100.0 * (s.before_compression_total_bytes - s.after_compression_total_bytes) / s.before_compression_total_bytes, 2)
                    ELSE 0 
                END as compression_ratio_pct
            FROM timescaledb_information.hypertables h
            LEFT JOIN LATERAL hypertable_compression_stats(h.hypertable_name::regclass) s ON TRUE;
            """,
            """
            CREATE OR REPLACE VIEW db_engine_health AS
            SELECT
                current_database() as db,
                version() as engine,
                (SELECT count(*) FROM pg_stat_activity WHERE state = 'active') as active_queries,
                (SELECT pg_size_pretty(pg_database_size(current_database()))) as total_size,
                round((SELECT sum(heap_blks_hit) / (sum(heap_blks_hit) + sum(heap_blks_read) + 1) FROM pg_statio_user_tables)::numeric, 4) as cache_hit_ratio;
            """
        ]
        for sql in views_sql:
            conn.execute(text(sql))
        print(" Monitoring views established.")

        # 6. God-Mode Prewarm Procedure
        print("\n--- Phase 6: God-Mode Prewarm ---")
        prewarm_sql = """
        CREATE OR REPLACE PROCEDURE god_mode_prewarm()
        LANGUAGE plpgsql
        AS $$
        DECLARE
            r RECORD;
        BEGIN
            RAISE NOTICE ' Warming up database engine...';
            FOR r IN SELECT tablename FROM pg_tables WHERE schemaname = 'public' LOOP
                RAISE NOTICE '  Prewarming table: %', r.tablename;
                BEGIN
                    PERFORM pg_prewarm(r.tablename::regclass);
                EXCEPTION WHEN OTHERS THEN
                    RAISE NOTICE '  ️ Failed to prewarm %: %', r.tablename, SQLERRM;
                END;
            END LOOP;
            RAISE NOTICE ' Engine is HOT and ready for HFT latency targets.';
        END;
        $$;
        """
        conn.execute(text(prewarm_sql))
        try:
            conn.execute(text("CALL god_mode_prewarm();"))
            print(" Database warmed up successfully.")
        except Exception as e:
            print(f"  ️ Prewarm failed: {e}")

        print("\n REVAMP COMPLETE. DATABASE IS FULLY OPTIMIZED.")

async def report_health():
    engine = db_manager.engine
    with engine.connect() as conn:
        print("\n" + "="*80)
        print(f"{'TIMESCALEDB ENGINE HEALTH REPORT':^80}")
        print("="*80)
        
        # 1. Engine Stats
        try:
            res = conn.execute(text("SELECT * FROM db_engine_health")).fetchone()
            if res:
                print(f"DB: {res[0]} | Cache Hit: {res[4]*100}% | Size: {res[3]} | Active: {res[2]}")
        except Exception as e:
            print(f"Error querying engine health: {e}")
        
        # 2. Hypertable Stats
        print("\n" + "-"*80)
        print(f"{'HYPERTABLE':<25} | {'CHUNKS':<8} | {'COMPRESSED':<12} | {'RATIO %':<10} | {'SIZE (U/C)'}")
        print("-" * 80)
        try:
            rows = conn.execute(text("SELECT * FROM timescale_health_overview")).fetchall()
            for row in rows:
                print(f"{row[0]:<25} | {row[1]:<8} | {'YES' if row[2] else 'NO':<12} | {row[5]:<10} | {row[3]} / {row[4]}")
        except Exception as e:
            print(f"Error querying hypertable stats: {e}")
        
        print("=" * 80 + "\n")

if __name__ == "__main__":
    asyncio.run(revamp_fully())
    asyncio.run(report_health())
