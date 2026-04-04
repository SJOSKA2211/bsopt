import asyncio
from sqlalchemy import text
from src.database import db_manager

async def verify_timescale():
    db_manager.initialize()
    engine = db_manager.async_engine
    
    queries = [
        ("Hypertables", "SELECT hypertable_name, num_chunks, compression_enabled FROM timescaledb_information.hypertables"),
        ("Background Job Stats", """
            SELECT 
                j.job_id, 
                j.proc_name, 
                js.last_run_status, 
                js.next_start 
            FROM timescaledb_information.jobs j 
            LEFT JOIN timescaledb_information.job_stats js ON j.job_id = js.job_id
        """),
        ("Continuous Aggregates", "SELECT view_name, materialization_hypertable_name FROM timescaledb_information.continuous_aggregates"),
        ("Compression Effectiveness", """
            SELECT 
                h.hypertable_name, 
                pg_size_pretty(s.uncompressed_total_size) as uncompressed, 
                pg_size_pretty(s.compressed_total_size) as compressed
            FROM timescaledb_information.hypertables h
            CROSS JOIN LATERAL hypertable_compression_stats(h.hypertable_name::regclass) s
            WHERE h.compression_enabled = true
        """)
    ]
    
    print("\n" + "="*60)
    print("   TIMESCALEDB OPTIMIZATION VERIFICATION")
    print("="*60)
    
    for title, query in queries:
        print(f"\n--- {title} ---")
        try:
            async with engine.connect() as conn:
                result = await conn.execute(text(query))
                rows = result.all()
                if not rows:
                    print("No data found.")
                for row in rows:
                    print(row)
        except Exception as e:
            print(f"Error fetching {title}: {e}")
    print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    asyncio.run(verify_timescale())
