import asyncio
from sqlalchemy import text
from src.database import db_manager
import os

async def report_health():
    # Ensure environment variables are loaded if this script is run standalone
    # (though I'll run it with the environment already set up)
    
    db_manager.initialize()
    engine = db_manager.engine
    
    with engine.connect() as conn:
        print("\n" + "="*80)
        print(f"{'TIMESCALEDB OPTIMIZATION & HEALTH REPORT':^80}")
        print("="*80)
        
        try:
            result = conn.execute(text("SELECT * FROM timescale_health_overview"))
            rows = result.fetchall()
            
            if not rows:
                print("No hypertables found or view is empty.")
            else:
                print(f"{'HYPERTABLE':<25} | {'CHUNKS':<8} | {'COMPRESSED':<12} | {'RATIO %':<10} | {'SIZE (U/C)'}")
                print("-" * 80)
                for row in rows:
                    name = row[0]
                    chunks = row[1]
                    comp_enabled = "YES" if row[2] else "NO"
                    u_size = row[3]
                    c_size = row[4]
                    ratio = row[5]
                    print(f"{name:<25} | {chunks:<8} | {comp_enabled:<12} | {ratio:<10} | {u_size} / {c_size}")
        except Exception as e:
            print(f"Error querying health view: {e}")
            
        print("=" * 80 + "\n")

if __name__ == "__main__":
    asyncio.run(report_health())
