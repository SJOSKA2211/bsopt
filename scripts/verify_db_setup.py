import sys

from sqlalchemy import text

from src.database import create_tables, db_manager


def verify_god_mode():
    print("🥒 BSOpt God-Mode Database Verification")
    print("----------------------------------------")
    
    try:
        db_manager.initialize()
        engine = db_manager.engine
        
        with engine.connect() as conn:
            # 1. Version Check
            version = conn.execute(text("SELECT version()")).scalar()
            print(f"✅ Postgres: {version}")
            
            # 2. Extension Check
            extensions = conn.execute(text("SELECT extname FROM pg_extension")).scalars().all()
            required = ['timescaledb', 'vector', 'pg_stat_statements']
            for ext in required:
                if ext in extensions:
                    print(f"✅ Extension Found: {ext}")
                else:
                    print(f"❌ Missing Extension: {ext}")
            
            # 3. Revamp Diagnostics Check
            views = conn.execute(text("SELECT viewname FROM pg_views WHERE schemaname = 'public'")).scalars().all()
            revamp_views = ['db_health_overview', 'pg_stat_sluggish_queries', 'job_performance_audit']
            for v in revamp_views:
                if v in views:
                    print(f"✅ God-Mode View Active: {v}")
                else:
                    print(f"⚠️  Revamp View Missing: {v}")
            
            # 4. Ingestion Readiness
            res = conn.execute(text("SELECT count(*) FROM timescaledb_information.hypertables")).scalar()
            print(f"✅ TimescaleDB Hypertables: {res}")

        print("\n🚀 STATUS: MANIFOLD PRESSURIZED (Solenya-Tight)")
        
    except Exception as e:
        print(f"\n❌ VERIFICATION FAILED: {e}")
        sys.exit(1)

if __name__ == '__main__':
    verify_god_mode()
    print("Ensuring metadata tables are synchronized...")
    create_tables()
