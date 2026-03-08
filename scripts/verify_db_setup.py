import asyncio

from sqlalchemy import text

from src.database import create_tables, get_async_engine, get_engine


def test_sync():
    print("Testing sync engine...")
    engine = get_engine()
    with engine.connect() as conn:
        res = conn.execute(text("SELECT version()")).scalar()
        print(f"Postgres Version: {res}")
        
        res_ext = conn.execute(text("SELECT extname FROM pg_extension WHERE extname = 'timescaledb'")).scalar()
        print(f"TimescaleDB Extension (Sync): {res_ext}")

async def test_async():
    print("Testing async engine...")
    engine = get_async_engine()
    async with engine.connect() as conn:
        res = await conn.execute(text("SELECT extname FROM pg_extension WHERE extname = 'vector'"))
        ext = res.scalar()
        print(f"pgvector Extension (Async): {ext}")

if __name__ == '__main__':
    try:
        test_sync()
        asyncio.run(test_async())
        print("Ensuring tables are created...")
        create_tables()
        print("Status: healthy")
    except Exception as e:
        print(f"Error: {e}")
        exit(1)
