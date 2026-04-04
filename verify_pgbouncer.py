import asyncio
import os
import sys

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

# Add src to path
sys.path.append(os.getcwd())


async def verify_pgbouncer():
    print("Connecting to PgBouncer at localhost:6432...")
    # SQLALchemy URL
    # Using asyncpg driver
    db_url = "postgresql+asyncpg://admin:bsopt_postgres_secret@localhost:6432/bsopt?ssl=require"
    
    try:
        engine = create_async_engine(db_url)
        async with engine.connect() as conn:
            result = await conn.execute(text("SELECT version();"))
            version = result.scalar()
            print(f"✅ PgBouncer Connection Successful! DB Version: {version}")
            
        await engine.dispose()
        print("🔌 Connection closed.")
    except Exception as e:
        print(f"❌ PgBouncer Connection Failed: {e}")

async def main():
    await verify_pgbouncer()

if __name__ == "__main__":
    asyncio.run(main())
