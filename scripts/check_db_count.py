
import asyncio

from sqlalchemy import text

from src.database import db_manager


async def check():
    db_manager.initialize()
    async with db_manager.async_engine.connect() as conn:
        res = await conn.execute(text("SELECT count(*) FROM options_prices"))
        print(f"Count: {res.scalar()}")
    await db_manager.dispose()

if __name__ == "__main__":
    asyncio.run(check())
