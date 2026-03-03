import asyncio
import uuid

import bcrypt
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

DATABASE_URL = "postgresql+asyncpg://admin:password@localhost:5433/bsopt"


async def main():
    engine = create_async_engine(DATABASE_URL)
    pwd = bcrypt.hashpw(b"password", bcrypt.gensalt()).decode()
    user_id = str(uuid.uuid4())
    async with engine.begin() as conn:
        await conn.execute(
            text("""
            INSERT INTO users (id, email, hashed_password, full_name, is_active, is_verified, tier)
            VALUES (:id, 'dev@example.com', :pwd, 'Dev User', true, true, 'free')
            ON CONFLICT (email) DO UPDATE SET hashed_password = EXCLUDED.hashed_password;
            """),
            {"id": user_id, "pwd": pwd},
        )
    print("Seeded successfully")


if __name__ == "__main__":
    asyncio.run(main())
