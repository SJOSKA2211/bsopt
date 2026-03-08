import asyncio
import os
import uuid

from sqlalchemy import text

# Use the internal docker URL for seeding
DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql+asyncpg://admin:29a47839acf362c9ebb5679a@postgres:5432/bsopt"
)


async def seed_core():
    """
    God-Mode Seeding: Uses native DB procedures for security and speed.
    """
    from src.database import db_manager
    db_manager.initialize()
    engine = db_manager.async_engine

    async with engine.begin() as conn:
        print("🥒 Seeding Dev User...")
        # Use our revamped native registration (Handles hashing DB-side)
        await conn.execute(
            text("SELECT register_user_native(:email, :password, :name)"),
            {"email": "admin@bsopt.com", "password": "admin_password_123", "name": "Super Admin"},
        )

        # Elevate to enterprise tier
        await conn.execute(
            text(
                "UPDATE users SET tier = 'enterprise', is_verified = true WHERE email = 'admin@bsopt.com'"
            )
        )

        # Create a default portfolio
        await conn.execute(
            text("""
                INSERT INTO portfolios (id, user_id, name, cash_balance)
                SELECT :pid, id, 'Primary Alpha', 1000000.00 FROM users WHERE email = 'admin@bsopt.com'
                ON CONFLICT DO NOTHING
            """),
            {"pid": str(uuid.uuid4())},
        )

    print("✨ Seed complete. Solenya-tight! 🥒")


if __name__ == "__main__":
    asyncio.run(seed_core())
