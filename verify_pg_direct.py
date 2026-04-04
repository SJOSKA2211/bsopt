import asyncio

import asyncpg


async def main():
    print("Connecting DIRECTLY to Postgres at localhost:5435...")
    try:
        # Port 5435 is mapped to 5432 in docker-compose for postgres service
        conn = await asyncpg.connect(
            user="admin",
            password="bsopt_postgres_secret",
            database="bsopt",
            host="localhost",
            port=5435,
            ssl='require'
        )
        version = await conn.fetchval("SELECT version();")
        print(f"✅ Postgres Direct Connection Successful! DB Version: {version}")
        await conn.close()
    except Exception as e:
        print(f"❌ Postgres Direct Connection Failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())
