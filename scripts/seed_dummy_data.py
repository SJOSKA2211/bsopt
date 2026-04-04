
import asyncio
from datetime import date, datetime, timedelta
from decimal import Decimal

from sqlalchemy import text

from src.database import db_manager


async def seed():
    db_manager.initialize()
    async with db_manager.async_engine.connect() as conn:
        # Create symbols if not exists
        await conn.execute(text("INSERT INTO symbols (symbol, name, exchange) VALUES ('SPY', 'SPDR S&P 500 ETF Trust', 'NYSE') ON CONFLICT DO NOTHING"))
        
        # Insert dummy option prices
        now = datetime.now()
        data = []
        for i in range(1100):
            t = now - timedelta(minutes=i)
            # symbol, expiry, strike, option_type, last, delta, gamma, implied_volatility, time
            data.append({
                "symbol": "SPY",
                "expiry": date(2026, 6, 19),
                "strike": Decimal("500.00"),
                "option_type": "call",
                "last": Decimal("15.50") + Decimal(str(i % 10)) / 10,
                "delta": Decimal("0.55"),
                "gamma": Decimal("0.02"),
                "implied_volatility": Decimal("0.18"),
                "time": t
            })
        
        query = text("""
            INSERT INTO options_prices (symbol, expiry, strike, option_type, last, delta, gamma, implied_volatility, time)
            VALUES (:symbol, :expiry, :strike, :option_type, :last, :delta, :gamma, :implied_volatility, :time)
        """)
        
        for chunk in [data[i:i + 100] for i in range(0, len(data), 100)]:
            await conn.execute(query, chunk)
        
        await conn.commit()
        print("Seeded 1100 dummy records.")
    
    await db_manager.dispose()

if __name__ == "__main__":
    asyncio.run(seed())
