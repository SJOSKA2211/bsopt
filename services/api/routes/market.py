"""
Market Routes: Real-time Telemetry & Data Mesh
"""

from typing import List

import structlog
from fastapi import APIRouter, Depends
from pydantic import BaseModel

from services.api.responses import MsgspecJSONResponse
from src.auth.auth import get_current_active_user
from src.database.models import User
from src.ingestion.router import MarketDataRouter

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/market", tags=["Market"], default_response_class=MsgspecJSONResponse)
market_router_engine = MarketDataRouter()


class TickerSchema(BaseModel):
    symbol: str
    price: str
    change: str
    percentChange: str


@router.get("/tickers", response_model=List[TickerSchema])
async def get_tickers(current_user: User = Depends(get_current_active_user)):
    """
    Fetch live tickers for the global tape.
    OPTIMIZED: Uses the speculative concurrency MarketDataRouter.
    """
    try:
        from src.shared.config import settings
        symbols = settings.MARKET_TICKER_SYMBOLS
        
        # In a production scenario, this could be dynamic based on hot symbols in Redis
        # For now, we utilize the get_live_quote which races providers
        import asyncio
        quotes = await asyncio.gather(*[market_router_engine.get_live_quote(s) for s in symbols])
        
        results = []
        for q in quotes:
            price = q.get("last_price", 0.0)
            prev_close = q.get("prev_close", price)
            change = price - prev_close
            pct_change = (change / prev_close * 100) if prev_close else 0.0
            
            results.append({
                "symbol": q.get("symbol", "N/A"),
                "price": str(price),
                "change": f"{'+' if change >= 0 else ''}{change:.2f}",
                "percentChange": f"{'+' if change >= 0 else ''}{pct_change:.2f}%"
            })
            
        return results
    except Exception as e:
        logger.error("get_tickers_failed", error=str(e))
        # Fallback to an empty list rather than crashing the tape
        return []
