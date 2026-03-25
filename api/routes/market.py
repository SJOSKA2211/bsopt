"""
Market Routes: Real-time Telemetry & Data Mesh
"""


import structlog
from fastapi import APIRouter, Depends

from api.responses import MsgspecJSONResponse
from src.auth.auth import get_current_active_user
from src.database.models import User
from src.ingestion.router import MarketDataRouter
from src.shared.schemas.market import MarketQuote, TickerSchema

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/market", tags=["Market"], default_response_class=MsgspecJSONResponse)
market_router_engine = MarketDataRouter()

# We use the TickerSchema from src.shared.schemas.market

@router.get("/tickers", response_model=list[TickerSchema])
async def get_tickers(current_user: User = Depends(get_current_active_user)):
    """
    Fetch live tickers for the global tape.
    OPTIMIZED: Uses the speculative concurrency MarketDataRouter with normalized MarketQuote objects.
    """
    try:
        from src.shared.config import settings
        symbols = settings.MARKET_TICKER_SYMBOLS
        
        import asyncio
        quotes: list[MarketQuote] = await asyncio.gather(*[market_router_engine.get_live_quote(s) for s in symbols])
        
        results = [q.to_ticker() for q in quotes]
            
        return results
    except Exception as e:
        logger.error("get_tickers_failed", error=str(e))
        # Fallback to an empty list rather than crashing the tape
        return []
