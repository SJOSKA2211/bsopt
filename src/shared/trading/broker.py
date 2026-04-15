"""
Trading Broker Integration Substrate.
Provides a unified interface for multiple brokerage execution venues.
"""

import abc
from typing import Any

import httpx
import structlog

from src.shared.config import settings

logger = structlog.get_logger(__name__)


class TradingBroker(abc.ABC):
    @abc.abstractmethod
    async def submit_order(self, symbol: str, qty: float, side: str, type: str = "market") -> dict[str, Any]:
        pass

    @abc.abstractmethod
    async def get_order_status(self, order_id: str) -> dict[str, Any]:
        pass

    @abc.abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        pass


class AlpacaBroker(TradingBroker):
    """
    Production-grade Alpaca API integration.
    """

    def __init__(self, paper: bool = True):
        self.base_url = (
            "https://paper-api.alpaca.markets" if paper else "https://api.alpaca.markets"
        )
        self.headers = {
            "APCA-API-KEY-ID": settings.ALPACA_API_KEY,
            "APCA-API-SECRET-KEY": settings.ALPACA_API_SECRET,
        }

    async def submit_order(self, symbol: str, qty: float, side: str, type: str = "market") -> dict[str, Any]:
        async with httpx.AsyncClient() as client:
            payload = {
                "symbol": symbol,
                "qty": str(qty),
                "side": side,
                "type": type,
                "time_in_force": "gtc",
            }
            try:
                response = await client.post(
                    f"{self.base_url}/v2/orders", json=payload, headers=self.headers
                )
                response.raise_for_status()
                data = response.json()
                logger.info("alpaca_order_submitted", order_id=data["id"], symbol=symbol)
                return data
            except httpx.HTTPStatusError as e:
                logger.error("alpaca_order_failed", status_code=e.response.status_code, error=e.response.text)
                raise
            except Exception as e:
                logger.error("alpaca_submission_error", error=str(e))
                raise

    async def get_order_status(self, order_id: str) -> dict[str, Any]:
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    f"{self.base_url}/v2/orders/{order_id}", headers=self.headers
                )
                response.raise_for_status()
                return response.json()
            except Exception as e:
                logger.error("alpaca_get_order_failed", order_id=order_id, error=str(e))
                raise

    async def cancel_order(self, order_id: str) -> bool:
        async with httpx.AsyncClient() as client:
            try:
                response = await client.delete(
                    f"{self.base_url}/v2/orders/{order_id}", headers=self.headers
                )
                return response.status_code == 204
            except Exception as e:
                logger.error("alpaca_cancel_order_failed", order_id=order_id, error=str(e))
                return False


class MockTradingBroker(TradingBroker):
    """
    Deterministic Mock Broker for local dev or tests.
    REMOVED per "NO placeholders" mandate.
    """
    pass


def get_broker() -> TradingBroker:
    # Factory to return the configured broker
    if settings.BROKER_TYPE == "alpaca":
        return AlpacaBroker(paper=settings.BROKER_USE_PAPER)
    
    # Default fallback to Alpaca Paper for now, but strictly non-placeholder
    return AlpacaBroker(paper=True)
