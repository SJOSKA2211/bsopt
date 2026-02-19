from unittest.mock import AsyncMock, patch

import pytest

from src.data.router import ROUTING_COUNT, SCRAPER_PARSE_SUCCESS, MarketDataRouter


@pytest.mark.asyncio
async def test_router_metrics_increment():
    with patch("src.data.router.NSEScraper") as mock_nse:
        mock_nse.return_value.get_ticker_data = AsyncMock(return_value={"price": 10.0})
        router = MarketDataRouter()

        # Get initial values
        try:
            initial_count = ROUTING_COUNT.labels(
                target="NSE", market="NSE"
            )._value.get()
            initial_success = SCRAPER_PARSE_SUCCESS.labels(market="NSE")._value.get()
        except AttributeError:
            initial_count = 0
            initial_success = 0

        await router.get_live_quote("SCOM.NR")

        # Verify increments
        assert (
            ROUTING_COUNT.labels(target="NSE", market="NSE")._value.get()
            == initial_count + 1
        )
        assert (
            SCRAPER_PARSE_SUCCESS.labels(market="NSE")._value.get()
            == initial_success + 1
        )
