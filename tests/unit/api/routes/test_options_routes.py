from datetime import UTC, date, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from api.index import app

client = TestClient(app)

@pytest.fixture
def mock_options_user():
    from src.auth.auth import get_current_active_user
    mock_user = MagicMock()
    app.dependency_overrides[get_current_active_user] = lambda: mock_user
    yield mock_user
    app.dependency_overrides.clear()

@pytest.fixture
def mock_greeks_mesh():
    with patch("api.routes.options._greeks_mesh") as mock_mesh:
        yield mock_mesh

def test_get_realtime_greeks_success(mock_options_user, mock_greeks_mesh):
    mock_greeks_mesh.read.return_value = {"delta": 0.5, "gamma": 0.02}
    
    response = client.get("/api/v1/options/greeks/AAPL")
    assert response.status_code == 200
    assert response.json()["data"]["delta"] == 0.5

def test_get_realtime_greeks_not_found(mock_options_user, mock_greeks_mesh):
    mock_greeks_mesh.read.return_value = None
    
    response = client.get("/api/v1/options/greeks/INVALID")
    assert response.status_code == 200
    assert response.json()["success"] is False

def test_get_batch_greeks(mock_options_user, mock_greeks_mesh):
    mock_greeks_mesh.read.side_effect = lambda s: {"delta": 0.1} if s == "AAPL" else None
    
    response = client.post("/api/v1/options/greeks/batch", json=["AAPL", "GOOGL"])
    assert response.status_code == 200
    assert "AAPL" in response.json()["data"]
    assert "GOOGL" not in response.json()["data"]

@pytest.mark.asyncio
async def test_get_options_chain_success(mock_options_user, mock_greeks_mesh):
    from src.database.models import OptionPrice
    
    mock_price = MagicMock(spec=OptionPrice)
    mock_price.symbol = "AAPL"
    mock_price.strike = 150.0
    mock_price.expiry = date(2023, 12, 19)
    mock_price.option_type = "call"
    mock_price.bid = 5.0
    mock_price.ask = 5.5
    mock_price.last = 5.25
    mock_price.volume = 100
    mock_price.open_interest = 1000
    mock_price.implied_volatility = 0.2
    # Greeks in DB
    mock_price.delta = 0.45
    mock_price.gamma = 0.01
    mock_price.vega = 0.1
    mock_price.theta = -0.05
    mock_price.rho = 0.01
    mock_price.time = datetime.now(UTC)

    mock_db = AsyncMock()
    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = [mock_price]
    mock_db.execute.return_value = mock_result
    
    from src.database import get_async_db
    app.dependency_overrides[get_async_db] = lambda: mock_db
    
    # Mock SHM enrichment (override DB greeks)
    mock_greeks_mesh.read.return_value = {"delta": 0.50}
    
    response = client.get("/api/v1/options/chain?symbol=AAPL&expiry=week")
    
    assert response.status_code == 200
    data = response.json()["data"]
    assert len(data) == 1
    assert data[0]["delta"] == 0.50  # Enriched from SHM
    assert data[0]["gamma"] == 0.01 # From DB
    
    app.dependency_overrides.clear()
