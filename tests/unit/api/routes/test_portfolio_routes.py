from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from api.index import app

client = TestClient(app)

@pytest.fixture
def mock_portfolio_user():
    from src.auth.auth import get_current_active_user
    from src.database.models import User
    mock_user = MagicMock(spec=User)
    mock_user.id = uuid4()
    mock_user.email = "test@example.com"
    app.dependency_overrides[get_current_active_user] = lambda: mock_user
    yield mock_user
    app.dependency_overrides.clear()

@pytest.fixture
def mock_db():
    from src.database import get_async_db
    mock_db = AsyncMock()
    app.dependency_overrides[get_async_db] = lambda: mock_db
    yield mock_db
    app.dependency_overrides.clear()

@pytest.mark.asyncio
async def test_get_portfolio_success(mock_portfolio_user, mock_db):
    from src.database.models import Portfolio, Position
    
    mock_pos = MagicMock(spec=Position)
    mock_pos.id = uuid4()
    mock_pos.symbol = "AAPL"
    mock_pos.quantity = 10
    mock_pos.entry_price = 150.0
    mock_pos.status = "open"
    
    mock_port = MagicMock(spec=Portfolio)
    mock_port.id = uuid4()
    mock_port.name = "Main"
    mock_port.cash_balance = 1000.0
    mock_port.positions = [mock_pos]
    
    # Mock db.execute for Portfolio lookup
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = mock_port
    mock_db.execute.return_value = mock_result
    
    with patch("src.database.crud.get_portfolio_total_value", new_callable=AsyncMock) as mock_val:
        mock_val.return_value = 2500.0
        
        response = client.get("/api/v1/portfolio")
        
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Main"
        assert data["total_value"] == 2500.0
        assert len(data["positions"]) == 1

@pytest.mark.asyncio
async def test_get_portfolio_empty(mock_portfolio_user, mock_db):
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = None
    mock_db.execute.return_value = mock_result
    
    response = client.get("/api/v1/portfolio")
    assert response.status_code == 200
    assert response.json()["message"] == "No portfolio found for user"

@pytest.mark.asyncio
async def test_get_portfolio_summary(mock_portfolio_user, mock_db):
    mock_result = MagicMock()
    mock_row = MagicMock()
    mock_row._mapping = {"total_positions": 5, "cash_balance": 5000.0}
    mock_result.fetchone.return_value = mock_row
    mock_db.execute.return_value = mock_result
    
    response = client.get("/api/v1/portfolio/summary")
    assert response.status_code == 200
    assert response.json()["data"]["total_positions"] == 5

@pytest.mark.asyncio
async def test_add_position_success(mock_portfolio_user, mock_db):
    from src.database.models import Portfolio
    mock_port = MagicMock(spec=Portfolio)
    mock_port.id = uuid4()
    
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = mock_port
    mock_db.execute.return_value = mock_result
    
    payload = {"symbol": "TSLA", "quantity": 5, "entry_price": 200.0}
    response = client.post("/api/v1/portfolio/positions", json=payload)
    
    assert response.status_code == 201
    assert "id" in response.json()["data"]
    mock_db.add.assert_called_once()
    mock_db.commit.assert_called_once()

@pytest.mark.asyncio
async def test_delete_position_success(mock_portfolio_user, mock_db):
    from src.database.models import Position
    mock_pos = MagicMock(spec=Position)
    mock_pos.id = uuid4()
    
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = mock_pos
    mock_db.execute.return_value = mock_result
    
    response = client.delete(f"/api/v1/portfolio/positions/{mock_pos.id}")
    assert response.status_code == 200
    mock_db.delete.assert_called_once()
    mock_db.commit.assert_called_once()

@pytest.mark.asyncio
async def test_delete_position_not_found(mock_portfolio_user, mock_db):
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = None
    mock_db.execute.return_value = mock_result
    
    response = client.delete(f"/api/v1/portfolio/positions/{uuid4()}")
    assert response.status_code == 404
