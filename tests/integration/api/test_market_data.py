import pytest
import httpx
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Dict, Any, List
import time # For timestamping test data
from datetime import datetime, timezone # For date comparisons

# Assuming api_client, db_session, test_user_token, auth_headers fixtures are available from conftest.py
# Import necessary models and schemas
from src.database.models import User, Portfolio, Trade # Import models for potential setup
from src.schemas.portfolio import PortfolioCreate # Needed if creating portfolio for risk metrics test

# Base URL for the API service
API_URL = "http://localhost:8000/api/v1"

pytestmark = pytest.mark.integration

# --- Helper Functions ---
async def create_test_portfolio_for_market_tests(api_client: AsyncClient, auth_headers: Dict[str, str]) -> Dict[str, Any]:
    """Helper to create a portfolio via the API for market data tests."""
    timestamp_suffix = str(int(time.time()))
    portfolio_name = f"MarketDataPortfolio {timestamp_suffix}"
    portfolio_data = {"name": portfolio_name, "cash": 50000.0}
    
    response = await api_client.post("/api/v1/portfolios/", json=portfolio_data, headers=auth_headers)
    response.raise_for_status()
    return response.json()

# --- Tests for Historical Data ---
@pytest.mark.asyncio
async def test_get_historical_data(api_client: AsyncClient, auth_headers: Dict[str, str]):
    """Tests fetching historical market data via API."""
    symbol = "TESTSYM"
    start_date = "2023-01-01"
    end_date = "2023-01-03" # Fetch 3 days of data

    response = await api_client.get(f"/api/v1/market/historical", params={"symbol": symbol, "start_date": start_date, "end_date": end_date}, headers=auth_headers)
    
    assert response.status_code == 200
    historical_data = response.json()
    
    assert isinstance(historical_data, list)
    assert len(historical_data) == 3 # Expecting 3 days of data
    
    # Check structure of the first data point
    first_point = historical_data[0]
    assert "date" in first_point
    assert "open" in first_point and isinstance(first_point["open"], float)
    assert "high" in first_point and isinstance(first_point["high"], float)
    assert "low" in first_point and isinstance(first_point["low"], float)
    assert "close" in first_point and isinstance(first_point["close"], float)
    assert "volume" in first_point and isinstance(first_point["volume"], int)
    
    # Basic date format check for the first entry
    try:
        datetime.strptime(first_point["date"], "%Y-%m-%d")
    except ValueError:
        pytest.fail("Date format is incorrect in historical data")

async def test_get_historical_data_invalid_date(api_client: AsyncClient, auth_headers: Dict[str, str]):
    """Tests fetching historical data with invalid date format."""
    symbol = "TESTSYM"
    start_date = "01-01-2023" # Invalid format
    end_date = "2023-12-31"
    
    response = await api_client.get(f"/api/v1/market/historical", params={"symbol": symbol, "start_date": start_date, "end_date": end_date}, headers=auth_headers)
    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid date format. Use YYYY-MM-DD."

async def test_get_historical_data_no_data(api_client: AsyncClient, auth_headers: Dict[str, str]):
    """Tests fetching historical data for a symbol that returns no data."""
    symbol = "NODATA" # Assume this symbol yields no data in simulation
    start_date = "2024-01-01"
    end_date = "2024-01-02"
    
    response = await api_client.get(f"/api/v1/market/historical", params={"symbol": symbol, "start_date": start_date, "end_date": end_date}, headers=auth_headers)
    assert response.status_code == 200
    assert response.json() == [] # Expect an empty list when no data is found

# --- Tests for Risk Metrics ---
@pytest.mark.asyncio
async def test_get_portfolio_risk_metrics(api_client: AsyncClient, auth_headers: Dict[str, str], db_session: AsyncSession):
    """Tests retrieving risk metrics for a portfolio."""
    # Create a portfolio first
    portfolio = await create_test_portfolio_for_market_tests(api_client, auth_headers)
    portfolio_id = portfolio["id"]
    
    response = await api_client.get(f"/api/v1/market/risk/{portfolio_id}", headers=auth_headers)
    
    assert response.status_code == 200
    risk_metrics = response.json()
    
    assert risk_metrics["portfolio_id"] == portfolio_id
    assert "greeks" in risk_metrics
    assert "delta" in risk_metrics["greeks"] and isinstance(risk_metrics["greeks"]["delta"], float)
    assert "gamma" in risk_metrics["greeks"] and isinstance(risk_metrics["greeks"]["gamma"], float)
    assert "var_99_1_day" in risk_metrics and isinstance(risk_metrics["var_99_1_day"], float)
    assert "timestamp" in risk_metrics
    try:
        datetime.strptime(risk_metrics["timestamp"], "%Y-%m-%dT%H:%M:%S.%fZ") # Check ISO format
    except ValueError:
        pytest.fail("Timestamp format is incorrect")


async def test_get_risk_metrics_portfolio_not_found(api_client: AsyncClient, auth_headers: Dict[str, str]):
    """Tests retrieving risk metrics for a non-existent portfolio."""
    non_existent_id = "non-existent-portfolio-for-risk"
    response = await api_client.get(f"/api/v1/market/risk/{non_existent_id}", headers=auth_headers)
    assert response.status_code == 404
    assert response.json()["detail"] == "Portfolio not found"

# --- Tests for Price Calculation ---
@pytest.mark.asyncio
async def test_calculate_price(api_client: AsyncClient, auth_headers: Dict[str, str]):
    """Tests the price calculation endpoint."""
    calculation_data = {
        "symbol": "MSFT",
        "quantity": 25.0,
        "price": 300.50
    }
    
    response = await api_client.post("/api/v1/market/calculate_price", json=calculation_data, headers=auth_headers)
    
    assert response.status_code == 200
    result = response.json()
    
    assert result["symbol"] == "MSFT"
    assert result["quantity"] == 25.0
    assert result["unit_price"] == 300.50
    assert "total_price" in result
    assert isinstance(result["total_price"], float)
    assert result["total_price"] > 0
    assert "calculation_timestamp" in result

async def test_calculate_price_missing_params(api_client: AsyncClient, auth_headers: Dict[str, str]):
    """Tests price calculation with missing required parameters."""
    calculation_data = {"symbol": "GOOG", "quantity": 5} # Missing price
    response = await api_client.post("/api/v1/market/calculate_price", json=calculation_data, headers=auth_headers)
    assert response.status_code == 400
    assert "Missing required parameters" in response.json()["detail"]

async def test_calculate_price_invalid_params(api_client: AsyncClient, auth_headers: Dict[str, str]):
    """Tests price calculation with invalid parameter types or values."""
    calculation_data = {"symbol": "AMZN", "quantity": -5, "price": 100.0} # Negative quantity
    response = await api_client.post("/api/v1/market/calculate_price", json=calculation_data, headers=auth_headers)
    assert response.status_code == 400
    assert "Quantity must be a positive number" in response.json()["detail"]

# --- Tests for Market Data Ingestion Task Trigger ---
@pytest.mark.asyncio
async def test_trigger_market_data_ingestion(api_client: AsyncClient, auth_headers: Dict[str, str]):
    """Tests triggering market data ingestion task."""
    ingestion_params = {"symbol": "TESTSYM_INGEST", "num_days": 10}
    
    response = await api_client.post("/api/v1/market/ingest_market_data", json=ingestion_params, headers=auth_headers)
    
    assert response.status_code == 202 # Accepted
    task_info = response.json()
    
    assert task_info["message"] == "Market data ingestion task enqueued successfully"
    assert task_info["symbol"] == "TESTSYM_INGEST"
    assert task_info["num_days"] == 10
    assert task_info["status"] == "queued"
    assert "timestamp" in task_info

async def test_trigger_market_data_ingestion_missing_symbol(api_client: AsyncClient, auth_headers: Dict[str, str]):
    """Tests triggering ingestion with missing symbol."""
    ingestion_params = {"num_days": 5} # Missing symbol
    response = await api_client.post("/api/v1/market/ingest_market_data", json=ingestion_params, headers=auth_headers)
    assert response.status_code == 400
    assert response.json()["detail"] == "Symbol is required for market data ingestion"

# --- Tests for Current Market Prices ---
@pytest.mark.asyncio
async def test_get_current_market_prices(api_client: AsyncClient, auth_headers: Dict[str, str]):
    """Tests fetching current market prices for multiple symbols."""
    symbols = ["MSFT", "AMZN", "GOOG"]
    
    response = await api_client.get("/api/v1/market/current_prices", params={"symbols": symbols}, headers=auth_headers)
    
    assert response.status_code == 200
    current_prices = response.json()
    
    assert isinstance(current_prices, dict)
    assert len(current_prices) == len(symbols) # Should return prices for all requested symbols
    
    for symbol in symbols:
        assert symbol in current_prices
        assert isinstance(current_prices[symbol], float)
        assert current_prices[symbol] > 0 # Price should be positive

async def test_get_current_market_prices_no_symbols(api_client: AsyncClient, auth_headers: Dict[str, str]):
    """Tests fetching current prices with no symbols provided."""
    response = await api_client.get("/api/v1/market/current_prices", params={}, headers=auth_headers)
    assert response.status_code == 400
    assert "At least one symbol is required" in response.json()["detail"]

# Note: Additional tests could be added for edge cases, authentication failures, etc.
# These tests assume the backend API endpoints are running and correctly configured.
