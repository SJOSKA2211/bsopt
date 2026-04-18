import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from api.index import app  # Assuming api.index contains the FastAPI app
from src.database.models import Portfolio, Trade  # Import models
from src.database.session import get_async_db  # Import the real DB session provider

# Assume base and settings are correctly configured and imported
# from src.core.config import settings # Use if settings are defined elsewhere

# Base URL for the API service (should be dynamically set or configured)
API_URL = "http://localhost:8000" # This should ideally come from env vars or conftest setup

# --- Database Setup for Integration Tests ---
# Use a separate test database URL, or ensure proper cleanup if using the same DB
# The existing conftest.py already sets DATABASE_URL for the test environment.
# We will use that, assuming it points to a test-specific database or a shared one with cleanup.
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://admin:password@localhost:5432/bsopt_test")
engine = create_async_engine(DATABASE_URL)
TestingSessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
    class_=AsyncSession,
)

# Override the dependency in the app to use our test session
async def override_get_async_db():
    async with TestingSessionLocal() as session:
        yield session

app.dependency_overrides[get_async_db] = override_get_async_db

@pytest_asyncio.fixture(scope="module")
async def init_db():
    """Initializes the database for integration tests."""
    async with engine.begin() as conn:
        # Create tables if they don't exist (for clean test env setup)
        # In a real scenario, migrations would handle this. For tests, we might drop/create.
        await conn.run_sync(SQLModel.metadata.create_all)
    yield
    # Optional: Clean up tables after tests if needed, or rely on TRUNCATE in api_client fixture

@pytest.fixture(scope="module")
def api_client():
    """Returns a FastAPI TestClient targeting the real app with NO mocks."""
    # The api_client fixture in conftest.py already provides TestClient and handles truncation.
    # This is redundant if conftest.py is correctly imported and used.
    # We will rely on the conftest.py fixture.

@pytest.fixture
async def create_test_portfolio(db_session: AsyncSession) -> Portfolio:
    """Creates a test portfolio in the database."""
    portfolio_data = {
        "name": "Test Integration Portfolio",
        "cash": 10000.0,
        "user_id": "test-user-for-portfolio", # Assume user exists or create one
    }
    # Assume User model and creation logic exists or is handled elsewhere
    # For simplicity, let's assume user exists or is mocked/created by another fixture
    # If not, a User creation step would be needed here.
    new_portfolio = Portfolio(**portfolio_data)
    db_session.add(new_portfolio)
    await db_session.commit()
    await db_session.refresh(new_portfolio)
    return new_portfolio

@pytest.fixture
async def create_test_trade(db_session: AsyncSession, create_test_portfolio: Portfolio) -> Trade:
    """Creates a test trade linked to a portfolio."""
    trade_data = {
        "portfolio_id": create_test_portfolio.id,
        "symbol": "TESTSYM",
        "quantity": 10,
        "price": 150.50,
        "side": "buy", # or "sell"
        "order_type": "market",
        "status": "filled",
    }
    new_trade = Trade(**trade_data)
    db_session.add(new_trade)
    await db_session.commit()
    await db_session.refresh(new_trade)
    return new_trade

# Placeholder for ML Model integration test - requires actual ML logic implementation
# @pytest.mark.asyncio
# async def test_ml_model_integration(db_session: AsyncSession, api_client: AsyncClient):
#     """
#     Tests integration with ML models. Requires actual ML model implementation.
#     """
#     # Example: Create an MLModel entry in DB and test API endpoint that uses it
#     ml_model_data = {
#         "name": "TestMLModel",
#         "version": "1.0.0",
#         "description": "A test ML model for integration testing."
#     }
#     new_ml_model = MLModel(**ml_model_data)
#     db_session.add(new_ml_model)
#     await db_session.commit()
#     await db_session.refresh(new_ml_model)

#     # Now test an API endpoint that might use this model
#     # response = await api_client.get(f"/api/v1/ml/predict/{new_ml_model.id}")
#     # assert response.status_code == 200
#     # assert response.json()["prediction"] is not None # Expecting a prediction
#     pass # Remove pass when implemented

