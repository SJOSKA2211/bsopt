from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from src.database import crud
from src.database.models import User


@pytest.mark.asyncio
async def test_user_crud(mock_db_session):
    # Setup mock_db_session for async operations
    mock_db_session.execute = AsyncMock()
    mock_db_session.commit = AsyncMock()
    mock_db_session.refresh = AsyncMock()

    # Create
    user_data = {
        "email": "crud_test@example.com",
        "hashed_password": "hashed",
        "full_name": "CRUD Test",
    }
    user = User(id=uuid4(), **user_data)
    mock_db_session.add(user)

    # Read
    # Mock the execute result for get_user_by_email
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = user
    mock_db_session.execute.return_value = mock_result

    db_user = await crud.get_user_by_email(mock_db_session, user_data["email"])
    assert db_user is not None
    assert db_user.email == user_data["email"]

    # Update
    await crud.update_user_tier(mock_db_session, db_user.id, "pro")
    assert mock_db_session.execute.called


@pytest.mark.asyncio
async def test_portfolio_crud(mock_db_session):
    mock_db_session.execute = AsyncMock()
    mock_db_session.commit = AsyncMock()
    mock_db_session.refresh = AsyncMock()

    user_id = uuid4()
    # Mock the return value of create_portfolio (it doesn't actually call execute in crud.py, it calls add & commit)
    # Actually create_portfolio calls db.add(portfolio), await db.commit(), await db.refresh(portfolio)

    portfolio = await crud.create_portfolio(mock_db_session, user_id, "Test Portfolio")
    assert portfolio.user_id == user_id
    assert portfolio.name == "Test Portfolio"

    # Mock execute for get_user_portfolios
    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = [portfolio]
    mock_db_session.execute.return_value = mock_result

    db_portfolios = await crud.get_user_portfolios(mock_db_session, user_id)
    assert len(db_portfolios) > 0
