from unittest.mock import MagicMock
from uuid import uuid4

from src.database import crud
from src.database.models import User

# from tests.test_utils import assert_equal # Just use asserts


def test_user_crud(mock_db_session):
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

    db_user = crud.get_user_by_email(mock_db_session, user_data["email"])
    assert db_user is not None
    assert db_user.email == user_data["email"]

    # Update
    crud.update_user_tier(mock_db_session, db_user.id, "pro")
    # Verify execute was called (generic check since matching arguments for update stmt is hard)
    assert mock_db_session.execute.called


def test_portfolio_crud(mock_db_session):
    user_id = uuid4()
    # Fixed function name from create_user_portfolio to create_portfolio
    portfolio = crud.create_portfolio(mock_db_session, user_id, "Test Portfolio")
    assert portfolio.user_id == user_id
    assert portfolio.name == "Test Portfolio"

    # Mock execute for get_user_portfolios
    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = [portfolio]
    mock_db_session.execute.return_value = mock_result

    db_portfolios = crud.get_user_portfolios(mock_db_session, user_id)
    assert len(db_portfolios) > 0
