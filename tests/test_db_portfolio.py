import os
from uuid import uuid4

import pytest
import sqlalchemy
from sqlalchemy import text

# Skip if no DB connection
DATABASE_URL = os.getenv("DATABASE_URL")


@pytest.fixture(scope="module")
def db_engine():
    if not DATABASE_URL:
        pytest.skip("DATABASE_URL not set")
    return sqlalchemy.create_engine(DATABASE_URL)


def test_portfolio_tables_exist(db_engine):
    """Test that users, portfolios, and positions tables exist."""
    with db_engine.connect() as conn:
        for table in ["users", "portfolios", "positions"]:
            result = conn.execute(text(f"SELECT to_regclass('{table}');"))
            assert result.scalar() is not None, f"{table} table does not exist"


def test_portfolio_relationships(db_engine):
    """Test inserting related user, portfolio, and position."""
    with db_engine.connect() as conn:
        try:
            conn.execute(text("TRUNCATE users CASCADE;"))
            conn.commit()
        except Exception:
            conn.rollback()

        # Create User
        user_id = str(uuid4())
        conn.execute(
            text(
                """
            INSERT INTO users (id, email, full_name, role)
            VALUES (:id, :email, 'Test User', 'trader')
        """
            ),
            {"id": user_id, "email": "test@example.com"},
        )

        # Create Portfolio
        portfolio_id = str(uuid4())
        conn.execute(
            text(
                """
            INSERT INTO portfolios (id, user_id, name)
            VALUES (:id, :user_id, 'Main Portfolio')
        """
            ),
            {"id": portfolio_id, "user_id": user_id},
        )

        # Create Position
        position_id = str(uuid4())
        conn.execute(
            text(
                """
            INSERT INTO positions (id, portfolio_id, symbol, quantity, avg_entry_price)
            VALUES (:id, :portfolio_id, 'AAPL', 10, 150.00)
        """
            ),
            {"id": position_id, "portfolio_id": portfolio_id},
        )

        conn.commit()

        # Verify join
        result = conn.execute(
            text(
                """
            SELECT u.email, p.name, pos.symbol 
            FROM users u
            JOIN portfolios p ON u.id = p.user_id
            JOIN positions pos ON p.id = pos.portfolio_id
            WHERE u.id = :user_id
        """
            ),
            {"user_id": user_id},
        )

        row = result.fetchone()
        assert row is not None
        assert row.email == "test@example.com"
        assert row.symbol == "AAPL"


def test_rls_enforcement(db_engine):
    """Test that RLS prevents unauthorized access."""
    with db_engine.connect() as conn:
        # Create two users
        user_a = str(uuid4())
        user_b = str(uuid4())

        try:
            conn.execute(text("TRUNCATE users CASCADE;"))
            conn.commit()
        except Exception:
            conn.rollback()

        conn.execute(
            text(
                """
            INSERT INTO users (id, email, role) VALUES (:id, 'a@test.com', 'trader')
        """
            ),
            {"id": user_a},
        )

        conn.execute(
            text(
                """
            INSERT INTO users (id, email, role) VALUES (:id, 'b@test.com', 'trader')
        """
            ),
            {"id": user_b},
        )

        conn.execute(
            text(
                """
            INSERT INTO portfolios (id, user_id, name) VALUES (:id, :uid, 'A Portfolio')
        """
            ),
            {"id": str(uuid4()), "uid": user_a},
        )

        conn.commit()

        # Simulate User B session
        # Note: In real app we set current_user_id in session variable or role
        # For testing RLS, we need to SET ROLE or SET app.current_user_id
        # We assume the policy uses `current_setting('app.current_user_id')`

        conn.execute(text(f"SET app.current_user_id = '{user_b}';"))

        # User B should NOT see User A's portfolio
        result = conn.execute(text("SELECT * FROM portfolios"))
        rows = result.fetchall()
        assert len(rows) == 0, "RLS failed: User B saw User A's portfolio"

        # Simulate User A
        conn.execute(text(f"SET app.current_user_id = '{user_a}';"))
        result = conn.execute(text("SELECT * FROM portfolios"))
        rows = result.fetchall()
        assert len(rows) == 1, "RLS failed: User A could not see their own portfolio"
