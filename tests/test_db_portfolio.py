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
        unique_email = f"test_{uuid4()}@example.com"
        conn.execute(
            text(
                """
            INSERT INTO users (id, email, full_name, tier, hashed_password)
            VALUES (:id, :email, 'Test User', 'pro', 'hashed_pass_placeholder')
        """
            ),
            {"id": user_id, "email": unique_email},
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
            INSERT INTO positions (id, portfolio_id, symbol, quantity, entry_price)
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
        assert row.email == unique_email
        assert row.symbol == "AAPL"


def test_rls_enforcement(db_engine):
    """Test that RLS prevents unauthorized access."""
    with db_engine.connect() as conn:
        try:
            conn.execute(text("TRUNCATE users CASCADE;"))
            conn.commit()
        except Exception:
            conn.rollback()

        # Create two users
        user_a = str(uuid4())
        user_b = str(uuid4())

        # Reset session context and set to a dummy admin or similar if needed
        # Actually, for admin to insert a specific user ID, they need to BE that user or bypass RLS.
        # Since we FORCE RLS, we must BE the user we are inserting for that specific row.
        
        # 1. Setup two users and portfolios
        user_a = str(uuid4())
        user_b = str(uuid4())
        user_a_portfolio = str(uuid4())
        user_b_portfolio = str(uuid4())

        email_a = f"a_{uuid4()}@test.com"
        email_b = f"b_{uuid4()}@test.com"

        # To insert user_a, we must set context to user_a
        conn.execute(text(f"SET app.current_user_id = '{user_a}'"))
        conn.execute(
            text(
                """
            INSERT INTO users (id, email, tier, hashed_password) VALUES (:id, :email, 'pro', 'h1')
        """
            ),
            {"id": user_a, "email": email_a},
        )

        # To insert user_b, we must set context to user_b
        conn.execute(text(f"SET app.current_user_id = '{user_b}'"))
        conn.execute(
            text(
                """
            INSERT INTO users (id, email, tier, hashed_password) VALUES (:id, :email, 'pro', 'h2')
        """
            ),
            {"id": user_b, "email": email_b},
        )

        # To insert portfolios, we must follow similar logic
        conn.execute(text(f"SET app.current_user_id = '{user_a}'"))
        conn.execute(
            text(
                """
            INSERT INTO portfolios (id, user_id, name) VALUES (:id, :uid, 'A Portfolio')
        """
            ),
            {"id": user_a_portfolio, "uid": user_a},
        )

        conn.execute(text(f"SET app.current_user_id = '{user_b}'"))
        conn.execute(
            text(
                """
            INSERT INTO portfolios (id, user_id, name) VALUES (:id, :uid, 'B Portfolio')
        """
            ),
            {"id": user_b_portfolio, "uid": user_b},
        )

        conn.commit()

        # Reset session context
        conn.execute(text("SET app.current_user_id = ''"))

        # Use a raw connection/cursor for RLS because SQLAlchemy session management 
        # can conflict with manual transaction control for RLS variables.
        raw_conn = conn.connection.driver_connection
        
        # Prepare a non-superuser role for testing RLS
        # We use 'app_user' which we already created and granted permissions to.

        # Reset session context
        conn.execute(text("SET app.current_user_id = ''"))
        conn.execute(text("RESET ROLE"))

        # Simulate User B session
        with raw_conn.cursor() as cur:
            cur.execute("SET ROLE app_user;")
            cur.execute(f"SET app.current_user_id = '{user_b}';")

            # Verify RLS on portfolios
            cur.execute("SELECT id FROM portfolios;")
            rows = cur.fetchall()
            print(f"\nDEBUG User B (app_user) saw portfolios: {[str(r[0]) for r in rows]}")
            for r in rows:
                assert str(r[0]) != user_a_portfolio, "RLS failure: User B saw User A's data"
            assert len(rows) == 1, f"User B should see 1 portfolio, saw {len(rows)}"
            cur.execute("RESET ROLE;")

        # Simulate User A
        with raw_conn.cursor() as cur:
            cur.execute("SET ROLE app_user;")
            cur.execute(f"SET app.current_user_id = '{user_a}';")
            cur.execute("SELECT id FROM portfolios;")
            rows = cur.fetchall()
            print(f"DEBUG User A (app_user) saw portfolios: {[str(r[0]) for r in rows]}")
            assert len(rows) == 1, "User A should see 1 portfolio"
            assert str(rows[0][0]) == user_a_portfolio
            cur.execute("RESET ROLE;")

