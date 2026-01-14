"""enable_rls_policies

Revision ID: 45e3b58efffa
Revises: 8de15eacc076
Create Date: 2026-01-14 13:33:30.082560

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '45e3b58efffa'
down_revision: Union[str, Sequence[str], None] = '8de15eacc076'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Enable RLS
    op.execute("ALTER TABLE portfolios ENABLE ROW LEVEL SECURITY;")
    op.execute("ALTER TABLE positions ENABLE ROW LEVEL SECURITY;")
    
    # Create Policies
    # Portfolios: Users can only see their own portfolios
    op.execute("""
        CREATE POLICY portfolio_isolation_policy ON portfolios
        FOR ALL
        USING (user_id = current_setting('app.current_user_id', true)::uuid);
    """)
    
    # Positions: Users can only see positions in their own portfolios
    # This requires a join or subquery. Performance wise, we often denormalize user_id to positions or use EXISTS.
    op.execute("""
        CREATE POLICY position_isolation_policy ON positions
        FOR ALL
        USING (
            portfolio_id IN (
                SELECT id FROM portfolios 
                WHERE user_id = current_setting('app.current_user_id', true)::uuid
            )
        );
    """)


def downgrade() -> None:
    """Downgrade schema."""
    op.execute("DROP POLICY IF EXISTS portfolio_isolation_policy ON portfolios;")
    op.execute("DROP POLICY IF EXISTS position_isolation_policy ON positions;")
    op.execute("ALTER TABLE portfolios DISABLE ROW LEVEL SECURITY;")
    op.execute("ALTER TABLE positions DISABLE ROW LEVEL SECURITY;")
