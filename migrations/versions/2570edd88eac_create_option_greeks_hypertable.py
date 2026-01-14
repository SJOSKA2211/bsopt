"""create_option_greeks_hypertable

Revision ID: 2570edd88eac
Revises: a9484af4f1f0
Create Date: 2026-01-14 13:10:10.710987

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '2570edd88eac'
down_revision: Union[str, Sequence[str], None] = 'a9484af4f1f0'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        'option_greeks',
        sa.Column('time', sa.DateTime(timezone=True), nullable=False),
        sa.Column('contract_id', sa.String(length=64), nullable=False),
        sa.Column('delta', sa.Numeric(precision=10, scale=6)),
        sa.Column('gamma', sa.Numeric(precision=10, scale=6)),
        sa.Column('theta', sa.Numeric(precision=10, scale=6)),
        sa.Column('vega', sa.Numeric(precision=10, scale=6)),
        sa.Column('rho', sa.Numeric(precision=10, scale=6)),
        sa.Column('implied_vol', sa.Numeric(precision=10, scale=6)),
        sa.Column('price', sa.Numeric(precision=18, scale=8)),
        sa.Column('calculation_ms', sa.Float()),
    )
    op.create_index('ix_option_greeks_contract_id_time', 'option_greeks', ['contract_id', sa.text('time DESC')])
    op.execute("SELECT create_hypertable('option_greeks', 'time', chunk_time_interval => INTERVAL '1 day');")


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table('option_greeks')
