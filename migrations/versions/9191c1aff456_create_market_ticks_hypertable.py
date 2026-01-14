"""create_market_ticks_hypertable

Revision ID: 9191c1aff456
Revises: a3074b6bde4c
Create Date: 2026-01-14 13:03:37.694356

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '9191c1aff456'
down_revision: Union[str, Sequence[str], None] = 'a3074b6bde4c'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        'market_ticks',
        sa.Column('time', sa.DateTime(timezone=True), nullable=False),
        sa.Column('symbol', sa.String(length=20), nullable=False),
        sa.Column('price', sa.Numeric(precision=18, scale=8), nullable=False),
        sa.Column('volume', sa.Numeric(precision=18, scale=8), nullable=False),
    )
    op.create_index('ix_market_ticks_symbol_time', 'market_ticks', ['symbol', sa.text('time DESC')])
    op.execute("SELECT create_hypertable('market_ticks', 'time', chunk_time_interval => INTERVAL '1 day');")


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table('market_ticks')
