"""create_portfolio_tables

Revision ID: 8de15eacc076
Revises: f78445036451
Create Date: 2026-01-14 13:32:31.599559

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '8de15eacc076'
down_revision: Union[str, Sequence[str], None] = 'f78445036451'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Users Table
    op.create_table(
        'users',
        sa.Column('id', sa.UUID(), server_default=sa.text('uuid_generate_v4()'), primary_key=True),
        sa.Column('email', sa.String(length=255), nullable=False, unique=True),
        sa.Column('full_name', sa.String(length=255), nullable=True),
        sa.Column('role', sa.String(length=50), server_default='trader', nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False)
    )
    
    # Portfolios Table
    op.create_table(
        'portfolios',
        sa.Column('id', sa.UUID(), server_default=sa.text('uuid_generate_v4()'), primary_key=True),
        sa.Column('user_id', sa.UUID(), sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False)
    )
    op.create_index('ix_portfolios_user_id', 'portfolios', ['user_id'])
    
    # Positions Table
    op.create_table(
        'positions',
        sa.Column('id', sa.UUID(), server_default=sa.text('uuid_generate_v4()'), primary_key=True),
        sa.Column('portfolio_id', sa.UUID(), sa.ForeignKey('portfolios.id', ondelete='CASCADE'), nullable=False),
        sa.Column('symbol', sa.String(length=20), nullable=False),
        sa.Column('quantity', sa.Numeric(precision=18, scale=8), nullable=False),
        sa.Column('avg_entry_price', sa.Numeric(precision=18, scale=8), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now(), nullable=False)
    )
    op.create_index('ix_positions_portfolio_id', 'positions', ['portfolio_id'])
    op.create_index('ix_positions_symbol', 'positions', ['symbol'])


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table('positions')
    op.drop_table('portfolios')
    op.drop_table('users')
