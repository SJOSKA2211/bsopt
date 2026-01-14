"""create_option_contracts_table

Revision ID: a9484af4f1f0
Revises: e409083f5e0f
Create Date: 2026-01-14 13:09:40.475657

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a9484af4f1f0'
down_revision: Union[str, Sequence[str], None] = 'e409083f5e0f'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        'option_contracts',
        sa.Column('id', sa.String(length=64), primary_key=True),
        sa.Column('underlying', sa.String(length=20), nullable=False),
        sa.Column('expiry', sa.Date(), nullable=False),
        sa.Column('strike', sa.Numeric(precision=18, scale=2), nullable=False),
        sa.Column('option_type', sa.String(length=4), nullable=False),
        sa.Column('multiplier', sa.Integer(), server_default='100', nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False)
    )
    op.create_index('ix_option_contracts_underlying_expiry_strike', 'option_contracts', ['underlying', 'expiry', 'strike'])


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table('option_contracts')
