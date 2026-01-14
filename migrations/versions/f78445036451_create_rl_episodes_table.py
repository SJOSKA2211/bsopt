"""create_rl_episodes_table

Revision ID: f78445036451
Revises: 2e44d390af06
Create Date: 2026-01-14 13:18:36.353878

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'f78445036451'
down_revision: Union[str, Sequence[str], None] = '2e44d390af06'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


from sqlalchemy.dialects import postgresql


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        'rl_episodes',
        sa.Column('id', sa.Integer(), autoincrement=True, primary_key=True),
        sa.Column('agent_id', sa.String(length=64), nullable=False),
        sa.Column('episode_reward', sa.Float(), nullable=False),
        sa.Column('steps', sa.Integer(), nullable=False),
        sa.Column('hyperparameters', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False)
    )
    op.create_index('ix_rl_episodes_agent_id', 'rl_episodes', ['agent_id'])


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table('rl_episodes')
