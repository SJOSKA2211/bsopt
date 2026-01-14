"""enable_extensions

Revision ID: a3074b6bde4c
Revises: 
Create Date: 2026-01-14 12:56:25.890728

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a3074b6bde4c'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.execute("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;")
    op.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    op.execute("CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\";")


def downgrade() -> None:
    """Downgrade schema."""
    op.execute("DROP EXTENSION IF EXISTS \"uuid-ossp\";")
    op.execute("DROP EXTENSION IF EXISTS vector;")
    # Note: Dropping timescaledb might be restricted or require special handling
    op.execute("DROP EXTENSION IF EXISTS timescaledb CASCADE;")
