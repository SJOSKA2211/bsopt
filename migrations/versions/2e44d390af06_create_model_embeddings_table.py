"""create_model_embeddings_table

Revision ID: 2e44d390af06
Revises: 2570edd88eac
Create Date: 2026-01-14 13:17:43.705939

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '2e44d390af06'
down_revision: Union[str, Sequence[str], None] = '2570edd88eac'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


from pgvector.sqlalchemy import Vector


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        'model_embeddings',
        sa.Column('id', sa.Integer(), autoincrement=True, primary_key=True),
        sa.Column('model_id', sa.String(length=64), nullable=False),
        sa.Column('version', sa.Integer(), nullable=False),
        sa.Column('embedding', Vector(1536)),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False)
    )
    # Create HNSW index for L2 distance (vector_l2_ops)
    op.execute("CREATE INDEX ix_model_embeddings_hnsw ON model_embeddings USING hnsw (embedding vector_l2_ops);")


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table('model_embeddings')
