"""add_compression_and_aggregates

Revision ID: e409083f5e0f
Revises: 9191c1aff456
Create Date: 2026-01-14 13:06:18.778401

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'e409083f5e0f'
down_revision: Union[str, Sequence[str], None] = '9191c1aff456'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Enable compression
    op.execute("ALTER TABLE market_ticks SET (timescaledb.compress, timescaledb.compress_segmentby = 'symbol');")
    op.execute("SELECT add_compression_policy('market_ticks', INTERVAL '1 day');")
    
    # Create Continuous Aggregate
    op.execute("""
        CREATE MATERIALIZED VIEW market_candles_1m
        WITH (timescaledb.continuous) AS
        SELECT
            time_bucket('1 minute', time) AS bucket,
            symbol,
            FIRST(price, time) as open,
            MAX(price) as high,
            MIN(price) as low,
            LAST(price, time) as close,
            SUM(volume) as volume
        FROM market_ticks
        GROUP BY bucket, symbol;
    """)
    
    op.execute("SELECT add_continuous_aggregate_policy('market_candles_1m', start_offset => INTERVAL '3 days', end_offset => INTERVAL '1 minute', schedule_interval => INTERVAL '1 minute');")


def downgrade() -> None:
    """Downgrade schema."""
    op.execute("DROP MATERIALIZED VIEW IF EXISTS market_candles_1m CASCADE;")
    # remove_compression_policy throws error if not exists? if_exists=true supported in newer timescale?
    # We use a safe block or just try.
    op.execute("SELECT remove_compression_policy('market_ticks', if_exists => true);")
    op.execute("ALTER TABLE market_ticks SET (timescaledb.compress = false);")
