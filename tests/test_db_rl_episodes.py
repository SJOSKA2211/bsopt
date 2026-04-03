import json
import os

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


def test_rl_episodes_table_exists(db_engine):
    """Test that rl_episodes table exists."""
    with db_engine.connect() as conn:
        result = conn.execute(text("SELECT to_regclass('rl_episodes');"))
        assert result.scalar() is not None, "rl_episodes table does not exist"


def test_insert_and_query_rl_episode(db_engine):
    """Test inserting and retrieving an RL episode record."""
    with db_engine.connect() as conn:
        try:
            conn.execute(text("TRUNCATE rl_episodes;"))
            conn.commit()
        except Exception:
            conn.rollback()

        agent_id = "TD3_Agent_1"
        reward = 1250.75
        steps = 2000
        config = {"learning_rate": 0.0003, "gamma": 0.99, "tau": 0.005}

        conn.execute(
            text(
                """
            INSERT INTO rl_episodes (agent_id, episode_reward, steps, hyperparameters)
            VALUES (:agent_id, :reward, :steps, :config)
        """
            ),
            {
                "agent_id": agent_id,
                "reward": reward,
                "steps": steps,
                "config": json.dumps(config),
            },
        )
        conn.commit()

        # Query
        result = conn.execute(
            text("SELECT * FROM rl_episodes WHERE agent_id = :agent_id"),
            {"agent_id": agent_id},
        )
        row = result.fetchone()
        assert row is not None
        assert row.episode_reward == reward
        assert row.hyperparameters["gamma"] == 0.99
