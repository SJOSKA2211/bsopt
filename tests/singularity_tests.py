import pytest
import torch
from gymnasium import spaces
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.auth.service import AuthService
from src.config import settings
from src.database.models import Base, OAuth2Client
from src.ml.reinforcement_learning.transformer_policy import (
    TransformerSingularityExtractor,
)

# --- DATABASE & AUTH TESTS ---


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


def test_auth_service_flow(db_session):
    service = AuthService(db_session)

    # Create client
    client = OAuth2Client(
        client_id="test-id",
        client_secret="test-secret",
        scopes=["read"],
        user_id=None,  # For test simplicity
    )
    db_session.add(client)
    db_session.commit()

    # Verify
    verified = service.verify_client("test-id", "test-secret")
    assert verified is not None
    assert verified.client_id == "test-id"

    # Token
    token = service.create_token("test-id", ["read"])
    assert isinstance(token, str)

    # Validate
    payload = service.validate_token(token)
    assert payload["sub"] == "test-id"
    assert "read" in payload["roles"]


# --- ML POLICY TESTS ---


def test_transformer_extractor_forward():
    # Obs space: [window=10, features=5]
    obs_space = spaces.Box(low=-1, high=1, shape=(10, 5))
    extractor = TransformerSingularityExtractor(obs_space, features_dim=128)

    # Batch of 4
    obs = torch.randn(4, 10, 5)
    features = extractor(obs)

    assert features.shape == (4, 128)
    assert not torch.isnan(features).any()


def test_transformer_extractor_unbatched():
    obs_space = spaces.Box(low=-1, high=1, shape=(5,))
    extractor = TransformerSingularityExtractor(obs_space, features_dim=64)

    obs = torch.randn(4, 5)  # Batch of 4, but single step
    features = extractor(obs)

    assert features.shape == (4, 64)


# --- CONFIG TESTS ---


def test_settings_env_override():
    assert settings.PROJECT_NAME == "BSOpt Singularity"
    # Verify transient key generation doesn't crash
    assert "BEGIN RSA PRIVATE KEY" in settings.rsa_private_key
