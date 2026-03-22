import pytest


def test_dockerfile_ml_non_root_user_verification_skipped():
    """
    Verification of Dockerfile.ml running as non-root user is skipped
    due to user instruction not to run or build Docker containers.
    Manual inspection of Dockerfile.ml is required to confirm 'USER algo_trader' is present.
    """
    pytest.skip("Docker container build and run explicitly skipped by user instruction.")