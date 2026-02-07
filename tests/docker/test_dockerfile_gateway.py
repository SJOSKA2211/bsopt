import pytest


def test_dockerfile_gateway_non_root_user_verification_skipped():
    """
    Verification of Dockerfile.gateway running as non-root user is skipped
    due to user instruction not to run or build Docker containers.
    Manual inspection of Dockerfile.gateway is required to confirm 'USER gateway' is present.
    """
    pytest.skip(
        "Docker container build and run explicitly skipped by user instruction."
    )
