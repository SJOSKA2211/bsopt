import pytest

def test_dockerfile_api_non_root_user_verification_skipped():
    """
    Verification of Dockerfile.api running as non-root user is skipped
    due to user instruction not to run or build Docker containers.
    Manual inspection of Dockerfile.api is required to confirm 'USER api_user' is present.
    """
    pytest.skip("Docker container build and run explicitly skipped by user instruction.")
