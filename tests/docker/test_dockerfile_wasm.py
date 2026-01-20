import pytest

def test_dockerfile_wasm_non_root_user_verification_skipped():
    """
    Verification of Dockerfile.wasm running as non-root user is skipped
    due to user instruction not to run or build Docker containers.
    Manual inspection of Dockerfile.wasm is required to confirm 'USER' directive.
    """
    pytest.skip("Docker container build and run explicitly skipped by user instruction.")
