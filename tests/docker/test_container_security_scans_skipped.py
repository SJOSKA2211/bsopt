import pytest


def test_container_security_scans_skipped():
    """
    Container security scans verification is skipped due to user instruction
    not to run or build Docker containers.
    Manual execution of security scans is required to validate security.
    """
    pytest.skip("Container security scans explicitly skipped by user instruction.")
