import os

import pytest


def test_dockerfile_ci_exists():
    """Verify that the CI runner Dockerfile exists."""
    assert os.path.exists("docker/Dockerfile.ci")


def test_dockerfile_ci_contents():
    """Verify that the CI runner Dockerfile contains necessary tools."""
    with open("docker/Dockerfile.ci") as f:
        content = f.read()

    # Check base image
    assert "FROM python:3.10-slim" in content

    # Check system dependencies
    required_sys_deps = ["git", "curl", "jq", "build-essential", "docker-ce-cli"]
    for dep in required_sys_deps:
        assert dep in content

    # Check Python tools
    required_py_tools = ["pylint", "mypy", "pytest", "bandit", "black"]
    for tool in required_py_tools:
        assert tool in content


def test_dockerfile_ci_non_root_verification_skipped():
    """
    Verification of Dockerfile.ci running as non-root user is skipped
    due to user instruction not to run or build Docker containers.
    """
    pytest.skip(
        "Docker container build and run explicitly skipped by user instruction."
    )


def test_dockerfile_scraper_hardened():
    """Verify that Dockerfile.scraper uses a non-root user."""
    assert os.path.exists("docker/Dockerfile.scraper")
    with open("docker/Dockerfile.scraper") as f:
        content = f.read()

    assert "USER " in content
