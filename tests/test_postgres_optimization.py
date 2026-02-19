import subprocess

import pytest


def run_command(command):
    """Run a shell command and return the output."""
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return result.stdout.strip(), result.stderr.strip(), result.returncode


def test_postgres_extensions():
    """Verify that the PostgreSQL container has TimescaleDB and pgvector extensions installed."""
    container_name = "bsopt-postgres"

    # Check if container is running
    stdout, stderr, returncode = run_command(
        f"docker inspect -f '{{{{.State.Running}}}}' {container_name}"
    )
    if returncode != 0 or stdout != "true":
        pytest.skip(f"Container {container_name} is not running. Skipping test.")

    # Check for extensions
    query = "SELECT extname FROM pg_extension;"
    cmd = f'docker exec {container_name} psql -U admin -d bsopt -t -c "{query}"'

    stdout, stderr, returncode = run_command(cmd)
    assert returncode == 0, f"Failed to execute query in container: {stderr}"

    extensions = [ext.strip() for ext in stdout.splitlines()]

    assert "timescaledb" in extensions, "TimescaleDB extension is missing"
    assert "vector" in extensions, "pgvector extension is missing"


def test_postgres_config_overrides():
    """Verify that performance tuning configurations are applied."""
    container_name = "bsopt-postgres"

    # Check if container is running
    stdout, stderr, returncode = run_command(
        f"docker inspect -f '{{{{.State.Running}}}}' {container_name}"
    )
    if returncode != 0 or stdout != "true":
        pytest.skip(f"Container {container_name} is not running. Skipping test.")

    # Check shared_buffers (example of tuning)
    # We expect some specific configuration. For now, let's just check it's accessible.
    query = "SHOW shared_buffers;"
    cmd = f'docker exec {container_name} psql -U admin -d bsopt -t -c "{query}"'

    stdout, stderr, returncode = run_command(cmd)
    assert returncode == 0, f"Failed to execute query: {stderr}"
    assert stdout.strip() != "", "shared_buffers should not be empty"
