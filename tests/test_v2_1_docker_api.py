import subprocess
import pytest

def test_api_container_user():
    """Verify that the API container runs as a non-root user."""
    # This test requires docker-compose to be running or at least the image to be built.
    # We can check the image metadata or run a command in a temporary container.
    try:
        # Build the image first
        subprocess.run(["docker", "compose", "build", "api"], check=True, capture_output=True)
        
        # Run id command in the container
        result = subprocess.run(
            ["docker", "compose", "run", "--rm", "api", "id", "-u"],
            capture_output=True,
            text=True,
            check=True
        )
        uid = result.stdout.strip()
        assert uid != "0", f"Container is running as root (uid={uid})"
        assert uid == "1000", f"Container is not running as the expected appuser (uid={uid})"
        
        # Check username
        result = subprocess.run(
            ["docker", "compose", "run", "--rm", "api", "whoami"],
            capture_output=True,
            text=True,
            check=True
        )
        username = result.stdout.strip()
        assert username == "appuser", f"Container username is {username}, expected appuser"
        
    except subprocess.CalledProcessError as e:
        pytest.fail(f"Docker command failed: {e.stderr}")
    except FileNotFoundError:
        pytest.skip("Docker or docker-compose not found")

def test_api_container_read_only_fs():
    """Verify that the API container has some level of protection or correct permissions."""
    # We can check if we can write to root
    try:
        result = subprocess.run(
            ["docker", "compose", "run", "--rm", "api", "touch", "/test_root_write"],
            capture_output=True,
            text=True
        )
        # Should fail if non-root and permissions are correct
        assert result.returncode != 0, "Should not be able to write to root as non-root user"
        
    except subprocess.CalledProcessError as e:
        pytest.fail(f"Docker command failed: {e.stderr}")
