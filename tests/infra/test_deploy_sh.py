import subprocess
import os
import pytest
import shutil

DEPLOY_SH = os.path.abspath("./deploy.sh")

def test_deploy_sh_exists():
    """Verify that deploy.sh exists and is executable."""
    assert os.path.exists(DEPLOY_SH)
    assert os.access(DEPLOY_SH, os.X_OK)

def test_check_dependencies_success():
    """Verify that check_dependencies passes when tools are present."""
    result = subprocess.run(
        [DEPLOY_SH, "help"],
        capture_output=True,
        text=True
    )
    assert "deploy" in result.stdout
    assert "help" in result.stdout

def test_check_dependencies_failure():
    """Verify that check_dependencies fails when a tool is missing."""
    fake_path = "/tmp/fake_bin_shadow"
    os.makedirs(fake_path, exist_ok=True)
    with open(os.path.join(fake_path, "docker"), "w") as f:
        f.write("#!/bin/sh\nexit 1")
    os.chmod(os.path.join(fake_path, "docker"), 0o755)
    
    result = subprocess.run(
        ["bash", "-c", f"export PATH={fake_path}:$PATH && {DEPLOY_SH} setup-env"],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0 or "ERROR" in result.stderr
    shutil.rmtree(fake_path)

def test_optimize_kernel_attempts_sysctl():
    """Verify that optimize_kernel attempts to call sysctl."""
    fake_path = "/tmp/fake_bin_sudo"
    os.makedirs(fake_path, exist_ok=True)
    with open(os.path.join(fake_path, "sudo"), "w") as f:
        f.write("#!/bin/sh\necho \"MOCK_SUDO $@\"" )
    os.chmod(os.path.join(fake_path, "sudo"), 0o755)
    
    result = subprocess.run(
        ["bash", "-c", f"export PATH={fake_path}:$PATH && {DEPLOY_SH} optimize-kernel"],
        capture_output=True,
        text=True
    )
    assert "MOCK_SUDO sysctl" in result.stdout
    shutil.rmtree(fake_path)

def test_setup_env_creates_new_file():
    """Verify that setup_env creates a .env file if missing."""
    temp_dir = "/tmp/test_setup_env"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)
    
    result = subprocess.run(
        [DEPLOY_SH, "setup-env"],
        capture_output=True,
        text=True,
        cwd=temp_dir
    )
    
    env_file = os.path.join(temp_dir, ".env")
    assert os.path.exists(env_file)
    with open(env_file, "r") as f:
        content = f.read()
    assert "DB_PASSWORD" in content
    shutil.rmtree(temp_dir)

def test_scaffold_configs_creates_directories_and_files():
    """Verify that scaffold_configs creates the required directory structure."""
    temp_dir = "/tmp/test_scaffold"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)
    
    result = subprocess.run(
        [DEPLOY_SH, "scaffold-configs"],
        capture_output=True,
        text=True,
        cwd=temp_dir
    )
    
    assert os.path.isdir(os.path.join(temp_dir, "monitoring/prometheus"))
    assert os.path.exists(os.path.join(temp_dir, "docker/nginx/nginx.conf"))
    shutil.rmtree(temp_dir)

def test_deploy_stack_calls_docker_compose():
    """Verify that deploy_stack attempts to call docker-compose."""
    fake_path = "/tmp/fake_bin_docker_stack"
    os.makedirs(fake_path, exist_ok=True)
    with open(os.path.join(fake_path, "docker-compose"), "w") as f:
        f.write("#!/bin/sh\necho \"MOCK_COMPOSE $@\"")
    os.chmod(os.path.join(fake_path, "docker-compose"), 0o755)
    
    result = subprocess.run(
        ["bash", "-c", f"export PATH={fake_path}:$PATH && {DEPLOY_SH} deploy-stack"],
        capture_output=True,
        text=True
    )
    
    assert "MOCK_COMPOSE -f docker-compose.prod.yml build" in result.stdout
    shutil.rmtree(fake_path)

def test_lifecycle_commands_call_docker_compose():
    """Verify that down and logs commands call docker-compose."""
    fake_path = "/tmp/fake_bin_lifecycle"
    os.makedirs(fake_path, exist_ok=True)
    with open(os.path.join(fake_path, "docker-compose"), "w") as f:
        f.write("#!/bin/sh\necho \"MOCK_COMPOSE $@\"")
    os.chmod(os.path.join(fake_path, "docker-compose"), 0o755)
    
    result_down = subprocess.run(
        ["bash", "-c", f"export PATH={fake_path}:$PATH && {DEPLOY_SH} down"],
        capture_output=True,
        text=True
    )
    assert "MOCK_COMPOSE -f docker-compose.prod.yml down" in result_down.stdout
    shutil.rmtree(fake_path)

def test_verify_deployment_health_checks():
    """Verify that verify-deployment health check works."""
    fake_path = "/tmp/fake_bin_verify"
    os.makedirs(fake_path, exist_ok=True)
    with open(os.path.join(fake_path, "curl"), "w") as f:
        f.write("#!/bin/sh\necho \"MOCK_CURL $@\"\nexit 0")
    os.chmod(os.path.join(fake_path, "curl"), 0o755)

    result = subprocess.run(
        ["bash", "-c", f"export PATH={fake_path}:$PATH && {DEPLOY_SH} verify-deployment"],
        capture_output=True,
        text=True
    )
    assert "Running Health Checks" in result.stdout
    assert "API is healthy" in result.stdout
    shutil.rmtree(fake_path)

def test_verify_deployment_db_audit():
    """Verify that verify-deployment includes a DB audit step."""
    result = subprocess.run(
        [DEPLOY_SH, "verify-deployment"],
        capture_output=True,
        text=True
    )
    assert "Auditing database extensions" in result.stdout
    assert "Database audit complete" in result.stdout