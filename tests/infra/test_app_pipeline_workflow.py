import os

import yaml


def test_app_pipeline_workflow_exists():
    """Verify that the app-pipeline workflow file exists."""
    assert os.path.exists(".github/workflows/app-pipeline.yml")


def test_app_pipeline_workflow_contents():
    """Verify that the app-pipeline workflow contains the required quality gates."""
    with open(".github/workflows/app-pipeline.yml") as f:
        # Load without tag transformation to avoid 'on' key issues with booleans
        workflow = yaml.load(f, Loader=yaml.SafeLoader)

    # Check trigger
    on_config = workflow.get("on")
    assert "push" in on_config
    assert "main" in on_config["push"]["branches"]

    jobs = workflow["jobs"]

    # Check Quality Gate job
    assert "quality-gate" in jobs
    qg_steps = str(jobs["quality-gate"]["steps"])
    assert "pylint" in qg_steps.lower()
    assert "mypy" in qg_steps.lower()
    assert "bandit" in qg_steps.lower()
    assert "--fail-under=9.0" in qg_steps  # Pylint threshold

    # Check Tests job
    assert "tests" in jobs
    test_steps = str(jobs["tests"]["steps"])
    assert "pytest" in test_steps.lower()
    assert "--cov-fail-under=85" in test_steps  # Coverage threshold


def test_app_pipeline_matrix_build():
    """Verify that the app-pipeline workflow contains matrix build and push logic."""
    with open(".github/workflows/app-pipeline.yml") as f:
        workflow = yaml.load(f, Loader=yaml.SafeLoader)

    jobs = workflow["jobs"]
    assert "build-push" in jobs

    bp_job = jobs["build-push"]
    # Check matrix
    strategy = bp_job.get("strategy", {})
    matrix = strategy.get("matrix", {})
    services = matrix.get("service", [])
    assert "api" in services
    assert "worker-pricing" in services
    assert "worker-ml" in services
    assert "scraper" in services

    # Check build/push steps
    steps = str(bp_job["steps"])
    assert "docker/setup-buildx-action" in steps.lower()
    assert "docker/login-action" in steps.lower()
    assert "docker/build-push-action" in steps.lower()
    assert "'push': true" in steps.lower() or "'push': True" in steps
    assert "ghcr.io" in steps.lower()
    assert "latest" in steps.lower()


def test_app_pipeline_gitops_update():
    """Verify that the app-pipeline workflow contains GitOps update logic."""
    with open(".github/workflows/app-pipeline.yml") as f:
        workflow = yaml.load(f, Loader=yaml.SafeLoader)

    jobs = workflow["jobs"]
    assert "update-manifests" in jobs

    um_job = jobs["update-manifests"]
    assert "needs" in um_job
    assert "build-push" in um_job["needs"]

    steps = str(um_job["steps"])
    assert "yq" in steps.lower()
    assert "git commit" in steps.lower()
    assert "infrastructure/manifests" in steps.lower()


def test_infrastructure_manifests_exist():
    """Verify that the infrastructure manifests exist."""
    services = ["api", "worker-pricing", "worker-ml", "scraper"]
    for service in services:
        assert os.path.exists(f"infrastructure/manifests/{service}-deployment.yaml")


def test_argocd_canary_manifests_exist():
    """Verify that the ArgoCD and Canary manifests exist."""
    assert os.path.exists("infrastructure/argocd/application.yaml")
    assert os.path.exists("infrastructure/manifests/api-rollout.yaml")
