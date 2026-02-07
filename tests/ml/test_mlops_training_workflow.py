import os

import yaml


def test_mlops_training_workflow_exists():
    """Verify that the mlops-training workflow file exists."""
    assert os.path.exists(".github/workflows/mlops-training.yml")

def test_mlops_training_workflow_contents():
    """Verify that the mlops-training workflow contains the required triggers and jobs."""
    with open(".github/workflows/mlops-training.yml") as f:
        workflow = yaml.load(f, Loader=yaml.SafeLoader)
    
    # Check triggers
    on_config = workflow.get("on")
    assert "schedule" in on_config
    assert "workflow_dispatch" in on_config
    assert "repository_dispatch" in on_config
    assert "data-drift" in on_config["repository_dispatch"]["types"]
    
    jobs = workflow["jobs"]
    
    # Check Training job
    assert "train" in jobs
    train_job = jobs["train"]
    assert "self-hosted" in str(train_job.get("runs-on"))
    
    steps = str(train_job["steps"])
    assert "mlflow" in steps.lower()
    assert "train" in steps.lower()
