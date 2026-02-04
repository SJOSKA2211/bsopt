import os
import re
import yaml
import time
from pathlib import Path
from typing import Optional, Callable
import structlog
from src.tasks.ml_tasks import train_model_task

logger = structlog.get_logger(__name__)

class MLOpsService:
    def run_pipeline(
        self,
        pipeline_type: str,
        model_repo: Optional[str],
        data_repo: Optional[str],
        deploy_target: str,
        monitor_metrics: str,
        service_name: str,
        docker_image: str,
        model_name: str,
        model_version: str,
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> str:
        """
        Run the MLOps pipeline.
        
        Args:
            pipeline_type: Type of pipeline (ci/cd, manual, scheduled)
            model_repo: URI of the model repository
            data_repo: URI of the data repository
            deploy_target: Target platform (kubernetes, docker, lambda)
            monitor_metrics: Monitoring tool (prometheus, mlflow, grafana)
            service_name: Name of the service
            docker_image: Docker image tag
            model_name: Name of the model
            model_version: Version of the model
            progress_callback: Optional callback for status updates
            
        Returns:
            str: Task ID of the triggered training task
            
        Raises:
            ValueError: If validation fails
        """
        # Validation Logic
        if model_repo:
            s3_uri_pattern = re.compile(r"^s3://[a-zA-Z0-9.-]+(?:/[a-zA-Z0-9-._~/]*)?$")
            if not s3_uri_pattern.match(model_repo):
                 raise ValueError(f"Invalid S3 model repository URI: {model_repo}")

        name_pattern = re.compile(r"^[a-zA-Z0-9.-]+$")
        if not name_pattern.match(service_name):
            raise ValueError(f"Invalid service name: {service_name}")
        if not name_pattern.match(model_name):
             raise ValueError(f"Invalid model name: {model_name}")

        docker_image_pattern = re.compile(r"^[a-zA-Z0-9./-]+(?::[a-zA-Z0-9.-]+)?$")
        if not docker_image_pattern.match(docker_image):
             raise ValueError(f"Invalid docker image: {docker_image}")

        version_pattern = re.compile(r"^(v)?\d+(\.\d+){0,2}$")
        if not version_pattern.match(str(model_version)):
             raise ValueError(f"Invalid model version: {model_version}")

        # Initialization
        if progress_callback: progress_callback("Initializing pipeline...")
        time.sleep(1) # Mock

        # S3 Config
        if model_repo and model_repo.startswith("s3://"):
             if progress_callback: progress_callback(f"Configuring S3 model repo: {model_repo}...")
             os.environ["MLFLOW_ARTIFACT_ROOT"] = model_repo
             time.sleep(1)

        # K8s Generation
        if deploy_target == 'kubernetes':
            if progress_callback: progress_callback("Preparing Kubernetes deployment manifests...")
            self._generate_k8s_manifests(service_name, docker_image, model_repo, model_name, model_version)
            time.sleep(1)

        # Task Trigger
        if progress_callback: progress_callback("Starting training task...")
        task = train_model_task.delay(model_type="xgboost")
        
        logger.info("mlops_pipeline_started", task_id=task.id, service=service_name)
        return task.id

    def _generate_k8s_manifests(self, service_name, docker_image, model_repo, model_name, model_version):
        k8s_path = Path("k8s/base/deployment.yaml")
        if not k8s_path.parent.exists():
            k8s_path.parent.mkdir(parents=True, exist_ok=True)
        
        manifest_dict = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": f"{service_name}-model-serving"
            },
            "spec": {
                "replicas": 1,
                "selector": {
                    "matchLabels": {
                        "app": f"{service_name}-model-serving"
                    }
                },
                "template": {
                    "metadata": {
                        "labels": {
                            "app": f"{service_name}-model-serving"
                        }
                    },
                    "spec": {
                        "containers": [
                            {
                                "name": f"{service_name}-model-serving",
                                "image": docker_image,
                                "ports": [{"containerPort": 8000}],
                                "env": [
                                    {"name": "MODEL_REPO", "value": model_repo},
                                    {"name": "MODEL_NAME", "value": model_name},
                                    {"name": "MODEL_VERSION", "value": str(model_version)}
                                ]
                            }
                        ]
                    }
                }
            }
        }
        manifest = yaml.dump(manifest_dict, sort_keys=False)
        with open(k8s_path, 'w') as f:
            f.write(manifest)
