import importlib
import os

import pytest


def test_src_directories_exist():
    """Verify that core source directories exist."""
    assert os.path.isdir("src/ml")
    assert os.path.isdir("src/shared")

def test_tests_directory_exists():
    """Verify that the ml tests directory exists."""
    assert os.path.isdir("tests/ml")

@pytest.mark.parametrize("module_name", [
    "numpy",
    "pandas",
    "xgboost",
    "optuna",
    "mlflow",
    "prometheus_client",
    "structlog",
    "sklearn"
])
def test_dependencies_installed(module_name):
    """Verify that core dependencies are loadable."""
    try:
        importlib.import_module(module_name)
    except ImportError:
        pytest.fail(f"Dependency {module_name} is not installed.")
