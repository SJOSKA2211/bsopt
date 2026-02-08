import os

import pytest


def test_src_directories_exist():
    """Verify that required source directories exist."""
    assert os.path.exists("src/ml")
    assert os.path.exists("src/shared")

def test_tests_directory_exists():
    """Verify that required tests directories exist."""
    assert os.path.exists("tests/ml")

def test_pyproject_toml_exists():
    """Verify that pyproject.toml exists."""
    assert os.path.exists("pyproject.toml")

def test_dependencies_loadable():

    """Verify that core dependencies can be imported."""

    import importlib

    deps = ["minio", "mlflow", "numpy", "optuna", "pandas", "prometheus_client", "structlog", "xgboost"]

    for dep in deps:

        try:

            importlib.import_module(dep)

        except ImportError as e:

            pytest.fail(f"Failed to import core dependency: {dep}. Error: {e}")
