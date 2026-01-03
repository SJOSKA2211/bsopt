import os
import tomllib
from pathlib import Path

def test_project_structure():
    """Verify that the core project directories exist."""
    root = Path(__file__).parent.parent
    
    expected_dirs = [
        root / "src" / "ml",
        root / "src" / "shared",
        root / "tests" / "ml",
    ]
    
    for directory in expected_dirs:
        assert directory.exists(), f"Directory {directory} should exist"
        assert directory.is_dir(), f"{directory} should be a directory"
        # Verify __init__.py exists to make it a package
        assert (directory.parent / "__init__.py").exists() or (directory / "__init__.py").exists(), \
            f"Directory {directory} or its parent should have an __init__.py"

def test_pyproject_toml_exists_and_valid():
    """Verify pyproject.toml exists and is valid TOML."""
    root = Path(__file__).parent.parent
    pyproject_path = root / "pyproject.toml"
    
    assert pyproject_path.exists(), "pyproject.toml should exist"
    
    with open(pyproject_path, "rb") as f:
        data = tomllib.load(f)
        
    assert "project" in data, "pyproject.toml should have a [project] section"
    assert "name" in data["project"], "pyproject.toml should define the project name"
    assert data["project"]["name"] == "bsopt", "Project name should be bsopt"
