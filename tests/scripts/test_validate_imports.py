import pytest
import subprocess
import sys
import os
from pathlib import Path
import random
import string

VALIDATE_IMPORTS_SCRIPT = Path(__file__).parent.parent.parent / "scripts" / "validate_imports.py"
os.chmod(VALIDATE_IMPORTS_SCRIPT, 0o755)

def run_validator_script(base_path: Path) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    # Ensure project root (parent of the mock package) is in PYTHONPATH
    env['PYTHONPATH'] = str(base_path.parent) + os.pathsep + env.get('PYTHONPATH', '')
    result = subprocess.run(
        [sys.executable, str(VALIDATE_IMPORTS_SCRIPT), str(base_path)],
        capture_output=True,
        text=True,
        env=env,
        check=False
    )
    return result

class TestImportValidationIntegration:
    def create_mock_package(self, tmp_path):
        # Use a UNIQUE name for each test to avoid sys.modules cache issues
        suffix = ''.join(random.choices(string.ascii_lowercase, k=6))
        pkg_name = f"mock_src_{suffix}"
        pkg_path = tmp_path / pkg_name
        pkg_path.mkdir()
        (pkg_path / "__init__.py").touch()

        utils_dir = pkg_path / "utils"
        utils_dir.mkdir()
        (utils_dir / "__init__.py").touch()
        (utils_dir / "lazy_import.py").write_text(
            "import importlib\n"
            "from unittest.mock import MagicMock\n"
            "class LazyImportError(Exception): pass\n"
            "class CircularImportError(Exception): pass\n"
            "def lazy_import(module_name, import_map, attribute_name, module_obj):\n"
            "    if attribute_name in import_map:\n"
            "        target_module_name = import_map[attribute_name]\n"
            "        if 'circular' in module_name:\n"
            "             raise CircularImportError(f'Simulated circular import for {target_module_name}')\n"
            "        try:\n"
            "            if target_module_name.startswith('.'):\n"
            "                return importlib.import_module(target_module_name, package=module_name)\n"
            "            return importlib.import_module(target_module_name)\n"
            "        except ImportError as e:\n"
            "             raise ImportError(f'Simulated import failure for {target_module_name}') from e\n"
            "    return MagicMock()\n"
            "def preload_modules(*args, **kwargs): pass\n"
            "def reset_import_stats(): pass\n"
            "def get_import_stats(): return {}\n"
        )
        return pkg_name, pkg_path

    def test_all_valid_imports(self, tmp_path):
        pkg_name, pkg_path = self.create_mock_package(tmp_path)
        ml_dir = pkg_path / "ml"
        ml_dir.mkdir()
        (ml_dir / "__init__.py").write_text(
            "import sys\n"
            f"from {pkg_name}.utils.lazy_import import lazy_import\n"
            "_import_map = {'ValidMLClass': '.valid_ml_submodule'}\n"
            "def __getattr__(name): return lazy_import(__name__, _import_map, name, sys.modules[__name__])\n"
        )
        (ml_dir / "valid_ml_submodule.py").write_text("class ValidMLClass: pass")
    
        result = run_validator_script(pkg_path)
        assert result.returncode == 0
        assert "Validation successful: All lazy imports confirmed valid." in result.stdout

    def test_broken_import_map(self, tmp_path):
        pkg_name, pkg_path = self.create_mock_package(tmp_path)
        broken_dir = pkg_path / "broken"
        broken_dir.mkdir()
        (broken_dir / "__init__.py").write_text(
            "import sys\n"
            f"from {pkg_name}.utils.lazy_import import lazy_import\n"
            "_import_map = {'BrokenClass': '.non_existent_submodule'}\n"
            "def __getattr__(name): return lazy_import(__name__, _import_map, name, sys.modules[__name__])\n"
        )
    
        result = run_validator_script(pkg_path)
        assert result.returncode == 1
        assert "Validation failed: Found broken lazy import mappings." in result.stdout
        assert f"{pkg_name}.broken.BrokenClass (maps to .non_existent_submodule): Simulated import failure for .non_existent_submodule" in result.stdout

    def test_circular_import_map(self, tmp_path):
        pkg_name, pkg_path = self.create_mock_package(tmp_path)
        circular_dir = pkg_path / "circular"
        circular_dir.mkdir()
        (circular_dir / "__init__.py").write_text(
            "import sys\n"
            f"from {pkg_name}.utils.lazy_import import lazy_import\n"
            "_import_map = {'CircularClass': '.circular_submodule'}\n"
            "def __getattr__(name): return lazy_import(__name__, _import_map, name, sys.modules[__name__])\n"
        )
    
        result = run_validator_script(pkg_path)
        assert result.returncode == 1
        assert "Validation failed: Found broken lazy import mappings." in result.stdout
        assert f"{pkg_name}.circular.CircularClass (maps to .circular_submodule): Simulated circular import for .circular_submodule" in result.stdout

    def test_no_lazy_loaded_modules_found(self, tmp_path):
        pkg_name, pkg_path = self.create_mock_package(tmp_path)
        no_lazy_dir = pkg_path / "no_lazy"
        no_lazy_dir.mkdir()
        (no_lazy_dir / "__init__.py").write_text("print('No lazy imports here')")
    
        result = run_validator_script(no_lazy_dir)
        assert result.returncode == 0
        assert "No lazy-loaded modules with _import_map and __getattr__ found under the given path." in result.stdout
