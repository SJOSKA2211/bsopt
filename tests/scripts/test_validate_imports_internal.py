import pytest
from unittest.mock import patch, MagicMock
import sys
import os
import importlib.util
from scripts.validate_imports import _find_lazy_loaded_modules, validate_lazy_imports

class TestImportValidationInternal:
    @pytest.fixture
    def mock_structure(self, tmp_path):
        # Point to the REAL src directory to ensure coverage is recorded
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        return os.path.join(project_root, "src")

    def test_find_lazy_loaded_modules(self, mock_structure):
        modules = _find_lazy_loaded_modules(str(mock_structure))
        assert 'src.ml' in modules or 'ml' in modules

    def test_validate_lazy_imports_success(self, mock_structure, mocker): # Added mocker
        # This will actually run against the real src/ but we mock import_module
        # to avoid triggering the heavy loads during the validation loop logic test.
        mock_import = mocker.patch('importlib.import_module') # Changed to mocker.patch and removed 'with' statement
        mock_utils = MagicMock()
        class MockLazyImportError(Exception): pass
        class MockCircularImportError(Exception): pass
        mock_utils.LazyImportError = MockLazyImportError
        mock_utils.CircularImportError = MockCircularImportError
        
        mock_parent = MagicMock()
        # Just return the mock parent for any module requested
        mock_import.side_effect = lambda name, package=None: mock_utils if 'lazy_import' in name else mock_parent # Added package parameter
        
        # validate_lazy_imports will call getattr(parent, attr)
        # Since mock_parent is a MagicMock, getattr(mock_parent, attr) returns another Mock (truthy)
        
        validate_lazy_imports(str(mock_structure))

    def test_validate_lazy_imports_failure(self, mock_structure, mocker): # Added mocker
        mock_import = mocker.patch('importlib.import_module') # Changed to mocker.patch and removed 'with' statement
        mock_utils = MagicMock()
        class MockLazyImportError(Exception): pass
        class MockCircularImportError(Exception): pass
        mock_utils.LazyImportError = MockLazyImportError
        mock_utils.CircularImportError = MockCircularImportError
        
        class FailingModule:
            def __getattr__(self, name):
                raise AttributeError(f"Missing {name}")
        
        mock_parent = FailingModule()
        mock_import.side_effect = lambda name, package=None: mock_utils if 'lazy_import' in name else mock_parent # Added package parameter
        
        with pytest.raises(SystemExit) as cm:
            validate_lazy_imports(str(mock_structure))
        assert cm.value.code == 1

    def test_no_modules_found(self, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        validate_lazy_imports(str(empty_dir))

    def test_module_import_error(self, mock_structure, mocker): # Added mocker
        def failing_side_effect(name, package=None): # Added package parameter
            if 'lazy_import' in name:
                return MagicMock() # Return a mock for lazy_import utils
            raise ImportError("Failed")
            
        mocker.patch('importlib.import_module', side_effect=failing_side_effect) # Changed to mocker.patch and removed 'with' statement
            
        with pytest.raises(SystemExit) as cm:
            validate_lazy_imports(str(mock_structure))
        assert cm.value.code == 1
