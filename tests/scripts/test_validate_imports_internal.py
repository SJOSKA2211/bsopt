import pytest
from unittest.mock import patch, MagicMock
import sys
import os
import importlib
import scripts.validate_imports

class TestImportValidationInternal:
    @pytest.fixture(autouse=True)
    def cleanup_lazy_import(self):
        yield
        if 'src.utils.lazy_import' in sys.modules:
            del sys.modules['src.utils.lazy_import']

    @pytest.fixture
    def mock_structure(self, tmp_path):
        # Point to the REAL src directory to ensure coverage is recorded
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        return os.path.join(project_root, "src")

    def test_find_lazy_loaded_modules(self, mock_structure):
        modules = scripts.validate_imports._find_lazy_loaded_modules(str(mock_structure))
        assert isinstance(modules, dict)

    def test_validate_lazy_imports_success(self, mock_structure, mocker):
        fake_mod_name = "fake_success_module"
        fake_mod = MagicMock()
        fake_mod.some_attr = "exists"
        
        # Patch importlib.import_module which is used by lazy_import
        mocker.patch('importlib.import_module', return_value=fake_mod)
        mocker.patch.object(scripts.validate_imports, '_find_lazy_loaded_modules', 
                            return_value={fake_mod_name: {'some_attr': 'some/path'}})
        
        scripts.validate_imports.validate_lazy_imports(str(mock_structure))

    def test_validate_lazy_imports_failure(self, mock_structure, mocker):
        fake_mod_name = "fake_attr_error_module"
        
        class EmptyModule:
            pass
        
        fake_mod = EmptyModule()
        # Mock import_module to return a module that will raise AttributeError on access
        mocker.patch('importlib.import_module', return_value=fake_mod)
        mocker.patch.object(scripts.validate_imports, '_find_lazy_loaded_modules', 
                            return_value={fake_mod_name: {'missing_attr': 'some/path'}})
        
        with pytest.raises(SystemExit) as cm:
            scripts.validate_imports.validate_lazy_imports(str(mock_structure))
        assert cm.value.code == 1

    def test_no_modules_found(self, tmp_path, mocker):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        mocker.patch.object(scripts.validate_imports, '_find_lazy_loaded_modules', return_value={})
        scripts.validate_imports.validate_lazy_imports(str(empty_dir))

    def test_module_import_error(self, mock_structure, mocker):
        fake_mod_name = "non_existent_module_xyz"
        
        mocker.patch('importlib.import_module', side_effect=ImportError("Failed"))
        mocker.patch.object(scripts.validate_imports, '_find_lazy_loaded_modules', 
                            return_value={fake_mod_name: {'attr': 'path'}})
            
        with pytest.raises(SystemExit) as cm:
            scripts.validate_imports.validate_lazy_imports(str(mock_structure))
        assert cm.value.code == 1