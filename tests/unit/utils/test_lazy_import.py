import sys
from unittest.mock import MagicMock, patch

import pytest

from src.utils.lazy_import import (
    CircularImportError,
    LazyImportError,
    get_import_stats,
    lazy_import,
    preload_modules,
    reset_import_stats,
)


class MockModule:
    def __init__(self):
        self.__dict__ = {}

def test_lazy_import_success():
    reset_import_stats()
    MockModule()
    {"test_attr": "os.path"}
    
    with patch("importlib.import_module") as mock_import:
        mock_mod = MagicMock()
        mock_mod.test_attr = "val"
        mock_import.return_value = mock_mod
        
        res = lazy_import("src", import_map, "test_attr", cache_module)
        assert res == "val"
        assert cache_module.test_attr == "val"
        
        stats = get_import_stats()
        assert stats['successful_imports'] == 1

def test_lazy_import_attribute_error():
    MockModule()
    {"a": "b"}
    with pytest.raises(AttributeError):
        lazy_import("src", import_map, "wrong", cache_module)

def test_lazy_import_circular():
    reset_import_stats()
    MockModule()
    {"circular": "src.circular"}
    
    # We need to manually trigger the circular detection by nested calls if we want to test the logic
    from src.utils.lazy_import import _track_import_stack
    
    with _track_import_stack("src.circular"):
        with pytest.raises(CircularImportError):
            with _track_import_stack("src.circular"):
                pass

def test_lazy_import_failure_caching():
    reset_import_stats()
    MockModule()
    {"fail": "nonexistent"}
    
    with patch("importlib.import_module", side_effect=ImportError("No module")):
        with pytest.raises(LazyImportError):
            lazy_import("src", import_map, "fail", cache_module)
            
        # Second attempt should raise LazyImportError from cache
        with pytest.raises(LazyImportError, match="Previous import"):
            lazy_import("src", import_map, "fail", cache_module)

def test_lazy_import_circular_reraise():
    reset_import_stats()
    MockModule()
    {"circular": "src.circular"}
    
    with patch("src.utils.lazy_import._track_import_stack") as mock_track:
        mock_track.side_effect = CircularImportError("Circular")
        with pytest.raises(CircularImportError):
            lazy_import("src", import_map, "circular", cache_module)

def test_preload_modules():

    reset_import_stats()
    MockModule()
    {"path": "os", "version": "sys"}
    
    with patch("importlib.import_module") as mock_import:
        mock_import.side_effect = [sys.modules['os'], sys.modules['sys']]
        preload_modules("src", import_map, ["path", "version"], cache_module_override=cache_module)
        assert hasattr(cache_module, "path")
        assert hasattr(cache_module, "version")

def test_lazy_import_double_check():
    """Test the case where attribute exists after lock acquisition."""
    reset_import_stats()
    MockModule()
    cache_module.test_attr = "existing"
    {"test_attr": "os.path"}
    
    # should return existing without importing
    with patch("importlib.import_module") as mock_import:
        res = lazy_import("src", import_map, "test_attr", cache_module)
        assert res == "existing"
        mock_import.assert_not_called()

def test_preload_modules_failure():
    """Test preload_modules when an import fails."""
    reset_import_stats()
    MockModule()
    {"path": "os"}
    
    with patch("src.utils.lazy_import.lazy_import", side_effect=Exception("Preload fail")):
        # Should not raise exception, just log warning
        preload_modules("src", import_map, ["path"], cache_module_override=cache_module)

def test_lazy_import_relative():
    reset_import_stats()
    MockModule()
    {"sub": ".submodule"}
    
    with patch("importlib.import_module") as mock_import:
        mock_mod = MagicMock()
        mock_mod.sub = "val"
        mock_import.return_value = mock_mod
        
        # Should call import_module('package.submodule', package='package')
        res = lazy_import("package", import_map, "sub", cache_module)
        assert res == "val"
        mock_import.assert_called_with(".submodule", package="package")

def test_preload_modules_failure_again():
    reset_import_stats()
    MockModule()
    {"fail": "nonexistent"}
    
    with patch("importlib.import_module", side_effect=Exception("Fail")):
        # Should catch exception and log warning, not raise
        preload_modules("src", import_map, ["fail"], cache_module_override=cache_module)
        # Verify it didn't crash

def test_lazy_import_already_loaded():
    reset_import_stats()
    MockModule()
    cache_module.existing = "value"
    {"existing": "os"}
    
    # Should return existing value without importing
    with patch("importlib.import_module") as mock_import:
        res = lazy_import("src", import_map, "existing", cache_module)
        assert res == "value"
        mock_import.assert_not_called()


