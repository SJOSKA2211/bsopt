import pytest
import sys
from unittest.mock import MagicMock, patch
from src.utils.lazy_import import lazy_import, get_import_stats, reset_import_stats, preload_modules, CircularImportError, LazyImportError

class MockModule:
    def __init__(self):
        self.__dict__ = {}

def test_lazy_import_success():
    reset_import_stats()
    cache_module = MockModule()
    import_map = {"test_attr": "os.path"}
    
    with patch("src.utils.lazy_import.import_module") as mock_import:
        mock_mod = MagicMock()
        mock_mod.test_attr = "val"
        mock_import.return_value = mock_mod
        
        res = lazy_import("src", import_map, "test_attr", cache_module)
        assert res == "val"
        assert cache_module.test_attr == "val"
        
        stats = get_import_stats()
        assert stats['successful_imports'] == 1

def test_lazy_import_attribute_error():
    cache_module = MockModule()
    import_map = {"a": "b"}
    with pytest.raises(AttributeError):
        lazy_import("src", import_map, "wrong", cache_module)

def test_lazy_import_circular():
    reset_import_stats()
    cache_module = MockModule()
    import_map = {"circular": "src.circular"}
    
    # We need to manually trigger the circular detection by nested calls if we want to test the logic
    from src.utils.lazy_import import _track_import_stack
    
    with _track_import_stack("src.circular"):
        with pytest.raises(CircularImportError):
            with _track_import_stack("src.circular"):
                pass

def test_lazy_import_failure_caching():
    reset_import_stats()
    cache_module = MockModule()
    import_map = {"fail": "nonexistent"}
    
    with patch("src.utils.lazy_import.import_module", side_effect=ImportError("No module")):
        with pytest.raises(LazyImportError):
            lazy_import("src", import_map, "fail", cache_module)
            
        # Second attempt should raise LazyImportError from cache
        with pytest.raises(LazyImportError, match="Previous import"):
            lazy_import("src", import_map, "fail", cache_module)

def test_lazy_import_circular_reraise():
    reset_import_stats()
    cache_module = MockModule()
    import_map = {"circular": "src.circular"}
    
    with patch("src.utils.lazy_import._track_import_stack") as mock_track:
        mock_track.side_effect = CircularImportError("Circular")
        with pytest.raises(CircularImportError):
            lazy_import("src", import_map, "circular", cache_module)

def test_preload_modules():

    reset_import_stats()
    cache_module = MockModule()
    import_map = {"path": "os", "version": "sys"}
    
    with patch("src.utils.lazy_import.import_module") as mock_import:
        mock_import.side_effect = [sys.modules['os'], sys.modules['sys']]
        preload_modules("src", import_map, ["path", "version"], cache_module_override=cache_module)
        assert hasattr(cache_module, "path")
        assert hasattr(cache_module, "version")

