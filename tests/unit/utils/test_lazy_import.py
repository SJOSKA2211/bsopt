import pytest
import sys
import threading
from src.utils.lazy_import import (
    lazy_import, 
    get_import_stats, 
    reset_import_stats, 
    preload_modules, 
    LazyImportError, 
    CircularImportError
)

class MockModule:
    def __init__(self):
        self.__dict__ = {}

@pytest.fixture(autouse=True)
def clean_stats():
    reset_import_stats()
    yield

def test_lazy_import_success(mocker):
    import_map = {"BSParameters": "src.pricing.models"}
    mock_mod = MockModule()
    # Mock import_module to prevent real imports and ensure stats record it
    mocker.patch("importlib.import_module")
    
    attr = lazy_import("src.pricing", import_map, "BSParameters", mock_mod)
    assert attr is not None
    assert mock_mod.BSParameters is attr
    
    # Second call should use cache
    attr2 = lazy_import("src.pricing", import_map, "BSParameters", mock_mod)
    assert attr2 is attr

def test_lazy_import_attribute_error():
    import_map = {"valid": "mod"}
    mock_mod = MockModule()
    with pytest.raises(AttributeError, match="has no attribute 'invalid'"):
        lazy_import("pkg", import_map, "invalid", mock_mod)

def test_lazy_import_previous_failure(mocker):
    import_map = {"Fail": "nonexistent"}
    mock_mod = MockModule()
    
    reset_import_stats()
    
    # First fail
    with pytest.raises(LazyImportError):
        lazy_import("pkg", import_map, "Fail", mock_mod)
        
    # Second call should raise from cache
    with pytest.raises(LazyImportError, match="Previous import of pkg.Fail failed"):
        lazy_import("pkg", import_map, "Fail", mock_mod)

def test_lazy_import_circular(mocker):
    import_map = {"Attr": ".mod"}
    mock_mod = MockModule()
    
    # We can trigger circular import by calling lazy_import within an import stack
    from src.utils.lazy_import import _track_import_stack
    
    # Mock import_module so it doesn't fail before stack check
    mocker.patch("importlib.import_module")
    
    with _track_import_stack("pkg.mod"):
        with pytest.raises(CircularImportError):
            lazy_import("pkg", import_map, "Attr", mock_mod)

def test_get_import_stats(mocker):
    reset_import_stats()
    stats = get_import_stats()
    assert stats["successful_imports"] == 0
    
    # One success
    mocker.patch("importlib.import_module")
    lazy_import("src.pricing", {"BSParameters": "src.pricing.models"}, "BSParameters", MockModule())
    stats = get_import_stats()
    assert stats["successful_imports"] == 1
    assert stats["total_import_time"] >= 0

def test_preload_modules():
    import_map = {"BSParameters": "src.pricing.models"}
    mock_mod = MockModule()
    preload_modules("src.pricing", import_map, ["BSParameters"], cache_module_override=mock_mod)
    assert hasattr(mock_mod, "BSParameters")

def test_preload_modules_failed(mocker):
    import_map = {"Fail": "nonexistent"}
    mock_mod = MockModule()
    # Should not raise, just log warning
    preload_modules("pkg", import_map, ["Fail"], cache_module_override=mock_mod)

def test_lazy_import_thread_safe():
    import_map = {"BSParameters": "src.pricing.models"}
    mock_mod = MockModule()
    
    def worker():
        lazy_import("src.pricing", import_map, "BSParameters", mock_mod)
        
    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads: t.start()
    for t in threads: t.join()
    
    assert hasattr(mock_mod, "BSParameters")
