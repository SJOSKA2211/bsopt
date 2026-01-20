"""
Test lazy loading behavior for the src.ml package.
"""
import sys
import pytest

def test_ml_does_not_load_heavy_deps_on_import():
    """
    Verify that importing src.ml does not load PyTorch or Ray.
    """
    if 'torch' in sys.modules:
        pytest.skip("torch already loaded")
    
    # Ensure they aren't already loaded
    assert 'torch' not in sys.modules
    assert 'ray' not in sys.modules
    
    import src.ml
    
    # Still should not be loaded
    assert 'torch' not in sys.modules
    assert 'ray' not in sys.modules

def test_ml_loads_dep_on_attribute_access():
    """
    Verify that accessing a class in src.ml triggers the lazy load.
    """
    import src.ml
    
    # This should trigger loading of something that depends on torch/ray
    # According to our plan, TFTModel is in src.ml.forecasting.tft_model
    # We'll check if torch is loaded after access
    try:
        _ = src.ml.TFTModel
    except (ImportError, AttributeError):
        # It's okay if it fails to import if not installed, 
        # but the attempt should have been made.
        pass
        
    # In a real environment with torch installed, it would be in sys.modules now.
    # For the test, we'll check if the import_map was at least consulted via stats.
    from src.utils.lazy_import import get_import_stats
    stats = get_import_stats()
    assert any('src.ml.TFTModel' in k for k in stats['slowest_imports']) or \
           any('src.ml.TFTModel' in k for k in stats['failures'])
