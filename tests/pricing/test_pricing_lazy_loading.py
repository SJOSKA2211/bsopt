"""
Test lazy loading behavior for the src.pricing package.
"""
import sys
import pytest

def test_pricing_does_not_load_heavy_deps_on_import():
    """
    Verify that importing src.pricing does not load Qiskit.
    """
    if 'qiskit' in sys.modules:
        pytest.skip("qiskit already loaded")

    # Ensure it's not already loaded
    assert 'qiskit' not in sys.modules
    
    import src.pricing
    
    # Still should not be loaded
    assert 'qiskit' not in sys.modules

def test_pricing_loads_dep_on_attribute_access():
    """
    Verify that accessing a quantum class in src.pricing triggers the lazy load.
    """
    import src.pricing
    
    # This should trigger loading of Qiskit
    try:
        _ = src.pricing.QuantumOptionPricer
    except (ImportError, AttributeError):
        # It's okay if it fails to import if not installed, 
        # but the attempt should have been made.
        pass
        
    # In a real environment with qiskit installed, it would be in sys.modules now.
    # For the test, we'll check if the import_map was at least consulted via stats.
    from src.utils.lazy_import import get_import_stats
    stats = get_import_stats()
    assert any('src.pricing.QuantumOptionPricer' in k for k in stats['slowest_imports']) or \
           any('src.pricing.QuantumOptionPricer' in k for k in stats['failures'])
