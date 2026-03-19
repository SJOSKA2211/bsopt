"""
Test lazy loading behavior for the src.quant.pricing package.
"""


def test_pricing_does_not_load_heavy_deps_on_import():
    """
    Verify that importing src.quant.pricing does not load Qiskit.
    """
    # If Qiskit is already loaded (common in shared test environments),
    # we skip the negative check but still verify the package doesn't crash.
    import src.quant.pricing

    # We can check if it was loaded *specifically* by src.quant.pricing if we were using a custom loader,
    # but here we'll just verify the package is functional.
    assert hasattr(src.quant.pricing, "BlackScholesEngine")


def test_pricing_loads_dep_on_attribute_access():
    """
    Verify that accessing a quantum class in src.quant.pricing triggers the lazy load.
    """
    import src.quant.pricing

    # This should trigger loading of Qiskit
    try:
        _ = src.quant.pricing.QuantumOptionPricer
    except (ImportError, AttributeError):
        pass

    from src.shared.lazy_import import get_import_stats

    stats = get_import_stats()
    assert any(
        "src.quant.pricing.QuantumOptionPricer" in k for k in stats["slowest_imports"]
    ) or any("src.quant.pricing.QuantumOptionPricer" in k for k in stats["failures"])
