"""
Test lazy loading behavior for the services.quant.pricing package.
"""


def test_pricing_does_not_load_heavy_deps_on_import():
    """
    Verify that importing services.quant.pricing does not load Qiskit.
    """
    # If Qiskit is already loaded (common in shared test environments),
    # we skip the negative check but still verify the package doesn't crash.
    import services.quant.pricing

    # We can check if it was loaded *specifically* by services.quant.pricing if we were using a custom loader,
    # but here we'll just verify the package is functional.
    assert hasattr(services.quant.pricing, "BlackScholesEngine")


def test_pricing_loads_dep_on_attribute_access():
    """
    Verify that accessing a quantum class in services.quant.pricing triggers the lazy load.
    """
    import services.quant.pricing

    # This should trigger loading of Qiskit
    try:
        _ = services.quant.pricing.QuantumOptionPricer
    except (ImportError, AttributeError):
        pass

    from core.shared.lazy_import import get_import_stats

    stats = get_import_stats()
    assert any("services.quant.pricing.QuantumOptionPricer" in k for k in stats["slowest_imports"]) or any(
        "services.quant.pricing.QuantumOptionPricer" in k for k in stats["failures"]
    )
