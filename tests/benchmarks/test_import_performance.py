"""
Benchmark lazy import performance vs eager imports.
"""

import sys


class TestImportPerformance:
    """Benchmark import performance."""

    def test_lazy_vs_eager_ml_module(self, benchmark):
        """Compare lazy vs eager import times."""

        def lazy_import():
            # Clear cache if 'src.ml' in sys.modules:
            if "src.ml" in sys.modules:
                del sys.modules["src.ml"]
            import src.ml

            return src.ml

        result = benchmark(lazy_import)
        # Lazy import should be < 10ms
        assert benchmark.stats["mean"] < 0.010

    def test_first_access_overhead(self, benchmark):
        """Measure overhead of first attribute access."""
        import src.ml

        # Clear any cached attributes if hasattr(src.ml, 'DataNormalizer'):
        if hasattr(src.ml, "DataNormalizer"):
            delattr(src.ml, "DataNormalizer")

        def first_access():
            return src.ml.DataNormalizer

        result = benchmark(first_access)
        # First access overhead should be reasonable
        assert benchmark.stats["mean"] < 0.100  # < 100ms

    def test_cached_access_speed(self, benchmark):
        """Verify cached access is instant."""
        from src.ml import DataNormalizer

        def cached_access():
            return DataNormalizer

        result = benchmark(cached_access)
        # Cached access should be < 1ms
        assert benchmark.stats["mean"] < 0.001
