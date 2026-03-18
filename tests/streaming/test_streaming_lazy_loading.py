"""
Test lazy loading behavior for the src.workers.streaming package.
"""

import sys


class TestStreamingLazyLoading:
    def setup_method(self):
        # Clear any cached imports from previous tests
        modules_to_clear = [
            mod
            for mod in sys.modules.keys()
            if mod.startswith("src.workers.streaming") or mod == "confluent_kafka"
        ]
        for mod in modules_to_clear:
            del sys.modules[mod]
        # Re-import src.workers.streaming to ensure a clean state for each test
        if "src.workers.streaming" in sys.modules:
            del sys.modules["src.workers.streaming"]

    def test_streaming_does_not_load_heavy_deps_on_import(self):
        """
        Verify that importing src.workers.streaming does not load Kafka C-extensions.
        """
        assert "confluent_kafka" not in sys.modules

    def test_streaming_loads_dep_on_attribute_access(self):
        """
        Verify that accessing a Kafka class in src.workers.streaming triggers the lazy load.
        """
        import src.workers.streaming

        # Accessing MarketDataProducer should trigger import
        _ = src.workers.streaming.MarketDataProducer

        assert "confluent_kafka" in sys.modules

    def test_dir_returns_all_exports(self):
        """
        Verify dir() returns all exported names.
        """
        import src.workers.streaming

        exports = dir(src.workers.streaming)
        assert "MarketDataProducer" in exports
        assert "KafkaHealthCheck" in exports
        assert "VolatilityAggregationStream" in exports
