import unittest
from unittest.mock import MagicMock, patch
import sys

# Pre-emptive mock for wasmer before any imports
mock_wasmer = MagicMock()
sys.modules["wasmer"] = mock_wasmer
sys.modules["wasmer_compiler_cranelift"] = MagicMock()

from src.pricing.factory import PricingEngineFactory, PricingEngineNotFound

class TestPricingFactory(unittest.TestCase):
    def setUp(self):
        # Clear instances for testing
        PricingEngineFactory._instances = {}
        # Mock WASM_AVAILABLE to False to avoid auto-routing to WASM
        self.wasm_patcher = patch("src.pricing.wasm_engine.WASM_AVAILABLE", False)
        self.wasm_patcher.start()

    def tearDown(self):
        self.wasm_patcher.stop()

    def test_get_bs_engine(self):
        engine = PricingEngineFactory.get_engine("black_scholes")
        self.assertIsNotNone(engine)
        self.assertEqual(type(engine).__name__, "BlackScholesEngine")

    def test_get_mc_engine(self):
        engine = PricingEngineFactory.get_engine("monte_carlo")
        self.assertIsNotNone(engine)
        self.assertEqual(type(engine).__name__, "MonteCarloEngine")

    def test_engine_not_found(self):
        with self.assertRaises(PricingEngineNotFound):
            PricingEngineFactory.get_engine("non_existent_engine")

    def test_singleton_behavior(self):
        e1 = PricingEngineFactory.get_engine("black_scholes")
        e2 = PricingEngineFactory.get_engine("black_scholes")
        self.assertIs(e1, e2)

    def test_default_override(self):
        PricingEngineFactory.set_default_engine("monte_carlo")
        engine = PricingEngineFactory.get_engine("black_scholes")
        self.assertEqual(type(engine).__name__, "MonteCarloEngine")
        PricingEngineFactory.set_default_engine(None)

if __name__ == '__main__':
    unittest.main()
