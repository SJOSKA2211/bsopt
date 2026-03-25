from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Pre-emptive mock for wasmer before any imports
mock_wasmer = MagicMock()
import sys

sys.modules["wasmer"] = mock_wasmer
sys.modules["wasmer_compiler_cranelift"] = MagicMock()

from src.math_kernel.arbiter import (
    BSParameters,
    EngineArbiter,
    PricingModel,
    PricingRequest,
)


class TestArbiter:
    def setUp(self):
        # Patch WASMPricingEngine to not try to load real WASM
        with patch("src.math_kernel.arbiter.WASMPricingEngine") as mock_wasm:
            mock_instance = mock_wasm.return_value
            mock_instance.instance = None  # Force fallback in arbiter
            self.arbiter = EngineArbiter()
        self.params = BSParameters(
            spot=100.0,
            strike=100.0,
            maturity=1.0,
            volatility=0.2,
            rate=0.05,
            dividend=0.0,
        )

    def test_route_bs(self):
        req = PricingRequest(params=self.params, model=PricingModel.BLACK_SCHOLES)
        price = self.arbiter.route_request(req)
        self.assertGreater(price, 0)

    def test_route_mc(self):
        req = PricingRequest(
            params=self.params,
            model=PricingModel.MONTE_CARLO,
            engine_config={"n_paths": 1000},
        )
        price = self.arbiter.route_request(req)
        self.assertGreater(price, 0)

    def test_route_american(self):
        req = PricingRequest(params=self.params, style="american")
        price = self.arbiter.route_request(req)
        self.assertGreater(price, 0)

    def test_route_batch(self):
        S = np.array([100.0, 110.0])
        K = np.array([100.0, 100.0])
        T = np.array([1.0, 1.0])
        sigma = np.array([0.2, 0.2])
        r = np.array([0.05, 0.05])
        is_call = np.array([True, True])
        prices = self.arbiter.route_batch(S, K, T, sigma, r, is_call)
        assert len(prices) == 2

