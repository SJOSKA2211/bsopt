import pytest

from src.math_kernel.exotic import (
    AsianOptionPricer,
    AsianType,
    BarrierOptionPricer,
    BarrierType,
    BSParameters,
    DigitalOptionPricer,
    ExoticParameters,
    LookbackOptionPricer,
    StrikeType,
)

class TestExotic:
    def setUp(self):
        self.base_params = BSParameters(
            spot=100.0,
            strike=100.0,
            maturity=1.0,
            volatility=0.2,
            rate=0.05,
            dividend=0.0,
        )
        self.exotic_params = ExoticParameters(
            base_params=self.base_params, barrier=120.0, rebate=5.0
        )

    def test_asian_geometric(self):
        price = AsianOptionPricer.price_geometric_asian(self.exotic_params, "call")
        self.assertGreater(price, 0)

    def test_asian_arithmetic_mc(self):
        price, std_err = AsianOptionPricer.price_arithmetic_asian_mc(
            self.exotic_params, "call", n_paths=1000
        )
        self.assertGreater(price, 0)

    def test_barrier_analytical(self):
        price = BarrierOptionPricer.price_barrier_analytical(
            self.exotic_params, "call", BarrierType.UP_AND_OUT
        )
        self.assertGreater(price, 0)

    def test_lookback_mc(self):
        price, std_err = LookbackOptionPricer.price_lookback_mc(
            self.exotic_params, "call", StrikeType.FLOATING, n_paths=1000
        )
        self.assertGreater(price, 0)

    def test_digital_cash(self):
        price = DigitalOptionPricer.price_cash_or_nothing(self.base_params, "call", payout=10.0)
        self.assertGreater(price, 0)

    def test_barrier_all_types(self):
        for bt in BarrierType:
            # Ensure H is on the correct side of S for the type
            if "up" in bt.value:
                self.exotic_params.barrier = 120.0
            else:
                self.exotic_params.barrier = 80.0

            price = BarrierOptionPricer.price_barrier_analytical(self.exotic_params, "call", bt)
            self.assertGreaterEqual(price, 0)

    def test_lookback_floating_analytical(self):
        price = LookbackOptionPricer.price_floating_strike_analytical(self.base_params, "call")
        self.assertGreater(price, 0)

    def test_digital_asset_or_nothing(self):
        price = DigitalOptionPricer.price_asset_or_nothing(self.base_params, "call")
        self.assertGreater(price, 0)

    def test_price_exotic_option_dispatch(self):
        from src.math_kernel.exotic import price_exotic_option

        p, _ = price_exotic_option(
            "asian", self.exotic_params, "call", asian_type=AsianType.GEOMETRIC
        )
        self.assertGreater(p, 0)

        p2, _ = price_exotic_option(
            "barrier", self.exotic_params, "call", barrier_type=BarrierType.UP_AND_OUT
        )
