from dataclasses import dataclass

@dataclass
class BSParameters:
    spot: float
    strike: float
    maturity: float
    volatility: float
    rate: float
    dividend: float = 0.0

class BlackScholesEngine:
    def price_options(self, *args, **kwargs):
        raise NotImplementedError("BlackScholesEngine.price_options not implemented in stub")

    def calculate_greeks_batch(self, *args, **kwargs):
        raise NotImplementedError("BlackScholesEngine.calculate_greeks_batch not implemented in stub")
