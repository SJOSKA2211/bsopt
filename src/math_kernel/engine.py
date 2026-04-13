from src.math_kernel.wasm_engine import WASMPricingEngine


class BlackScholesWASM(WASMPricingEngine):
    def __init__(self, model="black_scholes"):
        super().__init__(model=model)


class AmericanOptionsWASM(WASMPricingEngine):
    def __init__(self, model="fdm"):
        super().__init__(model=model)