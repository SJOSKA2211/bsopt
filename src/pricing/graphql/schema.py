import strawberry
from strawberry.federation import Schema
from typing import Optional, Any, Dict
from datetime import datetime
from src.pricing.black_scholes import black_scholes
from src.pricing.quantum_pricing import HybridQuantumClassicalPricer

@strawberry.federation.type(keys=["id"])
class Option:
    id: strawberry.ID
    strike: float = strawberry.federation.field(shareable=True)
    underlying_symbol: str = strawberry.federation.field(shareable=True)
    expiry: datetime = strawberry.federation.field(shareable=True)
    option_type: str = strawberry.federation.field(shareable=True)
    
    @strawberry.field
    def price(self, accuracy: float = 0.01, num_underlyings: int = 1) -> float:
        # In a production system, these would be fetched from a Market Data service
        S0 = 155.0 
        r = 0.05
        sigma = 0.2
        
        T = (self.expiry - datetime.now()).days / 365.0
        if T <= 0: T = 0.001
        
        pricer = HybridQuantumClassicalPricer()
        result = pricer.price_option_adaptive(
            S0=S0, K=self.strike, T=T, r=r, sigma=sigma,
            accuracy=accuracy, num_underlyings=num_underlyings
        )
        return result["price"]

    @strawberry.field
    def delta(self) -> float:
        S0 = 155.0
        r = 0.05
        sigma = 0.2
        T = (self.expiry - datetime.now()).days / 365.0
        if T <= 0: T = 0.001
        
        result = black_scholes(S0, self.strike, T, r, sigma, self.option_type)
        return result["delta"]

    @strawberry.field
    def gamma(self) -> float:
        S0 = 155.0
        r = 0.05
        sigma = 0.2
        T = (self.expiry - datetime.now()).days / 365.0
        if T <= 0: T = 0.001
        
        result = black_scholes(S0, self.strike, T, r, sigma, self.option_type)
        return result["gamma"]

    @classmethod
    def resolve_reference(cls, id: strawberry.ID, strike: float, underlyingSymbol: str, expiry: str, optionType: str):
        if isinstance(expiry, str):
            expiry_dt = datetime.fromisoformat(expiry)
        else:
            expiry_dt = expiry
            
        return cls(
            id=id, 
            strike=strike, 
            underlying_symbol=underlyingSymbol, 
            expiry=expiry_dt, 
            option_type=optionType
        )

@strawberry.type
class Query:
    @strawberry.field
    def _dummy(self) -> str:
        return "pricing"

schema = Schema(query=Query, types=[Option])
