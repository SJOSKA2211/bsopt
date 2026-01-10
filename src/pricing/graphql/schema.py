import strawberry
from strawberry.federation import Schema
from typing import Optional
from datetime import datetime
from src.pricing.black_scholes import black_scholes

@strawberry.federation.type(keys=["id"])
class Option:
    id: strawberry.ID
    strike: float = strawberry.federation.field(shareable=True)
    underlying_symbol: str = strawberry.federation.field(shareable=True)
    expiry: datetime = strawberry.federation.field(shareable=True)
    option_type: str = strawberry.federation.field(shareable=True)
    
    @strawberry.field
    def price(self) -> float:
        # Mocking spot price and vol for now
        # In real world, we would fetch these from Market Data service or context
        S = 155.0 # Spot price
        r = 0.05
        sigma = 0.2
        
        # Calculate Time to Maturity
        T = (self.expiry - datetime.now()).days / 365.0
        if T <= 0: T = 0.001
        
        result = black_scholes(S, self.strike, T, r, sigma, self.option_type)
        return result["price"]

    @strawberry.field
    def delta(self) -> float:
        S = 155.0
        r = 0.05
        sigma = 0.2
        T = (self.expiry - datetime.now()).days / 365.0
        if T <= 0: T = 0.001
        
        result = black_scholes(S, self.strike, T, r, sigma, self.option_type)
        return result["delta"]

    @strawberry.field
    def gamma(self) -> float:
        S = 155.0
        r = 0.05
        sigma = 0.2
        T = (self.expiry - datetime.now()).days / 365.0
        if T <= 0: T = 0.001
        
        result = black_scholes(S, self.strike, T, r, sigma, self.option_type)
        return result["gamma"]

    @classmethod
    def resolve_reference(cls, id: strawberry.ID, strike: float, underlyingSymbol: str, expiry: str, optionType: str):
        # When the gateway resolves this entity, it passes the keys and any other fields defined in the representation
        # Expiry comes as string, need to parse
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
