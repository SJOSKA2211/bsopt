from datetime import datetime

import strawberry
from strawberry.dataloader import DataLoader
from strawberry.federation import Schema

from src.api.schemas.pricing import PriceRequest
from src.services.pricing_service import PricingService

# Instantiate the optimized service
pricing_service = PricingService()


async def load_prices(keys: list[tuple]) -> list[float]:
    """Batch loader for option prices."""
    requests = []
    for key in keys:
        # key: (id, strike, underlying_symbol, expiry, option_type)
        _, strike, _, expiry, option_type = key
        T = (expiry - datetime.now()).days / 365.0
        if T <= 0:
            T = 0.001

        requests.append(
            PriceRequest(
                symbol="UNKNOWN",
                spot=155.0,  # Placeholder, should come from market data
                strike=strike,
                time_to_expiry=T,
                volatility=0.2,  # Placeholder
                rate=0.05,
                option_type=option_type.lower(),
                model="black_scholes",
            )
        )

    batch_res = await pricing_service.price_batch(requests)
    return [r.price for r in batch_res.results]


async def load_greeks(keys: list[tuple]) -> list[dict[str, float]]:
    """Batch loader for option Greeks."""
    requests = []
    for key in keys:
        _, strike, _, expiry, option_type = key
        T = (expiry - datetime.now()).days / 365.0
        if T <= 0:
            T = 0.001

        requests.append(
            PriceRequest(
                symbol="UNKNOWN",
                spot=155.0,
                strike=strike,
                time_to_expiry=T,
                volatility=0.2,
                rate=0.05,
                option_type=option_type.lower(),
                model="black_scholes",
            )
        )

    batch_res = await pricing_service.calculate_greeks_batch(requests)

    # Map results back to the original format expected by resolvers
    return [
        {
            "delta": r.delta,
            "gamma": r.gamma,
            "theta": r.theta,
            "vega": r.vega,
            "rho": r.rho,
            "price": r.option_price,
        }
        for r in batch_res.results
    ]


@strawberry.federation.type(keys=["id"])
class Option:
    id: strawberry.ID
    strike: float = strawberry.federation.field(shareable=True)
    underlying_symbol: str = strawberry.federation.field(shareable=True)
    expiry: datetime = strawberry.federation.field(shareable=True)
    option_type: str = strawberry.federation.field(shareable=True)

    @strawberry.field
    async def price(self, info: strawberry.Info) -> float:
        loader = info.context["price_loader"]
        key = (
            self.id,
            self.strike,
            self.underlying_symbol,
            self.expiry,
            self.option_type,
        )
        return await loader.load(key)

    @strawberry.field
    async def delta(self, info: strawberry.Info) -> float:
        loader = info.context["greeks_loader"]
        key = (
            self.id,
            self.strike,
            self.underlying_symbol,
            self.expiry,
            self.option_type,
        )
        res = await loader.load(key)
        return res["delta"]

    @strawberry.field
    async def gamma(self, info: strawberry.Info) -> float:
        loader = info.context["greeks_loader"]
        key = (
            self.id,
            self.strike,
            self.underlying_symbol,
            self.expiry,
            self.option_type,
        )
        res = await loader.load(key)
        return res["gamma"]

    @classmethod
    def resolve_reference(
        cls,
        id: strawberry.ID,
        strike: float,
        underlyingSymbol: str,
        expiry: str,
        optionType: str,
    ):
        if isinstance(expiry, str):
            expiry_dt = datetime.fromisoformat(expiry)
        else:
            expiry_dt = expiry

        return cls(
            id=id,
            strike=strike,
            underlying_symbol=underlyingSymbol,
            expiry=expiry_dt,
            option_type=optionType,
        )


@strawberry.type
class Query:
    @strawberry.field
    def dummy(self) -> str:
        return "pricing"


async def get_context():
    return {
        "price_loader": DataLoader(load_fn=load_prices),
        "greeks_loader": DataLoader(load_fn=load_greeks),
    }


schema = Schema(query=Query, types=[Option])
