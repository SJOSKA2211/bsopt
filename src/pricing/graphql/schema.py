from datetime import date, datetime

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
        # key: (id, strike, symbol, expiry, option_type)
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


@strawberry.federation.type(keys=["id"], extend=True)
class Option:
    id: strawberry.ID = strawberry.federation.field(external=True)
    strike: float = strawberry.federation.field(external=True)
    symbol: str = strawberry.federation.field(external=True)
    expiry: date = strawberry.federation.field(external=True)
    option_type: str = strawberry.federation.field(external=True)

    @strawberry.federation.field(
        requires=["strike", "symbol", "expiry", "optionType"], shareable=True
    )
    async def price(self, info: strawberry.Info) -> float:
        loader = info.context["price_loader"]
        key = (
            self.id,
            self.strike,
            self.symbol,
            self.expiry,
            self.option_type,
        )
        return await loader.load(key)

    @strawberry.federation.field(
        requires=["strike", "symbol", "expiry", "optionType"], shareable=True
    )
    async def delta(self, info: strawberry.Info) -> float | None:
        loader = info.context["greeks_loader"]
        key = (
            self.id,
            self.strike,
            self.symbol,
            self.expiry,
            self.option_type,
        )
        res = await loader.load(key)
        return res["delta"]

    @strawberry.federation.field(
        requires=["strike", "symbol", "expiry", "optionType"], shareable=True
    )
    async def gamma(self, info: strawberry.Info) -> float | None:
        loader = info.context["greeks_loader"]
        key = (
            self.id,
            self.strike,
            self.symbol,
            self.expiry,
            self.option_type,
        )
        res = await loader.load(key)
        return res["gamma"]

    @strawberry.field
    def iv(self) -> float:
        """Implied Volatility - currently returning mock value"""
        return 0.2

    @classmethod
    def resolve_reference(
        cls,
        id: strawberry.ID,
        strike: float,
        symbol: str,
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
            symbol=symbol,
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
