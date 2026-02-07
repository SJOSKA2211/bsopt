from datetime import datetime

import strawberry
from strawberry.federation import Schema


@strawberry.federation.type(keys=["id"])
class Option:
    id: strawberry.ID
    contract_symbol: str
    underlying_symbol: str
    strike: float
    expiry: datetime
    option_type: str

@strawberry.type
class Query:
    @strawberry.field
    def option(self, contract_symbol: str) -> Option | None:
        # Mock data for now, as we aren't connecting to DB yet
        if contract_symbol == "AAPL_20260115_C_150":
            return Option(
                id=strawberry.ID(contract_symbol),
                contract_symbol=contract_symbol,
                underlying_symbol="AAPL",
                strike=150.0,
                expiry=datetime(2026, 1, 15),
                option_type="call"
            )
        return None

schema = Schema(query=Query)
