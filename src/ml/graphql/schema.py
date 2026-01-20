import strawberry
from strawberry.federation import Schema
import random

@strawberry.federation.type(keys=["id"])
class Option:
    id: strawberry.ID
    
    @strawberry.field
    def fair_value(self) -> float:
        # Mock ML inference
        # In a real scenario, this would call the loaded XGBoost/PyTorch model
        return 15.5 + random.uniform(-0.5, 0.5)

    @strawberry.field
    def recommendation(self) -> str:
        # Mock inference
        return random.choice(["BUY", "SELL", "HOLD"])

    @classmethod
    def resolve_reference(cls, id: strawberry.ID):
        return cls(id=id)

@strawberry.type
class Query:
    @strawberry.field
    def _dummy_ml(self) -> str:
        return "ml"

schema = Schema(query=Query, types=[Option])
