import strawberry
from strawberry.federation import Schema
from strawberry.dataloader import DataLoader
from typing import List
import random
from src.services.ml_service import ml_service
from src.api.schemas.ml import InferenceRequest

async def load_fair_values(keys: List[strawberry.ID]) -> List[float]:
    """Batch loader for ML fair values."""
    # In a real implementation, we'd batch these to the ML microservice
    # For now, we simulate batching but use the service logic
    results = []
    for key in keys:
        # Mocking inference for now as we don't have a real batch inference endpoint
        # but using the service pattern.
        results.append(15.5 + random.uniform(-0.5, 0.5))
    return results

@strawberry.federation.type(keys=["id"])
class Option:
    id: strawberry.ID
    
    @strawberry.field
    async def fair_value(self, info: strawberry.Info) -> float:
        loader = info.context["fair_value_loader"]
        return await loader.load(self.id)

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

async def get_context():
    return {
        "fair_value_loader": DataLoader(load_fn=load_fair_values),
    }

schema = Schema(query=Query, types=[Option])
