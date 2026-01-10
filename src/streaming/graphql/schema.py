import strawberry
from strawberry.federation import Schema
from typing import Optional
import random

@strawberry.federation.type(keys=["id"])
class Option:
    id: strawberry.ID
    
    @strawberry.field
    def last_price(self) -> float:
        # In a real app, this would query Redis or Kafka Streams state store
        return 15.0 + random.uniform(0, 1.0)

    @strawberry.field
    def volume(self) -> int:
        return random.randint(100, 10000)

    @classmethod
    def resolve_reference(cls, id: strawberry.ID):
        return cls(id=id)

@strawberry.type
class Query:
    @strawberry.field
    def _dummy_market(self) -> str:
        return "market"

schema = Schema(query=Query, types=[Option])
