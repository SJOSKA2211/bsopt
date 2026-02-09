import sys
try:
    import strawberry
    from strawberry.federation import Schema
except ImportError as e:
    print(f"ImportError: {e}")
    sys.exit(1)

@strawberry.type
class Query:
    @strawberry.field
    def hello(self) -> str:
        return "world"

try:
    schema = Schema(query=Query)
    print("Strawberry schema created successfully")
except Exception as e:
    print(f"Schema creation failed: {e}")
    sys.exit(1)
