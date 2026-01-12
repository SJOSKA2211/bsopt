from strawberry.federation import Schema

def test_options_subgraph():
    from src.api.graphql.schema import schema
    assert isinstance(schema, Schema)
    sdl = schema.as_str()
    assert 'type Option @key(fields: "id")' in sdl
    assert 'option(contractSymbol: String!): Option' in sdl

def test_pricing_subgraph():
    from src.pricing.graphql.schema import schema
    assert isinstance(schema, Schema)
    sdl = schema.as_str()
    # It should extend Option and add price/delta/gamma
    assert 'type Option @key(fields: "id")' in sdl
    assert 'price(accuracy: Float! = 0.01, numUnderlyings: Int! = 1): Float!' in sdl

def test_ml_subgraph():
    from src.ml.graphql.schema import schema
    assert isinstance(schema, Schema)
    sdl = schema.as_str()
    assert 'type Option @key(fields: "id")' in sdl
    assert 'fairValue: Float!' in sdl
    assert 'recommendation: String!' in sdl

def test_portfolio_subgraph():
    from src.portfolio.graphql.schema import schema
    assert isinstance(schema, Schema)
    sdl = schema.as_str()
    assert 'extend type Option @key(fields: "id")' in sdl
    assert 'type Portfolio' in sdl
    assert 'type Order' in sdl
    assert 'createOrder' in sdl

def test_marketdata_subgraph():
    from src.streaming.graphql.schema import schema
    assert isinstance(schema, Schema)
    sdl = schema.as_str()
    assert 'type Option @key(fields: "id")' in sdl
    assert 'lastPrice: Float!' in sdl
    assert 'marketDataStream' in sdl