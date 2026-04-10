import re

filename = "tests/unit/api/test_pricing_api.py"
with open(filename, "r") as f:
    content = f.read()

content = content.replace('data = response.json()["data"]\n        assert data["price"]', 'data = response.json()\n        assert data["price"]')
content = content.replace('assert response.json()["data"]["price"]', 'assert response.json()["price"]')

with open(filename, "w") as f:
    f.write(content)

filename = "api/routes/pricing.py"
with open(filename, "r") as f:
    content = f.read()

# For calculate_price
content = content.replace(
    '    return await pricing_service.price_option(\n        params=params,\n        option_type=body.option_type,\n        model=body.model,\n        symbol=body.symbol,\n    )',
    '    res = await pricing_service.price_option(\n        params=params,\n        option_type=body.option_type,\n        model=body.model,\n        symbol=body.symbol,\n    )\n    return MsgspecJSONResponse(content=res)'
)

# For calculate_batch_prices
content = content.replace(
    '    return await pricing_service.price_batch(request.options)',
    '    res = await pricing_service.price_batch(request.options)\n    return MsgspecJSONResponse(content=res)'
)

# For calculate_batch_greeks
content = content.replace(
    '    return await pricing_service.calculate_greeks_batch(request.options)',
    '    res = await pricing_service.calculate_greeks_batch(request.options)\n    return MsgspecJSONResponse(content=res)'
)

# For calculate_greeks
content = content.replace(
    '    result = await pricing_service.calculate_greeks(params, body.option_type)\n    return result',
    '    result = await pricing_service.calculate_greeks(params, body.option_type)\n    return MsgspecJSONResponse(content=result)'
)

with open(filename, "w") as f:
    f.write(content)

