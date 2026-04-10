import re

filename = "tests/unit/api/test_pricing_api.py"
with open(filename, "r") as f:
    content = f.read()

content = content.replace('mock_price.return_value = MagicMock(\n            price=12.34,\n            model="Heston-FFT",\n            spot=100.0,\n            strike=105.0,\n            cached=False\n        )', 'from api.schemas.pricing import PriceResult\n        mock_price.return_value = PriceResult(\n            price=12.34,\n            model="Heston-FFT",\n            spot=100.0,\n            strike=105.0,\n            cached=False,\n            greeks={}, computation_time_ms=0.0\n        )')

content = content.replace('mock_batch.return_value = MagicMock(\n            results=[],\n            total_count=2,\n            cached_count=0,\n            computation_time_ms=10.0\n        )', 'from api.schemas.pricing import BatchPriceResult\n        mock_batch.return_value = BatchPriceResult(\n            results=[],\n            total_count=2,\n            cached_count=0,\n            computation_time_ms=10.0\n        )')

content = content.replace('-> PriceResult:', '-> MsgspecJSONResponse:')
content = content.replace('-> BatchPriceResult:', '-> MsgspecJSONResponse:')
content = content.replace('-> BatchGreeksResult:', '-> MsgspecJSONResponse:')

with open(filename, "w") as f:
    f.write(content)

filename = "api/routes/pricing.py"
with open(filename, "r") as f:
    content = f.read()

content = content.replace('-> PriceResult:', '-> MsgspecJSONResponse:')
content = content.replace('-> BatchPriceResult:', '-> MsgspecJSONResponse:')
content = content.replace('-> BatchGreeksResult:', '-> MsgspecJSONResponse:')

with open(filename, "w") as f:
    f.write(content)

