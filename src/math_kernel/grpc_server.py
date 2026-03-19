import asyncio
import grpc
from concurrent import futures
from src.protos import pricing_pb2, pricing_pb2_grpc
from src.math_kernel.service import PricingService
from src.math_kernel.black_scholes import BSParameters

class PricingServicer(pricing_pb2_grpc.PricingServiceServicer):
    def __init__(self):
        self.service = PricingService()

    async def PriceOption(self, request, context):
        params = BSParameters(
            spot=request.spot,
            strike=request.strike,
            maturity=request.time_to_expiry,
            volatility=request.volatility,
            rate=request.rate,
            dividend=request.dividend_yield
        )
        result = await self.service.price_option(
            params=params,
            option_type=request.option_type,
            model=request.model,
            symbol=request.symbol
        )
        return pricing_pb2.PriceResponse(
            price=result.price,
            spot=result.spot,
            strike=result.strike,
            time_to_expiry=result.time_to_expiry,
            rate=result.rate,
            volatility=result.volatility,
            option_type=result.option_type,
            model=result.model,
            computation_time_ms=result.computation_time_ms
        )

    async def PriceBatch(self, request, context):
        options = []
        for opt in request.options:
            options.append({
                "spot": opt.spot,
                "strike": opt.strike,
                "time_to_expiry": opt.time_to_expiry,
                "volatility": opt.volatility,
                "rate": opt.rate,
                "option_type": opt.option_type,
                "model": opt.model,
                "symbol": opt.symbol
            })
        result = await self.service.price_batch(options)
        
        proto_results = []
        for res in result.results:
            proto_results.append(pricing_pb2.PriceResponse(
                price=res.price,
                spot=res.spot,
                strike=res.strike,
                time_to_expiry=res.time_to_expiry,
                rate=res.rate,
                volatility=res.volatility,
                option_type=res.option_type,
                model=res.model,
                computation_time_ms=res.computation_time_ms
            ))
        
        return pricing_pb2.BatchPriceResponse(
            results=proto_results,
            total_count=result.total_count,
            computation_time_ms=result.computation_time_ms
        )

async def serve():
    server = grpc.aio.server()
    pricing_pb2_grpc.add_PricingServiceServicer_to_server(PricingServicer(), server)
    server.add_insecure_port('[::]:50052')
    print("gRPC Pricing Server starting on port 50052...")
    await server.start()
    await server.wait_for_termination()

if __name__ == '__main__':
    asyncio.run(serve())
