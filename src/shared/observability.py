def setup_logging(): pass
async def logging_middleware(request, call_next): return await call_next(request)
