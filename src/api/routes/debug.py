import tracemalloc
import json
from fastapi import APIRouter, status
from fastapi.responses import JSONResponse
from src.api.schemas.common import DataResponse, ErrorResponse
from src.api.exceptions import InternalServerException # Imported directly as it's a specific exception

router = APIRouter(prefix="/debug", tags=["Debug & Diagnostics"])

@router.get(
    "/tracemalloc_snapshot",
    response_model=DataResponse[dict],
    responses={
        500: {"model": ErrorResponse, "description": "Tracemalloc not active"}
    }
)
async def get_tracemalloc_snapshot():
    """
    Retrieves a snapshot of current memory allocations tracked by tracemalloc.
    WARNING: Exposing this endpoint publicly can be a security risk.
    """
    if not tracemalloc.is_tracing():
        raise InternalServerException(message="Tracemalloc is not active.")

    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics('lineno')

    report = []
    for stat in top_stats[:10]: # Top 10 by default
        report.append({
            "file": stat.traceback[0].filename,
            "line": stat.traceback[0].lineno,
            "size_kb": stat.size / 1024,
            "count": stat.count
        })
    
    return DataResponse(
        data={"top_10_memory_allocations": report},
        message="Tracemalloc snapshot taken successfully."
    )
