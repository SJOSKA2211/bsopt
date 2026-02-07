import tracemalloc

from fastapi import APIRouter

from src.api.exceptions import (
    InternalServerException,  # Imported directly as it's a specific exception
)
from src.api.schemas.common import DataResponse, ErrorResponse

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
    top_stats = snapshot.statistics('traceback') # Changed to 'traceback'

    report = []
    # Display more comprehensive traceback information
    for stat in top_stats[:20]: # Display top 20 items
        report.append({
            "size_kb": stat.size / 1024,
            "count": stat.count,
            "traceback": [
                {"file": frame.filename, "line": frame.lineno}
                for frame in stat.traceback # Iterate over all frames in the traceback
            ]
        })
    
    return DataResponse(
        data={"top_10_memory_allocations": report},
        message="Tracemalloc snapshot taken successfully."
    )
