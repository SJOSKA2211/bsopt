import tracemalloc

from fastapi import APIRouter, Depends
from sqlalchemy import text

from src.api.exceptions import (
    InternalServerException,  # Imported directly as it's a specific exception
)
from src.api.responses import MsgspecJSONResponse
from src.api.schemas.common import DataResponse, ErrorResponse
from src.database import get_async_db, health_check
from src.auth.auth import require_tier

router = APIRouter(
    prefix="/debug",
    tags=["Debug & Diagnostics"],
    dependencies=[Depends(require_tier(["admin"]))],
    default_response_class=MsgspecJSONResponse,
)


@router.get(
    "/tracemalloc_snapshot",
    response_model=DataResponse[dict],
    responses={500: {"model": ErrorResponse, "description": "Tracemalloc not active"}},
    dependencies=[Depends(require_tier(["admin"]))],
)
async def get_tracemalloc_snapshot():
    """
    Retrieves a snapshot of current memory allocations tracked by tracemalloc.
    WARNING: Exposing this endpoint publicly can be a security risk.
    """
    if not tracemalloc.is_tracing():
        raise InternalServerException(message="Tracemalloc is not active.")

    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics("traceback")  # Changed to 'traceback'

    # Display top 20 items with limited traceback depth for performance
    report = [
        {
            "size_kb": stat.size / 1024,
            "count": stat.count,
            "traceback": [
                {"file": frame.filename, "line": frame.lineno}
                for frame in stat.traceback[:10]  # Limit depth
            ],
        }
        for stat in top_stats[:20]
    ]

    return DataResponse(
        data={"top_memory_allocations": report},
        message="Tracemalloc snapshot taken successfully.",
    )


@router.get("/database/health", response_model=DataResponse[dict])
async def get_db_health():
    """Detailed database health audit."""
    return DataResponse(data=health_check(), message="Database health audit complete.")


@router.get("/database/sluggish_queries", response_model=DataResponse[list[dict]])
async def get_sluggish_queries(db=Depends(get_async_db)):
    """Fetch top 20 sluggish queries from the performance manifold."""
    result = await db.execute(text("SELECT * FROM pg_stat_sluggish_queries"))
    queries = [dict(row._mapping) for row in result]
    return DataResponse(data=queries, message="Sluggish queries retrieved.")
