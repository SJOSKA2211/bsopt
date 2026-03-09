from datetime import datetime

import strawberry
from strawberry.dataloader import DataLoader

from src.data.router import MarketDataRouter
from src.shared.shm_mesh import SharedMemoryRingBuffer

router = MarketDataRouter()

# Singleton SHM reader for DataLoaders
_shm_reader = SharedMemoryRingBuffer(create=False)


@strawberry.type
class Option:
    id: strawberry.ID
    contract_symbol: str
    underlying_symbol: str
    strike: float
    expiry: datetime
    option_type: str
    last: float | None = None
    delta: float | None = None


async def _load_options_vectorized(keys: list[str]) -> list[Option]:
    """Vectorized batch fetcher for DataLoaders using Speculative Concurrency."""
    # 1. Dispatch all fetches concurrently
    tasks = [router.get_live_quote(symbol) for symbol in keys]
    results_raw = await asyncio.gather(*tasks, return_exceptions=True)

    now = datetime.now()
    results = []

    for i, symbol in enumerate(keys):
        res = results_raw[i]
        if isinstance(res, Exception) or "error" in res:
            # Fallback to minimal object
            results.append(
                Option(
                    id=strawberry.ID(symbol),
                    contract_symbol=symbol,
                    underlying_symbol=symbol.split("_")[0] if "_" in symbol else symbol,
                    strike=100.0,
                    expiry=now,
                    option_type="CALL",
                )
            )
        else:
            results.append(
                Option(
                    id=strawberry.ID(symbol),
                    contract_symbol=symbol,
                    underlying_symbol=symbol.split("_")[0] if "_" in symbol else symbol,
                    strike=res.get("strike", 100.0),
                    expiry=res.get("expiry", now),
                    option_type=res.get("type", "CALL").upper(),
                    last=res.get("price"),
                    delta=res.get("delta"),
                )
            )
    return results


# Persistent DataLoader for the request context
option_loader = DataLoader(load_fn=_load_options_vectorized)


async def get_option(
    symbol: str, expiry: datetime, strike: float, option_type: str
) -> Option | None:
    """Fetch a single option using coordinates."""
    contract_symbol = f"{symbol}_{expiry.strftime('%Y%m%d')}_{option_type[0].upper()}_{int(strike)}"
    return await get_option_by_id(contract_symbol)


async def get_option_by_id(id: str) -> Option | None:
    """Fetch option by its unique manifold ID."""
    try:
        data = await router.get_live_quote(id)
        return Option(
            id=strawberry.ID(id),
            contract_symbol=id,
            underlying_symbol=id.split("_")[0] if "_" in id else id,
            strike=data.get("strike", 100.0),
            expiry=data.get("expiry", datetime.now()),
            option_type=data.get("type", "CALL").upper(),
            last=data.get("price"),
            delta=data.get("delta"),
        )
    except Exception:
        # Minimal return for federation compatibility
        return Option(
            id=strawberry.ID(id),
            contract_symbol=id,
            underlying_symbol=id.split("_")[0] if "_" in id else id,
            strike=100.0,
            expiry=datetime.now(),
            option_type="CALL",
        )


async def search_options_paginated(
    underlying: str,
    min_strike: float | None = None,
    max_strike: float | None = None,
    expiry: datetime | None = None,
    expiry_bucket: str | None = None,
    limit: int = 100,
    cursor: str | None = None,
) -> tuple[list[Option], bool, str | None]:
    """Search for options with cursor-based pagination."""
    # Handle expiry bucket mapping
    if expiry_bucket and expiry_bucket != "all":
        from datetime import date, timedelta

        today = date.today()
        if expiry_bucket == "week":
            expiry = datetime.combine(today + timedelta(days=7), datetime.min.time())
        elif expiry_bucket == "month":
            expiry = datetime.combine(today + timedelta(days=30), datetime.min.time())
        elif expiry_bucket == "quarter":
            expiry = datetime.combine(today + timedelta(days=90), datetime.min.time())

    # Fetch data
    raw_chain = await router.get_option_chain_snapshot(underlying)

    # Filter
    filtered = []
    for contract in raw_chain:
        if min_strike and contract["strike"] < min_strike:
            continue
        if max_strike and contract["strike"] > max_strike:
            continue

        # If expiry filter is active, only show that specific date or closer
        if expiry:
            contract_exp = (
                datetime.fromisoformat(contract["expiry"])
                if isinstance(contract["expiry"], str)
                else contract["expiry"]
            )
            if contract_exp.date() > expiry.date():
                continue

        filtered.append(contract)

    # Sort for deterministic pagination
    filtered.sort(key=lambda x: x["symbol"])

    # Apply cursor
    start_idx = 0
    if cursor:
        for i, contract in enumerate(filtered):
            if contract["symbol"] == cursor:
                start_idx = i + 1
                break

    # Slice
    paged = filtered[start_idx : start_idx + limit]
    has_next = len(filtered) > (start_idx + limit)
    next_cursor = paged[-1]["symbol"] if paged else None

    results = [
        Option(
            id=strawberry.ID(contract["symbol"]),
            contract_symbol=contract["symbol"],
            underlying_symbol=underlying,
            strike=contract["strike"],
            expiry=(
                datetime.fromisoformat(contract["expiry"])
                if isinstance(contract["expiry"], str)
                else contract["expiry"]
            ),
            option_type=contract["type"].upper(),
            last=contract["price"],
        )
        for contract in paged
    ]

    return results, has_next, next_cursor
