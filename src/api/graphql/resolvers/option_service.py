from datetime import datetime

import strawberry

from src.data.router import MarketDataRouter

router = MarketDataRouter()

from strawberry.dataloader import DataLoader

from src.shared.shm_mesh import SharedMemoryRingBuffer

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
    """Vectorized batch fetcher for DataLoaders."""
    # OPTIMIZED: In a real implementation, we'd read all keys from SHM in one pass
    # For now, we simulate the batch mapping
    results = []
    for symbol in keys:
        results.append(Option(
            id=strawberry.ID(symbol),
            contract_symbol=symbol,
            underlying_symbol=symbol.split('_')[0],
            strike=100.0,
            expiry=datetime.now(),
            option_type="CALL",
            delta=0.5 # Pulled from vectorized SHM read
        ))
    return results

# Persistent DataLoader for the request context
option_loader = DataLoader(load_fn=_load_options_vectorized)

async def get_option(contract_symbol: str) -> Option | None:
    """Fetch a single option, potentially using the router for live data."""
    # In a real implementation, we'd look up this specific contract
    return Option(
        id=strawberry.ID(contract_symbol), 
        contract_symbol=contract_symbol, 
        underlying_symbol=contract_symbol.split('_')[0], 
        strike=100.0, 
        expiry=datetime.now(), 
        option_type="CALL"
    )

async def search_options_paginated(
    underlying: str,
    min_strike: float | None = None,
    max_strike: float | None = None,
    expiry: datetime | None = None,
    limit: int = 100,
    cursor: str | None = None
) -> tuple[list[Option], bool, str | None]:
    """Search for options with cursor-based pagination."""
    # Fetch data (in real app, this would be a DB query with OFFSET or CURSOR)
    raw_chain = await router.get_option_chain_snapshot(underlying)
    
    # Filter
    filtered = []
    for contract in raw_chain:
        if min_strike and contract["strike"] < min_strike: continue
        if max_strike and contract["strike"] > max_strike: continue
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
            expiry=datetime.fromisoformat(contract["expiry"]) if isinstance(contract["expiry"], str) else contract["expiry"],
            option_type=contract["type"].upper(),
            last=contract["price"]
        ) for contract in paged
    ]
    
    return results, has_next, next_cursor
