import asyncio
from datetime import date, datetime
from typing import Any, cast

import strawberry
from strawberry.dataloader import DataLoader

from services.api.graphql.types import Option
from services.data.router import MarketDataRouter
from services.shared.shm_mesh import GreeksMesh, SharedMemoryRingBuffer

router = MarketDataRouter()

# Singleton SHM readers
_shm_reader = SharedMemoryRingBuffer(create=False)
_greeks_mesh = GreeksMesh(create=False)


async def _load_options_vectorized(keys: list[str]) -> list[Option]:
    """Vectorized batch fetcher for DataLoaders using Speculative Concurrency."""
    # 1. Dispatch all fetches concurrently
    tasks = [router.get_live_quote(symbol) for symbol in keys]
    results_raw = await asyncio.gather(*tasks, return_exceptions=True)

    now = datetime.now()
    results = []

    for i, symbol in enumerate(keys):
        res = results_raw[i]

        # Type-safe check for exceptions or error dictionaries
        is_error = isinstance(res, Exception) or (isinstance(res, dict) and "error" in res)

        if is_error:
            # Fallback to minimal object
            results.append(
                Option(
                    id=strawberry.ID(symbol),
                    symbol=symbol,
                    strike=100.0,
                    expiry=now.date(),
                    option_type="CALL",
                    time=now,
                )
            )
        else:
            # At this point, res is guaranteed to be a successful dict
            res_dict = cast(dict[str, Any], res)

            exp_val = res_dict.get("expiry", now)
            exp_date = (
                exp_val.date()
                if isinstance(exp_val, datetime)
                else (
                    datetime.fromisoformat(exp_val).date()
                    if isinstance(exp_val, str)
                    else cast(date, exp_val)
                )
            )

            opt = Option(
                id=strawberry.ID(symbol),
                symbol=symbol,
                strike=float(res_dict.get("strike", 100.0)),
                expiry=exp_date,
                option_type=str(res_dict.get("type", "CALL")).upper(),
                last=res_dict.get("price"),
                time=cast(datetime, res_dict.get("time", now)),
            )

            # Enrich with real-time SHM Greeks
            shm_greeks = _greeks_mesh.read(symbol)
            if shm_greeks:
                opt.delta = shm_greeks["delta"]
                opt.gamma = shm_greeks["gamma"]
                opt.theta = shm_greeks["theta"]
                opt.vega = shm_greeks["vega"]
                opt.rho = shm_greeks["rho"]
            else:
                # Fallback to provider Greeks if available
                opt.delta = res_dict.get("delta")
                opt.gamma = res_dict.get("gamma")
                opt.theta = res_dict.get("theta")
                opt.vega = res_dict.get("vega")
                opt.rho = res_dict.get("rho")

            results.append(opt)
    return results


# Persistent DataLoader for the request context
option_loader = DataLoader(load_fn=_load_options_vectorized)


async def get_option(
    symbol: str, expiry: date | datetime, strike: float, option_type: str
) -> Option | None:
    """Fetch a single option using coordinates."""
    expiry_str = (
        expiry.strftime("%Y%m%d") if hasattr(expiry, "strftime") else str(expiry).replace("-", "")
    )
    contract_symbol = f"{symbol}_{expiry_str}_{option_type[0].upper()}_{int(strike)}"
    return await get_option_by_id(contract_symbol)


async def get_option_by_id(id: str) -> Option | None:
    """Fetch option by its unique manifold ID."""
    try:
        data = cast(dict[str, Any], await router.get_live_quote(id))

        if "error" in data:
            raise RuntimeError(data["error"])

        now = datetime.now()
        exp_val = data.get("expiry", now)
        exp_date = (
            exp_val.date()
            if isinstance(exp_val, datetime)
            else (
                datetime.fromisoformat(exp_val).date()
                if isinstance(exp_val, str)
                else cast(date, exp_val)
            )
        )

        opt = Option(
            id=strawberry.ID(id),
            symbol=id,
            strike=float(data.get("strike", 100.0)),
            expiry=exp_date,
            option_type=str(data.get("type", "CALL")).upper(),
            last=data.get("price"),
            time=cast(datetime, data.get("time", now)),
        )

        # Enrich with real-time SHM Greeks
        shm_greeks = _greeks_mesh.read(id)
        if shm_greeks:
            opt.delta = shm_greeks["delta"]
            opt.gamma = shm_greeks["gamma"]
            opt.theta = shm_greeks["theta"]
            opt.vega = shm_greeks["vega"]
            opt.rho = shm_greeks["rho"]
        else:
            opt.delta = data.get("delta")
            opt.gamma = data.get("gamma")
            opt.theta = data.get("theta")
            opt.vega = data.get("vega")
            opt.rho = data.get("rho")

        return opt
    except Exception:
        # Minimal return for federation compatibility
        now = datetime.now()
        return Option(
            id=strawberry.ID(id),
            symbol=id,
            strike=100.0,
            expiry=now.date(),
            option_type="CALL",
            time=now,
        )


async def search_options_paginated(
    underlying: str,
    min_strike: float | None = None,
    max_strike: float | None = None,
    expiry: date | datetime | None = None,
    expiry_bucket: str | None = None,
    limit: int = 100,
    cursor: str | None = None,
) -> tuple[list[Option], bool, str | None]:
    """Search for options with cursor-based pagination."""
    # Handle expiry bucket mapping
    if expiry_bucket and expiry_bucket != "all":
        from datetime import timedelta

        today = date.today()
        if expiry_bucket == "week":
            expiry = today + timedelta(days=7)
        elif expiry_bucket == "month":
            expiry = today + timedelta(days=30)
        elif expiry_bucket == "quarter":
            expiry = today + timedelta(days=90)

    # Fetch data
    raw_chain = cast(list[dict[str, Any]], await router.get_option_chain_snapshot(underlying))

    # Filter
    filtered = []
    for contract in raw_chain:
        if min_strike and contract["strike"] < min_strike:
            continue
        if max_strike and contract["strike"] > max_strike:
            continue

        # If expiry filter is active, only show that specific date or closer
        if expiry:
            expiry_date = expiry if isinstance(expiry, date) else expiry.date()
            contract_exp = (
                datetime.fromisoformat(cast(str, contract["expiry"])).date()
                if isinstance(contract["expiry"], str)
                else (
                    contract["expiry"].date()
                    if isinstance(contract["expiry"], datetime)
                    else cast(date, contract["expiry"])
                )
            )
            if contract_exp > expiry_date:
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
    next_cursor = cast(str, paged[-1]["symbol"]) if paged else None

    results = []
    now = datetime.now()
    for contract in paged:
        symbol = cast(str, contract["symbol"])
        exp_val = contract["expiry"]
        exp_date = (
            exp_val.date()
            if isinstance(exp_val, datetime)
            else (
                datetime.fromisoformat(exp_val).date()
                if isinstance(exp_val, str)
                else cast(date, exp_val)
            )
        )

        opt = Option(
            id=strawberry.ID(symbol),
            symbol=symbol,
            strike=float(contract["strike"]),
            expiry=exp_date,
            option_type=str(contract["type"]).upper(),
            last=contract["price"],
            time=cast(datetime, contract.get("time", now)),
        )

        # Enrich with real-time SHM Greeks
        shm_greeks = _greeks_mesh.read(symbol)
        if shm_greeks:
            opt.delta = shm_greeks["delta"]
            opt.gamma = shm_greeks["gamma"]
            opt.theta = shm_greeks["theta"]
            opt.vega = shm_greeks["vega"]
            opt.rho = shm_greeks["rho"]
        else:
            opt.delta = contract.get("delta")
            opt.gamma = contract.get("gamma")
            opt.theta = contract.get("theta")
            opt.vega = contract.get("vega")
            opt.rho = contract.get("rho")

        results.append(opt)

    return results, has_next, next_cursor
