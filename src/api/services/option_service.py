from typing import List, Optional
from datetime import datetime
import strawberry
from src.data.router import MarketDataRouter

router = MarketDataRouter()

@strawberry.type
class Option:
    id: strawberry.ID
    contract_symbol: str
    underlying_symbol: str
    strike: float
    expiry: datetime
    option_type: str
    bid: Optional[float] = None
    ask: Optional[float] = None
    last: Optional[float] = None
    delta: Optional[float] = None
    gamma: Optional[float] = None
    vega: Optional[float] = None
    theta: Optional[float] = None
    rho: Optional[float] = None
    predicted_price: Optional[float] = None
    confidence: Optional[float] = None
    volume: Optional[int] = None
    open_interest: Optional[int] = None
    implied_volatility: Optional[float] = None

async def get_option(contract_symbol: str) -> Optional[Option]:
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

async def search_options(
    underlying: str,
    min_strike: Optional[float] = None,
    max_strike: Optional[float] = None,
    expiry: Optional[datetime] = None,
    limit: int = 100
) -> List[Option]:
    """Search for options using the MarketDataRouter."""
    raw_chain = await router.get_option_chain_snapshot(underlying)
    
    options = []
    for contract in raw_chain[:limit]:
        # Filter logic
        if min_strike and contract["strike"] < min_strike: continue
        if max_strike and contract["strike"] > max_strike: continue
        
        options.append(Option(
            id=strawberry.ID(contract["symbol"]),
            contract_symbol=contract["symbol"],
            underlying_symbol=underlying,
            strike=contract["strike"],
            expiry=datetime.fromisoformat(contract["expiry"]) if isinstance(contract["expiry"], str) else contract["expiry"],
            option_type=contract["type"].upper(),
            last=contract["price"]
        ))
    
    return options
