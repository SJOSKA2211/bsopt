import strawberry
from strawberry.federation import Schema
from typing import List, Optional
from datetime import datetime
import asyncio

# ============================================================================
# TYPE DEFINITIONS
# ============================================================================
@strawberry.federation.type(keys=["id"])
class Option:
    """Federated Option type - can be referenced from other services"""
    id: strawberry.ID
    contract_symbol: str
    underlying_symbol: str
    strike: float
    expiry: datetime
    option_type: str
    # Pricing data (from Pricing Service)
    bid: Optional[float] = None
    ask: Optional[float] = None
    last: Optional[float] = None
    # Greeks (from Pricing Service)
    delta: Optional[float] = None
    gamma: Optional[float] = None
    vega: Optional[float] = None
    theta: Optional[float] = None
    rho: Optional[float] = None
    # ML predictions (from ML Service)
    predicted_price: Optional[float] = None
    confidence: Optional[float] = None
    # Market data (from Market Data Service)
    volume: Optional[int] = None
    open_interest: Optional[int] = None
    implied_volatility: Optional[float] = None

@strawberry.type
class Portfolio:
    """User portfolio"""
    id: strawberry.ID
    user_id: strawberry.ID
    name: str
    cash_balance: float
    @strawberry.field
    async def positions(self) -> List["Position"]:
        """Fetch positions for this portfolio"""
        # Resolver implementation
        pass
    @strawberry.field
    async def total_value(self) -> float:
        """Calculate total portfolio value"""
        positions = await self.positions()
        # Fetch current prices for all positions
        option_values = await asyncio.gather(*[
            get_option_price(pos.contract_symbol) for pos in positions
        ])
        position_values = sum(
            pos.quantity * price for pos, price in zip(positions, option_values)
        )
        return self.cash_balance + position_values

@strawberry.type
class Position:
    """Option position in portfolio"""
    id: strawberry.ID
    portfolio_id: strawberry.ID
    contract_symbol: str
    quantity: int
    entry_price: float
    entry_date: datetime
    @strawberry.field
    async def option(self) -> "Option":
        """Fetch full option details"""
        # Federation - this calls Option service
        return await get_option_by_symbol(self.contract_symbol)
    @strawberry.field
    async def current_pnl(self) -> float:
        """Calculate current P&L"""
        option = await self.option()
        current_price = option.last or (option.bid + option.ask) / 2
        return self.quantity * (current_price - self.entry_price)

@strawberry.type
class MarketData:
    """Real-time market data"""
    symbol: str
    timestamp: datetime
    bid: float
    ask: float
    last: float
    volume: int
    @strawberry.field
    async def option_chain(self, expiry: Optional[datetime] = None) -> List["Option"]:
        """Get full options chain for underlying"""
        # Fetch from database or cache
        pass

@strawberry.type
class VolatilityPoint:
    """Represents a point on the implied volatility surface"""
    strike: float
    expiry: datetime
    implied_volatility: float

@strawberry.type
class Order:
    """Represents a trading order"""
    id: strawberry.ID
    portfolio_id: strawberry.ID
    contract_symbol: str
    side: str # BUY/SELL
    quantity: int
    order_type: str # LIMIT/MARKET
    status: str # PENDING/FILLED/CANCELLED
    limit_price: Optional[float] = None
    created_at: datetime
    updated_at: datetime

# ============================================================================
# QUERIES
# ============================================================================
@strawberry.type
class Query:
    """Root Query type"""
    @strawberry.field
    async def option(self, contract_symbol: str) -> "Option":
        """Get single option by contract symbol"""
        from src.api.services.option_service import get_option
        return await get_option(contract_symbol)
    @strawberry.field
    async def options(
        self,
        underlying: Optional[str] = None,
        min_strike: Optional[float] = None,
        max_strike: Optional[float] = None,
        expiry: Optional[datetime] = None,
        limit: int = 100
    ) -> List["Option"]:
        """Search options with filters"""
        from src.api.services.option_service import search_options
        return await search_options(
            underlying=underlying,
            min_strike=min_strike,
            max_strike=max_strike,
            expiry=expiry,
            limit=limit
        )
    @strawberry.field
    async def portfolio(self, id: strawberry.ID) -> "Portfolio":
        """Get portfolio by ID"""
        from src.api.services.portfolio_service import get_portfolio
        return await get_portfolio(id)
    @strawberry.field
    async def market_data(
        self,
        symbol: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List["MarketData"]:
        """Get historical market data"""
        from src.api.services.market_data_service import get_market_data
        return await get_market_data(
            symbol=symbol,
            start_time=start_time,
            end_time=end_time
        )
    @strawberry.field
    async def volatility_surface(
        self,
        underlying: str,
        expiry_range: Optional[int] = 90
    ) -> List["VolatilityPoint"]:
        """Get implied volatility surface"""
        from src.api.services.volatility_service import get_vol_surface
        return await get_vol_surface(
            underlying=underlying,
            expiry_range=expiry_range
        )

# ============================================================================
# MUTATIONS
# ============================================================================
@strawberry.type
class Mutation:
    """Root Mutation type"""
    @strawberry.mutation
    async def create_order(
        self,
        portfolio_id: strawberry.ID,
        contract_symbol: str,
        side: str,
        quantity: int,
        order_type: str,
        limit_price: Optional[float] = None
    ) -> "Order":
        """Place option order"""
        from src.api.services.trading_service import create_order
        order = await create_order(
            portfolio_id=portfolio_id,
            contract_symbol=contract_symbol,
            side=side,
            quantity=quantity,
            order_type=order_type,
            limit_price=limit_price
        )
        return order
    @strawberry.mutation
    async def cancel_order(self, order_id: strawberry.ID) -> "Order":
        """Cancel pending order"""
        from src.api.services.trading_service import cancel_order
        return await cancel_order(order_id)
    @strawberry.mutation
    async def create_portfolio(
        self,
        user_id: strawberry.ID,
        name: str,
        initial_cash: float
    ) -> "Portfolio":
        """Create new portfolio"""
        from src.api.services.portfolio_service import create_portfolio
        return await create_portfolio(
            user_id=user_id,
            name=name,
            initial_cash=initial_cash
        )

# ============================================================================
# SUBSCRIPTIONS (Real-time updates)
# ============================================================================
@strawberry.type
class Subscription:
    """Real-time subscriptions via WebSocket"""
    @strawberry.subscription
    async def market_data_stream(
        self,
        symbols: List[str]
    ) -> "MarketData":
        """Subscribe to real-time market data"""
        # Connect to Kafka stream
        from src.streaming.kafka_consumer import MarketDataConsumer
        consumer = MarketDataConsumer(topics=[f"market-data-{s}" for s in symbols])
        async for message in consumer.consume_messages():
            yield MarketData(**message)
    @strawberry.subscription
    async def portfolio_updates(
        self,
        portfolio_id: strawberry.ID
    ) -> "Portfolio":
        """Subscribe to portfolio value updates"""
        while True:
            # Fetch updated portfolio every second
            portfolio = await get_portfolio(portfolio_id)
            yield portfolio
            await asyncio.sleep(1)
    @strawberry.subscription
    async def option_greeks_stream(
        self,
        contract_symbol: str
    ) -> "Option":
        """Subscribe to real-time Greeks updates"""
        from src.streaming.kafka_consumer import MarketDataConsumer
        consumer = MarketDataConsumer(topics=[f"greeks-{contract_symbol}"])
        async for message in consumer.consume_messages():
            yield Option(**message)

# ============================================================================
# APOLLO FEDERATION - Subgraph Schema
# ============================================================================
schema = Schema(
    query=Query,
    mutation=Mutation,
    subscription=Subscription)
