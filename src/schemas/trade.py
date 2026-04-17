from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
import uuid

class TradeBase(BaseModel):
    symbol: str = Field(..., example="AAPL")
    quantity: float = Field(..., example=10.0)
    price: float = Field(..., example=150.50)
    side: str = Field(..., example="buy") # 'buy' or 'sell'
    order_type: str = Field(..., example="market") # e.g., 'market', 'limit'

class TradeCreate(TradeBase):
    portfolio_id: str

class TradeUpdate(BaseModel):
    # Define fields that can be updated, e.g., status
    status: Optional[str] = Field(None, example="filled")

class Trade(TradeBase):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    portfolio_id: str
    status: str = Field(default="pending")
    timestamp: datetime

    class Config:
        orm_mode = True # For Pydantic V1 compatibility, use from_attributes=True for V2
