import uuid
from datetime import datetime

from pydantic import BaseModel, Field


class PortfolioBase(BaseModel):
    name: str = Field(..., example="My Main Portfolio")
    cash: float = Field(..., example=10000.0)

class PortfolioCreate(PortfolioBase):
    pass

class PortfolioUpdate(BaseModel):
    name: str | None = Field(None, example="Updated Portfolio Name")
    cash: float | None = Field(None, example=10500.50)

class Portfolio(PortfolioBase):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    created_at: datetime
    updated_at: datetime | None = None

    class Config:
        orm_mode = True # For Pydantic V1 compatibility, use from_attributes=True for V2

