import uuid
from datetime import datetime

from pydantic import BaseModel, Field


class MLModelBase(BaseModel):
    name: str = Field(..., example="SentimentAnalyzer")
    version: str = Field(..., example="1.0.0")
    description: str | None = Field(None, example="Analyzes text sentiment")
    is_active: bool = Field(default=True)

class MLModelCreate(MLModelBase):
    pass

class MLModelUpdate(BaseModel):
    description: str | None = Field(None, example="Updated description")
    is_active: bool | None = Field(None, example=False)

class MLModel(MLModelBase):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime

    class Config:
        orm_mode = True # For Pydantic V1 compatibility, use from_attributes=True for V2
