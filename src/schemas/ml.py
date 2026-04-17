from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
import uuid

class MLModelBase(BaseModel):
    name: str = Field(..., example="SentimentAnalyzer")
    version: str = Field(..., example="1.0.0")
    description: Optional[str] = Field(None, example="Analyzes text sentiment")
    is_active: bool = Field(default=True)

class MLModelCreate(MLModelBase):
    pass

class MLModelUpdate(BaseModel):
    description: Optional[str] = Field(None, example="Updated description")
    is_active: Optional[bool] = Field(None, example=False)

class MLModel(MLModelBase):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime

    class Config:
        orm_mode = True # For Pydantic V1 compatibility, use from_attributes=True for V2
