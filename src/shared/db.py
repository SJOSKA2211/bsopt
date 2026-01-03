from sqlalchemy import create_engine, Column, Integer, String, Float, BigInteger
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()

class MarketData(Base):
    """
    SQLAlchemy model for market data stored in TimescaleDB.
    """
    __tablename__ = 'market_data'

    id = Column(Integer, primary_key=True)
    ticker = Column(String, nullable=False, index=True)
    timestamp = Column(BigInteger, nullable=False, index=True) # Unix timestamp in milliseconds
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)

def get_db_session(connection_string: str):
    """
    Creates a new database session.
    
    Args:
        connection_string: SQLAlchemy connection string.
        
    Returns:
        SQLAlchemy Session object.
    """
    engine = create_engine(connection_string)
    Session = sessionmaker(bind=engine)
    return Session()
