from pydantic import BaseModel
from sqlalchemy import Column, Integer, String, LargeBinary, create_engine
from sqlalchemy.ext.declarative import declarative_base


Base = declarative_base()

class DriverBase(BaseModel):
    driver_name: str
    driver_id: str

class Driver(DriverBase):
    id: int

    class Config:
        from_attributes = True

class DriverModel(Base):
    __tablename__ = 'drivers'

    id = Column(Integer, primary_key=True, index=True)
    driver_name = Column(String, index=True)
    driver_id = Column(String, unique=True, index=True)
    photo = Column(LargeBinary)