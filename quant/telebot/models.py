
import datetime
from sqlalchemy import Column, Integer, String, Boolean, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    telegram_id = Column(Integer, primary_key=True)
    username = Column(String)
    role = Column(String, default='user') # admin, user
    status = Column(String, default='pending') # pending, active, banned
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    context = relationship("UserContext", back_populates="user", uselist=False, cascade="all, delete-orphan")

class UserContext(Base):
    __tablename__ = 'user_context'
    telegram_id = Column(Integer, ForeignKey('users.telegram_id'), primary_key=True)
    # Binance credentials (crypto mode)
    binance_api_key = Column(String)
    binance_api_secret = Column(String)
    live_mode = Column(Boolean, default=False)
    is_active = Column(Boolean, default=False)
    strategy_profile = Column(String, default='core_v2')
    active_model_version = Column(String)
    active_model_source = Column(String)

    user = relationship("User", back_populates="context")
