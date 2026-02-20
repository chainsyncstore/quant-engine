
import datetime
from sqlalchemy import Column, Integer, String, Boolean, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    telegram_id = Column(Integer, primary_key=True)
    username = Column(String)
    role = Column(String, default='user') # admin, user
    status = Column(String, default='pending') # pending, approved, banned
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    context = relationship("UserContext", back_populates="user", uselist=False, cascade="all, delete-orphan")

class UserContext(Base):
    __tablename__ = 'user_context'
    telegram_id = Column(Integer, ForeignKey('users.telegram_id'), primary_key=True)
    capital_email = Column(String)
    capital_api_key = Column(String) # Encrypted
    capital_password = Column(String) # Encrypted
    live_mode = Column(Boolean, default=False)
    is_active = Column(Boolean, default=False)
    
    user = relationship("User", back_populates="context")
