
import datetime
from sqlalchemy import Column, Integer, String, Boolean, DateTime, Float, ForeignKey, Text
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
    auto_close_horizon_bars = Column(Integer, default=0)
    stop_loss_pct = Column(Float, default=0.0)
    maintenance_resume_payload = Column(Text)
    maintenance_resume_pending = Column(Boolean, default=False)
    maintenance_post_notified = Column(Boolean, default=False)
    hard_risk_paused = Column(Boolean, default=False)
    hard_risk_pause_reason = Column(Text)
    hard_risk_pause_triggered_at = Column(DateTime)
    hard_risk_pause_breach_type = Column(String)
    hard_risk_pause_details = Column(Text)
    lifetime_demo_pnl_usd = Column(Float, default=0.0)
    lifetime_live_pnl_usd = Column(Float, default=0.0)
    current_demo_equity_usd = Column(Float, default=0.0)
    current_live_equity_usd = Column(Float, default=0.0)
    current_demo_notional_usd = Column(Float, default=0.0)
    current_live_notional_usd = Column(Float, default=0.0)
    current_demo_symbols = Column(Integer, default=0)
    current_live_symbols = Column(Integer, default=0)
    last_demo_equity_usd = Column(Float)
    last_live_equity_usd = Column(Float)
    lifetime_stats_updated_at = Column(DateTime)
    paper_state_json = Column(Text)

    user = relationship("User", back_populates="context")


class ExecutionRouteEvent(Base):
    __tablename__ = "execution_route_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    telegram_id = Column(Integer, index=True, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow, index=True)
    pause_state = Column(String)
    is_active = Column(Boolean)
    live_mode = Column(Boolean, default=False)
    symbol = Column(String, index=True)
    side = Column(String)
    quantity = Column(Float, default=0.0)
    before_position = Column(Float)
    after_position = Column(Float)
    action_class = Column(String, index=True)
    reason = Column(String, index=True)
    accepted = Column(Boolean)
    status = Column(String)
    order_id = Column(String)
    idempotency_key = Column(String, index=True)
    mark_price = Column(Float, default=0.0)
    future_mark_price = Column(Float)
    future_return_bps = Column(Float)
    shadow_evaluated_at = Column(DateTime)
