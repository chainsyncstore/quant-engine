import logging
from typing import Any, Callable, Dict, List, Optional, Literal

from data.schemas import Bar
from hypotheses.base import TradeIntent, IntentType
from execution.simulator import ExecutionSimulator, CompletedTrade
from execution.cost_model import CostModel, CostSide
from state.market_state import MarketState
from state.position_state import PositionState, PositionSide
from portfolio.models import PortfolioState, PortfolioAllocation
from portfolio.ensemble import Ensemble
from portfolio.risk import (
    RiskRule,
    MaxDrawdownRule,
    TradeThrottle,
    LossStreakGuard,
    ExecutionPolicyRule,
)
from portfolio.risk_scaling import RiskTierResolver
from clock.clock import Clock
from config.execution_flags import EXECUTION_ENABLED
from config.competition_flags import COMPETITION_MODE
from execution_live.order_models import ExecutionIntent, IntentAction
from engine.decision_queue import QueuedDecision
from market.regime import RegimeClassifier, RegimeConfidence

logger = logging.getLogger(__name__)

META_ALLOCATION_KEY = "META_PORTFOLIO"

ExecutionIntentSink = Callable[[ExecutionIntent], None]

class MetaExecutionSimulator(ExecutionSimulator):
    """
    Simulator that interprets intent.size as absolute UNITS (shares/contracts).
    Used for rebalancing meta-portfolio.
    """
    def _execute_entry(
        self,
        side: PositionSide,
        execution_bar: Bar,
        decision: QueuedDecision,
        position_state: PositionState,
        size: float
    ) -> Optional[CompletedTrade]:
        # Size here is treated as UNITS
        target_units = size
        if target_units <= 0:
            return None
            
        # Check capital
        base_price = execution_bar.open
        cost_side = CostSide.BUY if side == PositionSide.LONG else CostSide.SELL
        effective_price = self._cost_model.apply_costs(base_price, cost_side)
        
        required_capital = target_units * effective_price
        
        if required_capital > self._available_capital:
            # Clip to available
            if effective_price > 0:
                target_units = self._available_capital / effective_price
                required_capital = target_units * effective_price
            else:
                target_units = 0
                required_capital = 0
            
        total_cost = self._cost_model.calculate_cost_amount(
            base_price, target_units, cost_side
        )
        
        position_state.open_position(
            side=side,
            entry_price=effective_price,
            size=target_units,
            entry_timestamp=execution_bar.timestamp,
            entry_capital=required_capital
        )
        
        self._available_capital -= required_capital
        
        trade = CompletedTrade(
            trade_type="ENTRY",
            side=side.value,
            execution_price=effective_price,
            size=target_units,
            execution_timestamp=execution_bar.timestamp,
            decision_timestamp=decision.decision_timestamp,
            cost_bps=self._cost_model.get_total_cost_bps(),
            total_cost=total_cost
        )
        self._completed_trades.append(trade)
        return trade

class MetaPortfolioEngine:
    """
    Orchestrates Dual-Track Simulation:
    1. Shadow Track: Simulates individual hypotheses (Virtual PnL).
    2. Meta Track: Simulates weighted net portfolio (Real PnL).
    """
    def __init__(
        self,
        ensemble: Ensemble,
        initial_capital: float,
        cost_model: CostModel,
        risk_rules: Optional[List[RiskRule]] = None,
        decay_check_interval: int = 0,
        symbol: str = "SYNTHETIC",
        execution_intent_sink: Optional[ExecutionIntentSink] = None,
        execution_mode: Literal["PAPER", "LIVE"] = "PAPER",
        risk_tier_resolver: Optional[RiskTierResolver] = None,
        telemetry: Optional[Callable[[str, dict], None]] = None,
        explain_decisions: bool = False,
        rotation_symbols: Optional[List[str]] = None,
    ):
        self.ensemble = ensemble
        self.initial_capital = initial_capital
        self.cost_model = cost_model
        self.risk_rules = list(risk_rules) if risk_rules else []
        self.decay_check_interval = decay_check_interval
        self.symbol = symbol
        self._execution_intent_sink = execution_intent_sink
        self.execution_mode = execution_mode
        self.risk_tier_resolver = risk_tier_resolver or RiskTierResolver()
        self.telemetry = telemetry
        self.explain_decisions = explain_decisions
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Crypto rotation for competition mode
        self.rotation_symbols = rotation_symbols or []
        self.active_symbol_index = 0
        
        # 1. Shadow Track Initialization
        # We give each shadow sim a hypothetical capital (e.g. 1M) just to track % returns and positions accurately.
        # It doesn't affect the Meta capital.
        SHADOW_CAP = 1_000_000.0 
        self.shadow_simulators: Dict[str, ExecutionSimulator] = {}
        self.shadow_position_states: Dict[str, PositionState] = {}
        
        for h in ensemble.hypotheses:
            hid = h.hypothesis_id
            self.shadow_simulators[hid] = ExecutionSimulator(cost_model, SHADOW_CAP)
            self.shadow_position_states[hid] = PositionState()
            
        # 2. Meta Track Initialization
        self.meta_simulator = MetaExecutionSimulator(cost_model, initial_capital)
        self.meta_position_state = PositionState()
        
        # Globals
        self.clock = Clock()
        # Shared Market State
        # Must be large enough for Regime Detection (SMA200 requires >200 bars)
        self.market_state = MarketState(lookback_window=300)
        self.regime_classifier = RegimeClassifier()
        
        # Per-symbol market states for multi-symbol rotation
        self._symbol_market_states: Dict[str, MarketState] = {}
        if self.rotation_symbols:
            for sym in self.rotation_symbols:
                self._symbol_market_states[sym] = MarketState(lookback_window=300)

    def run(self, bars: List[Bar]) -> List[PortfolioState]:
        history: List[PortfolioState] = []
        peak_equity = self.initial_capital
        
        # Keep track of shadow equity curves for decay calculation
        shadow_equity_curves: Dict[str, List[float]] = {h.hypothesis_id: [] for h in self.ensemble.hypotheses}
        
        for bar_idx, bar in enumerate(bars):
            # Multi-symbol: update the correct symbol's market state
            bar_symbol = getattr(bar, 'symbol', None)
            if bar_symbol and bar_symbol in self._symbol_market_states:
                self._symbol_market_states[bar_symbol].update(bar)
            
            # In competition mode with rotation, skip bars for other symbols
            if COMPETITION_MODE and self.rotation_symbols and bar_symbol:
                if bar_symbol != self.symbol:
                    continue  # Skip this bar, it's for a different symbol
            
            self.clock.set_time(bar.timestamp)
            # Use per-symbol market state if available, else shared
            if self.symbol in self._symbol_market_states:
                self.market_state = self._symbol_market_states[self.symbol]
            else:
                self.market_state.update(bar)
            if self.explain_decisions:
                self.logger.info(
                    "bar_evaluated | ts=%s | hypotheses=%s",
                    bar.timestamp,
                    [
                        getattr(h, "name", getattr(h, "hypothesis_id", h.__class__.__name__))
                        for h in self.ensemble.hypotheses
                    ],
                )
            
            # --- Decay Check ---
            if self.decay_check_interval > 0 and bar_idx > 0 and bar_idx % self.decay_check_interval == 0:
                self._check_decay(shadow_equity_curves)
            
            # --- A. Shadow Track Execution ---
            
            # 1. Generate Intents from Hypotheses
            shadow_intents: Dict[str, TradeIntent] = {}
            for h in self.ensemble.hypotheses:
                intent = h.on_bar(
                    self.market_state,
                    self.shadow_position_states[h.hypothesis_id],
                    self.clock
                )
                if intent:
                    shadow_intents[h.hypothesis_id] = intent
                    
            # 2. Execute in Shadow Simulators
            from engine.decision_queue import QueuedDecision
            for h in self.ensemble.hypotheses:
                hid = h.hypothesis_id
                if hid in shadow_intents:
                    decision = QueuedDecision(
                        intent=shadow_intents[hid],
                        decision_timestamp=bar.timestamp,
                        decision_bar_index=bar_idx
                    )
                    self.shadow_simulators[hid].execute_decisions(
                        [decision], bar, self.shadow_position_states[hid]
                    )

            pre_trade_snapshot = self._create_snapshot(bar, peak_equity)

            # --- Deterministic Exit Check (Competition Mode) ---
            if COMPETITION_MODE and self.meta_position_state.has_position:
                exit_now, exit_reason = self._should_exit(self.meta_position_state.position, bar)
                if exit_now:
                    pos = self.meta_position_state.position
                    close_intent = TradeIntent(type=IntentType.CLOSE, size=1.0)
                    close_decision = QueuedDecision(
                        intent=close_intent,
                        decision_timestamp=bar.timestamp,
                        decision_bar_index=bar_idx
                    )
                    self.meta_simulator.execute_decisions([close_decision], bar, self.meta_position_state)
                    
                    exec_intent = self._build_execution_intent(
                        action=IntentAction.CLOSE,
                        quantity=pos.size,
                        bar=bar,
                        bar_idx=bar_idx,
                        metadata={"reason": exit_reason},
                    )
                    self._publish_execution_intents([exec_intent])
                    
                    logger.info(
                        "[COMPETITION] Deterministic exit | reason=%s | pnl_pct=%.4f",
                        exit_reason,
                        (bar.close - pos.entry_price) / pos.entry_price,
                    )
                    
                    # Rotate symbol after exit
                    if self.rotation_symbols:
                        self.symbol = self._next_symbol()
                    
                    # Skip rest of decision loop for this bar
                    snapshot = self._create_snapshot(bar, peak_equity)
                    if snapshot.total_capital > peak_equity:
                        peak_equity = snapshot.total_capital
                    history.append(snapshot)
                    for hid, alloc in snapshot.allocations.items():
                        if hid != "META_PORTFOLIO":
                            shadow_equity_curves[hid].append(alloc.allocated_capital)
                    continue

            # --- B. Meta Track Execution ---
            
            # 3. Calculate Target Net Exposure
            # Regime Gating: If regime mismatch, exclude from Net Exposure.
            current_regime, regime_confidence = self.regime_classifier.classify_with_confidence(
                self.market_state
            )
            risk_tier = self.risk_tier_resolver.resolve(regime_confidence)
            net_exposure_target = 0.0
        
            for h in self.ensemble.hypotheses:
                hid = h.hypothesis_id
            
                # Check Regime - bypass in competition mode for UNKNOWN confidence
                regime_bypass = COMPETITION_MODE and regime_confidence == RegimeConfidence.UNKNOWN
                if h.allowed_regimes and current_regime not in h.allowed_regimes and not regime_bypass:
                    self._emit_decision_block_event(
                        reason="regime_unfavorable",
                        bar=bar,
                        extra={
                            "hypothesis_id": hid,
                            "current_regime": getattr(current_regime, "value", str(current_regime)),
                            "allowed_regimes": [getattr(reg, "value", str(reg)) for reg in h.allowed_regimes],
                            "bar_index": bar_idx,
                        },
                    )
                    continue
                
                if regime_bypass:
                    logger.info(
                        "[COMPETITION] Regime bypass active | hypothesis=%s regime_confidence=%s",
                        hid,
                        regime_confidence.value,
                    )
                
                weight = self.ensemble.weights.get(hid, 0.0)
                
                # COMPETITION FIX: Use current bar's signal directly instead of shadow position
                # This bypasses the stateless shadow tracking issue across process invocations
                sign = 0.0
                if hid in shadow_intents:
                    intent = shadow_intents[hid]
                    if intent.type == IntentType.BUY:
                        sign = 1.0
                    elif intent.type == IntentType.SELL:
                        sign = -1.0
                    # CLOSE intents don't contribute to exposure
                else:
                    # Fallback: check shadow position for multi-bar holds
                    pos_state = self.shadow_position_states[hid]
                    if pos_state.has_position:
                        if pos_state.position.side == PositionSide.LONG:
                            sign = 1.0
                        else:
                            sign = -1.0
            
                net_exposure_target += (weight * sign)
        
            # Determine target meta exposure measured in units using regime-aware risk fractions
            curr_equity = self.meta_simulator.get_total_capital(bar.open, self.meta_position_state)
            allocation_view = self._build_meta_allocation(bar, curr_equity)
            exposure_ratio = abs(net_exposure_target)
            risk_fraction = max(0.0, min(1.0, risk_tier.risk_fraction))
            target_value = curr_equity * exposure_ratio * risk_fraction
            max_notional = curr_equity * self.risk_tier_resolver.max_leverage
            if target_value > max_notional:
                target_value = max_notional
            # For high-priced assets like crypto, allow fractional units
            target_units = round(target_value / bar.open, 4) if bar.open > 0 else 0.0

            if target_units == 0 and exposure_ratio > 0 and risk_fraction == 0:
                self._emit_decision_block_event(
                    reason="confidence_below_threshold",
                    bar=bar,
                    extra={
                        "regime_confidence": getattr(regime_confidence, "value", str(regime_confidence)),
                        "exposure_ratio": exposure_ratio,
                        "risk_fraction": risk_fraction,
                        "bar_index": bar_idx,
                    },
                )

            target_side = None
            if target_units > 0:
                target_side = PositionSide.LONG if net_exposure_target > 0 else PositionSide.SHORT

                current_side = None
                current_units: float = 0.0
                if self.meta_position_state.has_position:
                    pos = self.meta_position_state.position
                    current_side = pos.side
                    current_units = pos.size

                # Rebalancing Logic (Close & Re-Open Strategy)
                decisions = []
                emitted_intents: List[ExecutionIntent] = []

                risk_metadata = {
                    "risk_label": risk_tier.label,
                    "risk_fraction": risk_fraction,
                    "regime_confidence": regime_confidence.value,
                }

                def enqueue_decision(
                    intent: TradeIntent,
                    action: IntentAction,
                    quantity: float,
                    metadata: Dict[str, Any],
                ) -> bool:
                    if quantity <= 0:
                        return False
                    if not self._is_trade_allowed(intent, allocation_view, pre_trade_snapshot, bar, bar_idx):
                        return False
                    decisions.append(QueuedDecision(
                        intent=intent,
                        decision_timestamp=bar.timestamp,
                        decision_bar_index=bar_idx
                    ))
                    emitted_intents.append(
                        self._build_execution_intent(
                            action=action,
                            quantity=quantity,
                            bar=bar,
                            bar_idx=bar_idx,
                            metadata=metadata,
                        )
                    )
                    return True

                # Case 1: Switch Side or Go Flat
                if current_side and (current_side != target_side or target_units == 0):
                    size_to_close = abs(current_units)
                    if enqueue_decision(
                        intent=TradeIntent(type=IntentType.CLOSE, size=1.0),
                        action=IntentAction.CLOSE,
                        quantity=size_to_close,
                        metadata={
                            "reason": "side_change_or_flat",
                            **risk_metadata,
                        },
                    ):
                        current_units = 0
                        current_side = None
                        # Rotate symbol after close in competition mode
                        if COMPETITION_MODE and self.rotation_symbols:
                            self.symbol = self._next_symbol()

                # Case 2: Size Change (Same Side)
                if current_side == target_side and current_side is not None and current_units != target_units:
                    size_to_close = abs(current_units)
                    if enqueue_decision(
                        intent=TradeIntent(type=IntentType.CLOSE, size=1.0),
                        action=IntentAction.CLOSE,
                        quantity=size_to_close,
                        metadata={
                            "reason": "resize",
                            **risk_metadata,
                        },
                    ):
                        current_units = 0
                        current_side = None
                        # Rotate symbol after close in competition mode
                        if COMPETITION_MODE and self.rotation_symbols:
                            self.symbol = self._next_symbol()

                # Case 3: Open Target (if not already there)
                if target_side and target_units > 0 and current_units == 0:
                    intent_type = IntentType.BUY if target_side == PositionSide.LONG else IntentType.SELL
                    enqueue_decision(
                        intent=TradeIntent(type=intent_type, size=target_units),  # Size is UNITS for MetaSim
                        action=IntentAction.BUY if intent_type == IntentType.BUY else IntentAction.SELL,
                        quantity=target_units,
                        metadata={
                            "reason": "target_entry",
                            **risk_metadata,
                        },
                    )

                # Execute Meta Decisions
                if decisions:
                    self.meta_simulator.execute_decisions(decisions, bar, self.meta_position_state)
                    self._publish_execution_intents(emitted_intents)

            # Update peak equity
            snapshot = self._create_snapshot(bar, peak_equity)
            if snapshot.total_capital > peak_equity:
                peak_equity = snapshot.total_capital

            history.append(snapshot)
                
            # Update Shadow Equity Curves
            for hid, alloc in snapshot.allocations.items():
                if hid != "META_PORTFOLIO":
                    shadow_equity_curves[hid].append(alloc.allocated_capital)

        return history

    def _publish_execution_intents(self, intents: List[ExecutionIntent]) -> None:
        if not intents or not self._execution_intent_sink:
            return

        for intent in intents:
            if self.execution_mode == "LIVE" and not EXECUTION_ENABLED:
                logger.info(
                    "intent_suppressed | symbol=%s action=%s qty=%.2f reason=%s",
                    intent.symbol,
                    intent.action.value,
                    intent.quantity,
                    "execution_disabled",
                )
                continue

            self._execution_intent_sink(intent)

    def _build_execution_intent(
        self,
        action: IntentAction,
        quantity: float,
        bar: Bar,
        bar_idx: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ExecutionIntent:
        # Crypto vs Forex minimum sizing (0.01 lot minimum for both)
        # Forex: 0.01 lots = 1000 units, Crypto: 0.01 lots = 0.01 BTC/ETH
        is_crypto = self.symbol and self.symbol.upper() in ("BTCUSD", "ETHUSD", "BTCUSDT", "ETHUSDT")
        MIN_UNITS = 0.01 if is_crypto else 1_000.0
        if 0 < quantity < MIN_UNITS:
            logger.info(
                "Quantity %.6f below min units %.6f; bumping to maintain parity",
                quantity,
                MIN_UNITS,
            )
            quantity = MIN_UNITS
        base_metadata = {"bar_index": bar_idx}
        if metadata:
            base_metadata.update(metadata)
        return ExecutionIntent(
            symbol=self.symbol,
            action=action,
            quantity=float(quantity),
            timestamp=bar.timestamp,
            reference_price=bar.open,
            metadata=base_metadata,
        )

    def _check_decay(self, equity_curves: Dict[str, List[float]]):
        """
        Check for decay based on equity curves.
        Simple logic for C3 MVP: Max Drawdown > 25% -> DECAYED.
        """
        from promotion.models import HypothesisStatus
        
        for hid, curve in equity_curves.items():
            if not curve:
                continue
            
            # Calculate Max DD
            peak = -1e9
            max_dd = 0.0
            for val in curve:
                if val > peak:
                    peak = val
                dd = (peak - val) / peak if peak > 0 else 0.0
                if dd > max_dd:
                    max_dd = dd
            
            # Threshold: 25%
            if max_dd > 0.25:
                current = self.ensemble.current_statuses.get(hid)
                if current == HypothesisStatus.PROMOTED:
                    logger.info(f"Dynamic Decay Triggered for {hid} (DD={max_dd:.2%}). Demoting to DECAYED.")
                    self.ensemble.set_status(hid, HypothesisStatus.DECAYED)

    def _create_snapshot(self, bar: Bar, peak_equity: float) -> PortfolioState:
        # 1. Shadow Allocations (Virtual)
        allocations: Dict[str, PortfolioAllocation] = {}
        for hid, sim in self.shadow_simulators.items():
            pos_state = self.shadow_position_states[hid]
            cap = sim.get_total_capital(bar.close, pos_state)
            unreal = pos_state.get_unrealized_pnl(bar.close) if pos_state.has_position else 0.0
            allocations[hid] = PortfolioAllocation(
                hypothesis_id=hid,
                allocated_capital=float(cap),
                current_position=pos_state.position if pos_state.has_position else None,
                unrealized_pnl=float(unreal),
                realized_pnl=float(cap - sim._initial_capital - unreal)
            )
            
        # 2. Meta Portfolio (Real)
        total_cap = float(self.meta_simulator.get_total_capital(bar.close, self.meta_position_state))
        total_cash = float(self.meta_simulator.get_available_capital())
        total_unreal = 0.0
        if self.meta_position_state.has_position:
            total_unreal = self.meta_position_state.get_unrealized_pnl(bar.close)
            
        realized = total_cap - self.initial_capital - total_unreal
        
        drawdown = 0.0
        if peak_equity > 0:
            drawdown = max(0.0, (peak_equity - total_cap) / peak_equity * 100.0)

        # For MVP, we reuse PortfolioState. 
        # Ideally we'd add "Meta Position" to it.
        # We can stuff meta position info into a special "META" allocation key?
        allocations["META_PORTFOLIO"] = PortfolioAllocation(
            hypothesis_id="META",
            allocated_capital=float(total_cap),
            current_position=self.meta_position_state.position if self.meta_position_state.has_position else None,
            unrealized_pnl=float(total_unreal),
            realized_pnl=float(realized)
        )

        return PortfolioState(
            timestamp=bar.timestamp,
            total_capital=total_cap,
            cash=total_cash,
            allocations=allocations,
            total_realized_pnl=realized,
            total_unrealized_pnl=total_unreal,
            drawdown_pct=drawdown
        )

    def _build_meta_allocation(self, bar: Bar, curr_equity: float) -> PortfolioAllocation:
        unrealized = 0.0
        if self.meta_position_state.has_position:
            unrealized = self.meta_position_state.get_unrealized_pnl(bar.open)
        realized = curr_equity - self.initial_capital - unrealized
        return PortfolioAllocation(
            hypothesis_id=META_ALLOCATION_KEY,
            allocated_capital=float(curr_equity),
            available_capital=float(self.meta_simulator.get_available_capital()),
            symbol=self.symbol,
            reference_price=bar.open,
            current_position=self.meta_position_state.position if self.meta_position_state.has_position else None,
            unrealized_pnl=float(unrealized),
            realized_pnl=float(realized),
        )

    def _is_trade_allowed(
        self,
        intent: TradeIntent,
        allocation: PortfolioAllocation,
        portfolio_state: PortfolioState,
        bar: Bar,
        bar_idx: int,
    ) -> bool:
        if not self.risk_rules:
            return True
        for rule in self.risk_rules:
            can_execute, reason = rule.can_execute(intent, allocation, portfolio_state)
            if can_execute:
                continue
            self._emit_decision_block_event(
                reason=self._map_risk_rule_reason(rule),
                bar=bar,
                extra={
                    "rule": rule.__class__.__name__,
                    "detail": reason,
                    "intent_type": intent.type.value,
                    "bar_index": bar_idx,
                },
            )
            return False
        return True

    def _next_symbol(self) -> str:
        """Rotate to the next symbol in the rotation pool."""
        if not self.rotation_symbols:
            return self.symbol
        self.active_symbol_index = (self.active_symbol_index + 1) % len(self.rotation_symbols)
        new_symbol = self.rotation_symbols[self.active_symbol_index]
        logger.info("[COMPETITION] Symbol rotation: %s -> %s", self.symbol, new_symbol)
        return new_symbol

    def _should_exit(self, position, bar: Bar) -> tuple[bool, Optional[str]]:
        """
        Deterministic exit layer for competition mode.
        Returns (should_exit, reason) tuple.
        """
        if position is None:
            return False, None
        
        pnl_pct = (bar.close - position.entry_price) / position.entry_price
        if position.side == PositionSide.SHORT:
            pnl_pct = -pnl_pct

        if pnl_pct >= 0.01:
            return True, "take_profit"
        if pnl_pct <= -0.006:
            return True, "stop_loss"

        return False, None

    def _map_risk_rule_reason(self, rule: RiskRule) -> str:
        if isinstance(rule, MaxDrawdownRule):
            return "drawdown_guard_active"
        if isinstance(rule, TradeThrottle):
            return "trade_throttled"
        if isinstance(rule, LossStreakGuard):
            return "loss_streak_guard_active"
        if isinstance(rule, ExecutionPolicyRule):
            return "execution_policy_rejected"
        return "risk_rule_blocked"

    def _emit_decision_block_event(
        self,
        reason: str,
        bar: Bar,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self.explain_decisions or not self.telemetry:
            return
        payload: Dict[str, Any] = {
            "reason": reason,
            "symbol": self.symbol,
            "timestamp": bar.timestamp.isoformat(),
        }
        if extra:
            payload.update(extra)
        self.telemetry("decision_blocked", payload)

