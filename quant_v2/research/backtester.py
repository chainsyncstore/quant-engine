"""Walk-forward backtester that replays through the exact live allocation chain.

Replays historical bars bar-by-bar, builds features, predicts, allocates,
and simulates fills — using the same code path as production.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterator

import pandas as pd
import numpy as np

from quant_v2.execution.cost_policy import ExecutionCostPolicy

logger = logging.getLogger(__name__)


@dataclass
class Fill:
    timestamp: datetime
    symbol: str
    side: str        # "BUY" or "SELL"
    quantity: float
    price: float
    fee_usd: float
    slippage_usd: float
    confidence: float
    policy_version: str = ""
    cost_breakdown: dict[str, float] = field(default_factory=dict)


@dataclass
class BacktestConfig:
    symbol: str = "BTCUSDT"
    start_date: str = "2024-01-01"
    end_date: str = "2024-12-31"
    initial_equity: float = 300.0
    max_symbol_exposure_frac: float = 0.15
    min_confidence: float = 0.65
    maker_fee_bps: float = 2.0
    taker_fee_bps: float = 4.0
    warmup_bars: int = 200
    limit_fill_rate: float = 0.95     # simulate 95% fill for limit orders
    model_version: str | None = None   # None = use active registry pointer
    signal_latency_bars: float = 0.0
    cost_policy_version: str = "wp07-execution-cost-v1"


@dataclass
class BacktestResult:
    config: BacktestConfig
    equity_curve: pd.Series          # indexed by timestamp
    fills: list[Fill]
    daily_returns: pd.Series
    total_trades: int = 0
    win_trades: int = 0
    gross_pnl: float = 0.0
    total_fees: float = 0.0
    total_slippage: float = 0.0
    cost_policy_version: str = ""
    cost_components_usd: dict[str, float] = field(default_factory=dict)
    cost_scenarios: dict[str, dict[str, float]] = field(default_factory=dict)

    @property
    def net_pnl(self) -> float:
        total_cost = self.cost_components_usd.get("total_cost_usd")
        if total_cost is None:
            total_cost = self.total_fees + self.total_slippage
        return self.gross_pnl - float(total_cost)

    @property
    def win_rate(self) -> float:
        return self.win_trades / max(self.total_trades, 1)

    @property
    def sharpe(self) -> float:
        if self.daily_returns.std() == 0:
            return 0.0
        return float(self.daily_returns.mean() / self.daily_returns.std() * np.sqrt(252))

    @property
    def max_drawdown(self) -> float:
        eq = self.equity_curve
        if eq.empty:
            return 0.0
        roll_max = eq.cummax()
        dd = (eq - roll_max) / roll_max
        return float(dd.min())

    @property
    def max_drawdown_duration(self) -> int:
        """Longest drawdown duration in number of bars (hours)."""
        eq = self.equity_curve
        if eq.empty:
            return 0
        roll_max = eq.cummax()
        in_dd = eq < roll_max
        if not in_dd.any():
            return 0
        groups = (~in_dd).cumsum()
        dd_lengths = in_dd.groupby(groups).sum()
        return int(dd_lengths.max()) if not dd_lengths.empty else 0

    @property
    def profit_factor(self) -> float:
        """Ratio of gross winning PnL to |gross losing PnL| from round-trip trades."""
        gross_wins = 0.0
        gross_losses = 0.0
        # Pair fills into round-trips: entry fill followed by exit fill
        i = 0
        while i + 1 < len(self.fills):
            entry = self.fills[i]
            exit_ = self.fills[i + 1]
            # Match entry→exit (BUY→SELL or SELL→BUY)
            if entry.side != exit_.side and entry.symbol == exit_.symbol:
                if entry.side == "BUY":
                    rt_pnl = (exit_.price - entry.price) * entry.quantity
                else:
                    rt_pnl = (entry.price - exit_.price) * entry.quantity
                rt_pnl -= (entry.fee_usd + entry.slippage_usd + exit_.fee_usd + exit_.slippage_usd)
                if rt_pnl > 0:
                    gross_wins += rt_pnl
                else:
                    gross_losses += abs(rt_pnl)
                i += 2
            else:
                i += 1
        return gross_wins / max(gross_losses, 1e-9)


def _load_model(config: BacktestConfig):
    """Load model from registry or by version."""
    import os
    from pathlib import Path
    from quant_v2.model_registry import ModelRegistry
    from quant_v2.models.trainer import load_model
    model_root = Path(os.getenv("BOT_MODEL_ROOT", "models/production"))
    if config.model_version:
        artifact_dir = model_root / config.model_version
    else:
        registry_root = Path(os.getenv("BOT_MODEL_REGISTRY_ROOT",
                                       str(model_root / "registry")))
        registry = ModelRegistry(registry_root)
        active = registry.get_active_version()
        if active is not None:
            artifact_dir = Path(active.artifact_dir)
        else:
            dirs = sorted(model_root.glob("model_*"))
            if not dirs:
                raise FileNotFoundError("No model found in " + str(model_root))
            artifact_dir = dirs[-1]

    models = {}
    for h in (2, 4, 8):
        p = artifact_dir / f"model_{h}m.pkl"
        if p.exists():
            models[h] = load_model(p)
    if not models:
        raise FileNotFoundError(f"No horizon models in {artifact_dir}")
    return models


def _fetch_bars(symbol: str, start: str, end: str) -> pd.DataFrame:
    """Fetch full OHLCV + funding + OI for the backtest window."""
    from quant.config import BinanceAPIConfig
    from quant.data.binance_client import BinanceClient
    from quant_v2.research.scheduled_retrain import fetch_symbol_dataset

    client = BinanceClient(BinanceAPIConfig())
    date_from = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
    date_to = datetime.fromisoformat(end).replace(tzinfo=timezone.utc)
    return fetch_symbol_dataset(symbol, date_from=date_from, date_to=date_to,
                                client=client, include_funding=True,
                                include_open_interest=True)


def _bar_windows(df: pd.DataFrame, warmup: int) -> Iterator[pd.DataFrame]:
    """Yield expanding windows starting after warmup bars."""
    for i in range(warmup, len(df)):
        yield df.iloc[:i + 1]


def _predict(models: dict, feature_row: pd.DataFrame) -> tuple[float, float]:
    """Ensemble predict: average probability across horizons."""
    probs = []
    for model in models.values():
        try:
            p = float(model.predict_proba(feature_row)[:, 1][0])
            probs.append(p)
        except Exception:
            pass
    if not probs:
        return 0.5, 1.0
    avg = float(np.mean(probs))
    uncertainty = float(np.std(probs)) if len(probs) > 1 else 0.0
    return avg, uncertainty


def _sim_fill(price: float, quantity: float, side: str,
              maker_fee_bps: float, notional_usd: float,
              adv_usd: float = 1_000_000_000.0) -> tuple[float, float]:
    """Return (fee_usd, slippage_usd) for a simulated fill."""
    fee_usd = notional_usd * maker_fee_bps / 10_000.0
    participation = notional_usd / max(adv_usd / 24.0, 1.0)
    impact_bps = 10.0 * (participation ** 0.5)
    slippage_usd = notional_usd * impact_bps / 10_000.0
    return fee_usd, slippage_usd


def _estimate_adv_usd(window: pd.DataFrame) -> float:
    if window.empty:
        return 50_000_000.0
    if "quote_volume" in window.columns:
        series = pd.to_numeric(window["quote_volume"], errors="coerce").tail(24)
    elif "volume" in window.columns and "close" in window.columns:
        volume = pd.to_numeric(window["volume"], errors="coerce")
        close = pd.to_numeric(window["close"], errors="coerce")
        series = (volume * close).tail(24)
    else:
        return 50_000_000.0
    adv = float(series.replace([np.inf, -np.inf], np.nan).dropna().mean()) if not series.empty else 0.0
    return max(adv, 1.0)


def _funding_rate_bps(bar: pd.Series) -> float:
    raw = bar.get("funding_rate_raw", bar.get("funding_rate", 0.0))
    try:
        return abs(float(raw)) * 10_000.0
    except (TypeError, ValueError):
        return 0.0


def run_backtest(config: BacktestConfig) -> BacktestResult:
    """Run a single-symbol walk-forward backtest."""
    from quant.features.pipeline import build_features, get_feature_columns
    from quant_v2.portfolio.allocation import allocate_signals
    from quant_v2.contracts import StrategySignal
    from quant_v2.portfolio.cost_model import BinanceCostModel

    logger.info("Backtest: loading model...")
    models = _load_model(config)
    cost_policy = ExecutionCostPolicy(
        policy_version=config.cost_policy_version,
        maker_fee_bps=config.maker_fee_bps,
        taker_fee_bps=config.taker_fee_bps,
    )

    logger.info("Backtest: fetching bars for %s %s -> %s...",
                config.symbol, config.start_date, config.end_date)
    raw = _fetch_bars(config.symbol, config.start_date, config.end_date)
    if raw.empty:
        raise RuntimeError("No data fetched for backtest")

    cost_model = BinanceCostModel(maker_fee_bps=config.maker_fee_bps,
                                  taker_fee_bps=config.taker_fee_bps)

    equity = config.initial_equity
    equity_ts: list[tuple] = []
    fills: list[Fill] = []
    position: float = 0.0   # current qty (positive = long)
    entry_price: float = 0.0
    gross_pnl = 0.0
    total_fees = 0.0
    total_slippage = 0.0
    total_spread = 0.0
    total_funding = 0.0
    total_latency = 0.0
    total_impact = 0.0
    total_cost_usd = 0.0
    win_trades = 0
    total_round_trips = 0
    scenario_totals: dict[str, dict[str, float]] = {
        name: {
            "notional_usd": 0.0,
            "fee_usd": 0.0,
            "spread_usd": 0.0,
            "slippage_usd": 0.0,
            "funding_usd": 0.0,
            "latency_usd": 0.0,
            "impact_usd": 0.0,
            "total_cost_usd": 0.0,
            "total_cost_bps": 0.0,
        }
        for name in ("base", "adverse", "severe")
    }

    logger.info("Backtest: running %d bars (warmup=%d)...",
                len(raw), config.warmup_bars)

    for window in _bar_windows(raw, config.warmup_bars):
        bar = window.iloc[-1]
        ts = bar.name if hasattr(bar.name, 'tzinfo') else pd.Timestamp(bar.name)
        close = float(bar["close"])
        adv_usd = _estimate_adv_usd(window)
        funding_bps = _funding_rate_bps(bar)

        try:
            frame = window.copy()
            if not isinstance(frame.index, pd.DatetimeIndex):
                frame.index = pd.to_datetime(frame.index, utc=True, errors="coerce")
            featured = build_features(frame)
            if featured.empty:
                equity_ts.append((ts, equity))
                continue
            feat_cols = get_feature_columns(featured)
            feature_row = featured.iloc[[-1]][feat_cols]
        except Exception as e:
            logger.debug("Feature build failed at %s: %s", ts, e)
            equity_ts.append((ts, equity))
            continue

        proba, uncertainty = _predict(models, feature_row)
        confidence = proba if proba >= 0.5 else (1.0 - proba)
        signal_type = "BUY" if proba >= 0.5 else "SELL"

        signal = StrategySignal(
            symbol=config.symbol,
            timeframe="1h",
            horizon_bars=4,
            signal=signal_type if confidence >= config.min_confidence else "HOLD",
            confidence=confidence,
            uncertainty=uncertainty,
        )

        alloc = allocate_signals(
            [signal],
            max_symbol_exposure_frac=config.max_symbol_exposure_frac,
            min_confidence=config.min_confidence,
            enable_session_filter=False,
            enable_regime_bias=False,
            enable_symbol_accuracy=False,
            enable_event_gate=False,
            enable_model_agreement=False,
            enable_cost_gate=True,
            equity_usd=equity,
            cost_model=cost_model,
        )

        target_frac = alloc.target_exposures.get(config.symbol, 0.0)
        target_notional = abs(target_frac) * equity
        target_qty = target_notional / close if close > 0 else 0.0
        # Close existing position if direction changed or signal is HOLD
        if position != 0.0 and (target_frac == 0.0 or
                                  (position > 0 and target_frac < 0) or
                                  (position < 0 and target_frac > 0)):
            exit_notional = abs(position) * close
            fee, slip = _sim_fill(close, abs(position), "SELL",
                                   config.maker_fee_bps, exit_notional)
            exit_estimates = cost_policy.scenario_estimates(
                config.symbol,
                "SELL" if position > 0 else "BUY",
                exit_notional,
                adv_usd=adv_usd,
                funding_rate_bps=funding_bps,
                latency_bars=config.signal_latency_bars,
            )
            base_exit = exit_estimates["base"]
            total_fill_cost = base_exit.total_cost_usd
            trade_pnl = (close - entry_price) * position
            net_trade_pnl = trade_pnl - total_fill_cost
            gross_pnl += trade_pnl
            total_fees += fee
            total_slippage += slip
            total_spread += base_exit.spread_usd
            total_funding += base_exit.funding_usd
            total_latency += base_exit.latency_usd
            total_impact += base_exit.impact_usd
            total_cost_usd += total_fill_cost
            for scenario_name, estimate in exit_estimates.items():
                totals = scenario_totals[scenario_name]
                for key, value in estimate.as_totals().items():
                    totals[key] += value
            equity += net_trade_pnl
            total_round_trips += 1
            if net_trade_pnl > 0:
                win_trades += 1
            fills.append(Fill(
                timestamp=ts, symbol=config.symbol,
                side="SELL" if position > 0 else "BUY",
                quantity=abs(position), price=close,
                fee_usd=fee, slippage_usd=slip, confidence=confidence,
                policy_version=config.cost_policy_version,
                cost_breakdown={
                    "fee_usd": base_exit.fee_usd,
                    "spread_usd": base_exit.spread_usd,
                    "slippage_usd": base_exit.slippage_usd,
                    "funding_usd": base_exit.funding_usd,
                    "latency_usd": base_exit.latency_usd,
                    "impact_usd": base_exit.impact_usd,
                    "total_cost_usd": base_exit.total_cost_usd,
                    "total_cost_bps": base_exit.total_cost_bps,
                },
            ))
            position = 0.0
            entry_price = 0.0

        # Open new position (with partial fill simulation)
        if target_qty > 0.0 and position == 0.0:
            filled_qty = target_qty * config.limit_fill_rate
            filled_notional = filled_qty * close
            fee, slip = _sim_fill(close, filled_qty,
                                   "BUY" if target_frac > 0 else "SELL",
                                   config.maker_fee_bps, filled_notional)
            entry_estimates = cost_policy.scenario_estimates(
                config.symbol,
                "BUY" if target_frac > 0 else "SELL",
                filled_notional,
                adv_usd=adv_usd,
                funding_rate_bps=funding_bps,
                latency_bars=config.signal_latency_bars,
            )
            base_entry = entry_estimates["base"]
            total_fill_cost = base_entry.total_cost_usd
            total_fees += fee
            total_slippage += slip
            total_spread += base_entry.spread_usd
            total_funding += base_entry.funding_usd
            total_latency += base_entry.latency_usd
            total_impact += base_entry.impact_usd
            total_cost_usd += total_fill_cost
            for scenario_name, estimate in entry_estimates.items():
                totals = scenario_totals[scenario_name]
                for key, value in estimate.as_totals().items():
                    totals[key] += value
            equity -= total_fill_cost
            position = filled_qty if target_frac > 0 else -filled_qty
            entry_price = close
            fills.append(Fill(
                timestamp=ts, symbol=config.symbol,
                side="BUY" if target_frac > 0 else "SELL",
                quantity=filled_qty, price=close,
                fee_usd=fee, slippage_usd=slip, confidence=confidence,
                policy_version=config.cost_policy_version,
                cost_breakdown={
                    "fee_usd": base_entry.fee_usd,
                    "spread_usd": base_entry.spread_usd,
                    "slippage_usd": base_entry.slippage_usd,
                    "funding_usd": base_entry.funding_usd,
                    "latency_usd": base_entry.latency_usd,
                    "impact_usd": base_entry.impact_usd,
                    "total_cost_usd": base_entry.total_cost_usd,
                    "total_cost_bps": base_entry.total_cost_bps,
                },
            ))

        equity_ts.append((ts, equity))

    equity_series = pd.Series(
        {t: e for t, e in equity_ts},
        name="equity_usd",
    )
    daily = equity_series.resample("D").last().ffill().pct_change().dropna()
    for totals in scenario_totals.values():
        notional = totals.get("notional_usd", 0.0)
        totals["total_cost_bps"] = (totals["total_cost_usd"] / notional * 10_000.0) if notional > 0.0 else 0.0

    return BacktestResult(
        config=config,
        equity_curve=equity_series,
        fills=fills,
        daily_returns=daily,
        total_trades=total_round_trips,
        win_trades=win_trades,
        gross_pnl=gross_pnl,
        total_fees=total_fees,
        total_slippage=total_slippage,
        cost_policy_version=config.cost_policy_version,
        cost_components_usd={
            "fees": total_fees,
            "slippage": total_slippage,
            "spread": total_spread,
            "funding": total_funding,
            "latency": total_latency,
            "impact": total_impact,
            "total_cost_usd": total_cost_usd,
        },
        cost_scenarios=scenario_totals,
    )
