"""Walk-forward backtester that replays through the exact live allocation chain.

Replays historical bars bar-by-bar, builds features, predicts, allocates,
and simulates fills — using the same code path as production.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import pandas as pd
import numpy as np

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
    model_version: str | None = None   # None = use active registry pointer


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

    @property
    def net_pnl(self) -> float:
        return self.gross_pnl - self.total_fees - self.total_slippage

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
    def profit_factor(self) -> float:
        wins = sum(f.price * f.quantity - f.fee_usd - f.slippage_usd
                   for f in self.fills if f.side == "BUY")
        losses = sum(f.fee_usd + f.slippage_usd
                     for f in self.fills if f.side == "SELL")
        return wins / max(abs(losses), 1e-9)


def _load_model(config: BacktestConfig):
    """Load model from registry or by version."""
    from pathlib import Path
    import os, pickle
    model_root = Path(os.getenv("BOT_MODEL_ROOT", "models/production"))
    if config.model_version:
        artifact_dir = model_root / config.model_version
    else:
        registry_root = Path(os.getenv("BOT_MODEL_REGISTRY_ROOT",
                                       str(model_root / "registry")))
        ptr = registry_root / "active.json"
        if ptr.exists():
            import json
            active = json.loads(ptr.read_text())
            version_id = active.get("version_id", "")
            artifact_dir = model_root / version_id
        else:
            dirs = sorted(model_root.glob("model_*"))
            if not dirs:
                raise FileNotFoundError("No model found in " + str(model_root))
            artifact_dir = dirs[-1]

    models = {}
    for h in (2, 4, 8):
        p = artifact_dir / f"model_{h}m.pkl"
        if p.exists():
            with open(p, "rb") as f:
                models[h] = pickle.load(f)
    if not models:
        raise FileNotFoundError(f"No horizon models in {artifact_dir}")
    return models


def _fetch_bars(symbol: str, start: str, end: str) -> pd.DataFrame:
    """Fetch full OHLCV + funding + OI for the backtest window."""
    from datetime import timedelta
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


def run_backtest(config: BacktestConfig) -> BacktestResult:
    """Run a single-symbol walk-forward backtest."""
    from quant.features.pipeline import build_features, get_feature_columns
    from quant_v2.portfolio.allocation import allocate_signals
    from quant_v2.contracts import StrategySignal
    from quant_v2.portfolio.cost_model import BinanceCostModel

    logger.info("Backtest: loading model...")
    models = _load_model(config)

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

    logger.info("Backtest: running %d bars (warmup=%d)...",
                len(raw), config.warmup_bars)

    for window in _bar_windows(raw, config.warmup_bars):
        bar = window.iloc[-1]
        ts = bar.name if hasattr(bar.name, 'tzinfo') else pd.Timestamp(bar.name)
        close = float(bar["close"])

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
            pnl = (close - entry_price) * position - fee - slip
            gross_pnl += pnl
            total_fees += fee
            total_slippage += slip
            equity += pnl
            fills.append(Fill(
                timestamp=ts, symbol=config.symbol,
                side="SELL" if position > 0 else "BUY",
                quantity=abs(position), price=close,
                fee_usd=fee, slippage_usd=slip, confidence=confidence,
            ))
            position = 0.0
            entry_price = 0.0

        # Open new position
        if target_qty > 0.0 and position == 0.0:
            fee, slip = _sim_fill(close, target_qty,
                                   "BUY" if target_frac > 0 else "SELL",
                                   config.maker_fee_bps, target_notional)
            total_fees += fee
            total_slippage += slip
            equity -= (fee + slip)
            position = target_qty if target_frac > 0 else -target_qty
            entry_price = close
            fills.append(Fill(
                timestamp=ts, symbol=config.symbol,
                side="BUY" if target_frac > 0 else "SELL",
                quantity=target_qty, price=close,
                fee_usd=fee, slippage_usd=slip, confidence=confidence,
            ))

        equity_ts.append((ts, equity))

    equity_series = pd.Series(
        {t: e for t, e in equity_ts},
        name="equity_usd",
    )
    daily = equity_series.resample("D").last().ffill().pct_change().dropna()
    open_fills = [f for f in fills if f.side in ("BUY", "SELL")]
    win_trades = sum(1 for f in fills if f.side == "SELL" and f.price > entry_price)

    return BacktestResult(
        config=config,
        equity_curve=equity_series,
        fills=fills,
        daily_returns=daily,
        total_trades=len([f for f in fills if f.side in ("BUY", "SELL")]),
        win_trades=win_trades,
        gross_pnl=gross_pnl,
        total_fees=total_fees,
        total_slippage=total_slippage,
    )
