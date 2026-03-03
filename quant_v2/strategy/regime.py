"""
Four-state Markov regime classifier for signal gating.

States:
    1 (Momentum):  Trending with non-crowded funding
    2 (Reversion): Counter-trend with non-crowded funding
    3 (Neutral):   Default state — reduced confidence
    4 (Adverse):   Extreme funding crowding or price deviation

RegimeRisk mapping:
    Regime 1, 2 → RegimeRisk = 0 (favourable for trading)
    Regime 3, 4 → RegimeRisk = 1 (reduced confidence)

Transitions require a 5-bar persistence guard to reduce whipsaw.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RegimeState:
    """Output of regime classification for a single bar."""

    regime: int  # 1-4
    regime_risk: int  # 0 or 1
    persistence_count: int  # consecutive bars in current regime


_REGIME_RISK_MAP: dict[int, int] = {1: 0, 2: 0, 3: 1, 4: 1}

_PERSISTENCE_GUARD: int = 5  # bars before confirming a transition


def classify_regime(
    close: pd.Series,
    funding_zscore: pd.Series,
    *,
    lookback: int = 30,
    persistence_guard: int = _PERSISTENCE_GUARD,
) -> list[RegimeState]:
    """Classify regime for each bar using log-returns and funding z-score.

    Args:
        close: Close price series (numeric, same length as funding_zscore).
        funding_zscore: Winsorised funding rate z-score series.
        lookback: Window for log-return sigma calculation.
        persistence_guard: Bars required before confirming a regime change.

    Returns:
        List of RegimeState, one per bar (aligned with input index).
    """
    n = len(close)
    if n == 0:
        return []

    close_arr = np.asarray(close, dtype=float)
    fz_arr = np.asarray(funding_zscore, dtype=float)

    # --- 30-bar cumulative log-return ---
    log_ret_30 = np.full(n, np.nan)
    for i in range(lookback, n):
        if close_arr[i - lookback] > 0.0 and close_arr[i] > 0.0:
            log_ret_30[i] = np.log(close_arr[i] / close_arr[i - lookback])

    # --- rolling sigma of 1-bar log returns (scaled to lookback horizon) ---
    one_bar_lr = np.full(n, np.nan)
    for i in range(1, n):
        if close_arr[i - 1] > 0.0 and close_arr[i] > 0.0:
            one_bar_lr[i] = np.log(close_arr[i] / close_arr[i - 1])

    rolling_sigma = np.full(n, np.nan)
    for i in range(lookback, n):
        window = one_bar_lr[max(0, i - lookback + 1): i + 1]
        valid = window[~np.isnan(window)]
        if len(valid) >= 10:
            rolling_sigma[i] = float(np.std(valid, ddof=1)) * np.sqrt(lookback)

    # --- raw regime per bar ---
    raw = np.full(n, 3, dtype=int)
    for i in range(lookback, n):
        lr = log_ret_30[i]
        sigma = rolling_sigma[i]
        fz = fz_arr[i] if not np.isnan(fz_arr[i]) else 0.0

        if np.isnan(lr) or np.isnan(sigma) or sigma <= 0.0:
            raw[i] = 3
            continue

        # Regime 4 (Adverse): extreme funding crowding
        if abs(fz) > 2.0:
            raw[i] = 4
            continue

        # Regime 1 (Momentum): strong uptrend + non-crowded funding
        if lr > sigma and fz < 1.0:
            raw[i] = 1
            continue

        # Regime 2 (Reversion): strong downtrend + non-crowded funding
        if lr < -sigma and fz > -1.0:
            raw[i] = 2
            continue

        # Default: Regime 3 (Neutral)
        raw[i] = 3

    # --- persistence guard ---
    states: list[RegimeState] = []
    current_regime = 3
    pending_regime = 3
    pending_count = 0
    bars_in_regime = 0

    for i in range(n):
        candidate = int(raw[i])

        if candidate == current_regime:
            pending_regime = current_regime
            pending_count = 0
            bars_in_regime += 1
        elif candidate == pending_regime:
            pending_count += 1
            if pending_count >= persistence_guard:
                current_regime = pending_regime
                bars_in_regime = pending_count
                pending_count = 0
            else:
                bars_in_regime += 1
        else:
            pending_regime = candidate
            pending_count = 1
            bars_in_regime += 1

        states.append(
            RegimeState(
                regime=current_regime,
                regime_risk=_REGIME_RISK_MAP.get(current_regime, 1),
                persistence_count=bars_in_regime,
            )
        )

    return states


def classify_latest(
    close: pd.Series,
    funding_zscore: pd.Series,
    *,
    lookback: int = 30,
    persistence_guard: int = _PERSISTENCE_GUARD,
) -> RegimeState:
    """Return the regime state for the latest (last) bar only."""

    states = classify_regime(
        close,
        funding_zscore,
        lookback=lookback,
        persistence_guard=persistence_guard,
    )
    if not states:
        return RegimeState(regime=3, regime_risk=1, persistence_count=0)
    return states[-1]
