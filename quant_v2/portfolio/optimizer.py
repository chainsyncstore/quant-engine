"""Portfolio optimizer — risk-parity weight allocation with correlation adjustment.

Replaces flat per-symbol exposure with weights that are inversely proportional
to volatility (risk-parity) and penalised for cross-symbol correlation.

Design choices:
- **Risk-parity** as default: allocate inversely proportional to 72h rolling
  volatility. Simple, robust, no return estimates needed.
- **Correlation penalty**: when two symbols are highly correlated (>0.7),
  reduce their combined weight so the portfolio isn't secretly concentrated
  in one factor.
- **Minimum notional filter**: positions below `min_notional_usd` are dropped
  as uneconomic (cost > edge at small capital).
- No scipy dependency: all math is pure numpy, keeping EC2 memory low.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_CORRELATION_PENALTY_THRESHOLD: float = 0.70   # above this, pairs are penalised
_CORRELATION_PENALTY_FACTOR: float = 0.75      # multiply each correlated symbol by this
_MIN_VOL_FLOOR: float = 1e-6                   # avoid div-by-zero in risk-parity
_DEFAULT_LOOKBACK_BARS: int = 72               # 72h rolling window for vol/corr estimates
_DEFAULT_MIN_NOTIONAL_USD: float = 10.0        # drop positions below this value


@dataclass(frozen=True)
class OptimizerResult:
    """Result of a portfolio optimization pass."""

    weights: dict[str, float]             # symbol → weight (signed, sums to ≤ 1.0)
    vols: dict[str, float]                # symbol → rolling std
    correlations: dict[tuple, float]      # (sym_a, sym_b) → correlation
    dropped_symbols: list[str]            # symbols dropped by min_notional filter
    constraints_applied: tuple[str, ...]


class RiskParityOptimizer:
    """Risk-parity portfolio optimizer with correlation-adjusted weights.

    Parameters
    ----------
    lookback_bars : int
        Number of 1h bars for rolling volatility / correlation estimates.
    min_notional_usd : float
        Minimum trade notional (USD). Positions below this are dropped.
    correlation_threshold : float
        Pair correlation above this triggers a penalty.
    """

    def __init__(
        self,
        lookback_bars: int = _DEFAULT_LOOKBACK_BARS,
        min_notional_usd: float = _DEFAULT_MIN_NOTIONAL_USD,
        correlation_threshold: float = _CORRELATION_PENALTY_THRESHOLD,
    ) -> None:
        self.lookback_bars = lookback_bars
        self.min_notional_usd = min_notional_usd
        self.correlation_threshold = correlation_threshold

    def optimize(
        self,
        target_exposures: dict[str, float],
        price_histories: dict[str, pd.Series],
        equity_usd: float,
    ) -> OptimizerResult:
        """Optimize exposures using risk-parity weights.

        Parameters
        ----------
        target_exposures : dict[str, float]
            Signed exposure fractions from ``allocate_signals`` (e.g. {BTC: 0.12, ETH: -0.08}).
        price_histories : dict[str, pd.Series]
            Recent close prices per symbol, at least ``lookback_bars`` bars.
        equity_usd : float
            Current portfolio equity in USD. Used for min-notional filter.

        Returns
        -------
        OptimizerResult
            Optimized signed weights, ready to replace ``target_exposures``.
        """
        if not target_exposures:
            return OptimizerResult(
                weights={}, vols={}, correlations={},
                dropped_symbols=[], constraints_applied=(),
            )

        symbols = list(target_exposures.keys())
        directions = {s: np.sign(v) for s, v in target_exposures.items()}
        constraints: list[str] = []

        # --- Step 1: Compute per-symbol rolling volatility ---
        vols: dict[str, float] = {}
        for sym in symbols:
            hist = price_histories.get(sym)
            if hist is None or len(hist) < 2:
                vols[sym] = 1.0   # no data → treat as average vol
                continue
            returns = hist.pct_change().dropna()
            tail = returns.iloc[-self.lookback_bars:]
            if len(tail) < 2:
                vols[sym] = 1.0
            else:
                vols[sym] = float(tail.std())
            vols[sym] = max(vols[sym], _MIN_VOL_FLOOR)

        # --- Step 2: Risk-parity weights (inverse-vol) ---
        inv_vols = {s: 1.0 / vols[s] for s in symbols}
        total_inv_vol = sum(inv_vols.values())
        raw_weights = {s: inv_vols[s] / total_inv_vol for s in symbols}

        # --- Step 3: Pairwise correlation and penalty ---
        correlations: dict[tuple, float] = {}
        returns_map: dict[str, pd.Series] = {}
        for sym in symbols:
            hist = price_histories.get(sym)
            if hist is not None and len(hist) >= 2:
                ret = hist.pct_change().dropna().iloc[-self.lookback_bars:]
                if not ret.empty:
                    returns_map[sym] = ret

        penalty_applied = False
        for i, sym_a in enumerate(symbols):
            for sym_b in symbols[i + 1:]:
                ret_a = returns_map.get(sym_a)
                ret_b = returns_map.get(sym_b)
                if ret_a is None or ret_b is None:
                    continue
                aligned = pd.concat([ret_a, ret_b], axis=1, join="inner")
                if len(aligned) < 10:
                    continue
                corr = float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))
                correlations[(sym_a, sym_b)] = corr

                # If high correlation and same direction: reduce both
                if abs(corr) >= self.correlation_threshold:
                    same_direction = directions.get(sym_a, 0) == directions.get(sym_b, 0)
                    if same_direction:
                        raw_weights[sym_a] *= _CORRELATION_PENALTY_FACTOR
                        raw_weights[sym_b] *= _CORRELATION_PENALTY_FACTOR
                        penalty_applied = True
                        logger.debug(
                            "Correlation penalty: %s↔%s corr=%.2f (same direction)",
                            sym_a, sym_b, corr,
                        )

        if penalty_applied:
            constraints.append("correlation_penalty")

        # Renormalise after penalties so weights still sum to ≤ 1.0
        total_w = sum(raw_weights.values())
        if total_w > 0:
            raw_weights = {s: w / total_w for s, w in raw_weights.items()}

        # --- Step 4: Apply directions and scale by original gross exposure ---
        original_gross = sum(abs(v) for v in target_exposures.values())
        signed_weights: dict[str, float] = {}
        for sym in symbols:
            w = raw_weights[sym] * original_gross * directions.get(sym, 1.0)
            signed_weights[sym] = w

        # --- Step 5: Minimum notional filter (dynamic: max(base, equity × 2%)) ---
        effective_min_notional = max(self.min_notional_usd, equity_usd * 0.02)
        dropped: list[str] = []
        for sym in list(signed_weights.keys()):
            notional = abs(signed_weights[sym]) * equity_usd
            if notional < effective_min_notional:
                dropped.append(sym)
                del signed_weights[sym]
                logger.debug(
                    "Min notional filter: dropped %s (notional=%.2f < %.2f)",
                    sym, notional, effective_min_notional,
                )

        if dropped:
            constraints.append("min_notional_filter")

        logger.info(
            "Optimizer: %d symbols → %d after filter | vols=%s | corr_pairs=%d",
            len(symbols), len(signed_weights),
            {s: f"{v:.4f}" for s, v in vols.items()},
            len(correlations),
        )

        return OptimizerResult(
            weights=signed_weights,
            vols=vols,
            correlations=correlations,
            dropped_symbols=dropped,
            constraints_applied=tuple(dict.fromkeys(constraints)),
        )


def compute_rolling_correlations(
    price_histories: dict[str, pd.Series],
    lookback_bars: int = _DEFAULT_LOOKBACK_BARS,
) -> dict[tuple[str, str], float]:
    """Compute all pairwise rolling correlations from price histories.

    Useful for diagnostics and Telegram `/stats` output.

    Returns
    -------
    dict mapping (sym_a, sym_b) → Pearson correlation over lookback window.
    """
    symbols = list(price_histories.keys())
    result: dict[tuple[str, str], float] = {}

    returns_map = {}
    for sym, prices in price_histories.items():
        if prices is not None and len(prices) >= 2:
            ret = prices.pct_change().dropna().iloc[-lookback_bars:]
            if not ret.empty:
                returns_map[sym] = ret

    for i, sym_a in enumerate(symbols):
        for sym_b in symbols[i + 1:]:
            ret_a = returns_map.get(sym_a)
            ret_b = returns_map.get(sym_b)
            if ret_a is None or ret_b is None:
                continue
            aligned = pd.concat([ret_a, ret_b], axis=1, join="inner")
            if len(aligned) < 10:
                continue
            corr = float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))
            result[(sym_a, sym_b)] = corr

    return result
