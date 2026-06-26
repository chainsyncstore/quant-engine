"""Canonical feature catalog for the research pipeline."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from typing import Any, Sequence

import pandas as pd


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    source_module: str
    group: str


_FEATURE_GROUP_DEFINITIONS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (
        "momentum",
        "quant.features.momentum",
        (
            "roc_1",
            "roc_3",
            "roc_5",
            "roc_10",
            "roc_20",
            "momentum_accel",
            "roc_divergence_5_20",
        ),
    ),
    (
        "volatility",
        "quant.features.volatility",
        (
            "atr_14",
            "rolling_std_10",
            "rolling_std_20",
            "bb_pct_b",
            "vol_ratio",
            "parkinson_vol",
            "realized_vol_5",
            "vol_of_vol",
            "garman_klass_vol",
        ),
    ),
    (
        "candle_geometry",
        "quant.features.candle_geometry",
        (
            "body_range_ratio",
            "upper_wick_ratio",
            "lower_wick_ratio",
            "consec_direction",
            "gap",
        ),
    ),
    (
        "trend",
        "quant.features.trend",
        (
            "ema_slope_5",
            "ema_cross_dist",
            "price_ema_spread",
            "ema_slope_50",
            "mean_reversion_score",
            "dist_ema_5_20",
            "dist_ema_20_50",
            "dist_close_ema_20",
        ),
    ),
    (
        "volume",
        "quant.features.volume",
        (
            "vol_zscore",
            "vol_ratio_20",
            "obv_slope",
            "vwap_dist",
        ),
    ),
    (
        "time_encoding",
        "quant.features.time_encoding",
        (
            "hour_sin",
            "hour_cos",
            "dow_sin",
            "dow_cos",
            "day_of_week_sin",
            "day_of_week_cos",
        ),
    ),
    (
        "microstructure",
        "quant.features.microstructure",
        (
            "return_autocorr_5",
            "return_kurtosis_20",
            "high_low_range_ratio",
            "trade_imbalance_10",
            "amihud_illiquidity",
            "kyle_lambda_20",
        ),
    ),
    (
        "cross_timeframe",
        "quant.features.cross_timeframe",
        (
            "roc_60",
            "atr_ratio_60_14",
            "trend_alignment",
        ),
    ),
    (
        "order_flow",
        "quant.features.order_flow",
        (
            "taker_buy_ratio",
            "taker_buy_ratio_ma8",
            "taker_buy_ratio_zscore",
            "cumulative_delta_8",
            "cumulative_delta_8_norm",
            "flow_imbalance_1",
            "flow_imbalance_4",
            "volume_weighted_flow",
        ),
    ),
    (
        "funding_rate",
        "quant.features.funding_rate",
        (
            "funding_rate",
            "funding_rate_ma8",
            "funding_rate_zscore",
            "funding_rate_extreme",
            "funding_cumulative_24h",
            "funding_momentum",
        ),
    ),
    (
        "open_interest",
        "quant.features.open_interest",
        (
            "oi_roc_1",
            "oi_roc_8",
            "oi_zscore",
            "oi_price_divergence",
            "oi_acceleration",
        ),
    ),
    (
        "liquidation",
        "quant.features.liquidation",
        (
            "liquidations_long_vol",
            "liquidations_short_vol",
            "liquidation_candle",
            "liquidation_up_pressure",
            "liquidation_down_pressure",
            "post_liquidation_flag",
        ),
    ),
    (
        "liquidation_proximity",
        "quant.features.liquidation_proximity",
        (
            "oi_funding_pressure",
            "price_position_24h",
            "liquidation_cascade_4h",
        ),
    ),
    (
        "cross_pair",
        "quant.features.cross_pair",
        (
            "btc_return_4h",
            "btc_divergence_4h",
            "btc_correlation_24h",
            "relative_vol_ratio",
        ),
    ),
    (
        "crypto_session",
        "quant.features.crypto_session",
        (
            "hours_to_funding",
            "hours_to_funding_sin",
            "hours_to_funding_cos",
            "post_funding_window",
            "asia_session",
            "europe_session",
            "us_session",
        ),
    ),
    (
        "order_book",
        "quant.features.order_book",
        (
            "bid_ask_spread_bps",
            "book_imbalance_5",
            "book_imbalance_20",
            "depth_ratio_5",
            "volume_at_touch_ratio",
            "spread_vol_ratio",
        ),
    ),
)


FEATURE_SPECS: tuple[FeatureSpec, ...] = tuple(
    FeatureSpec(name=name, source_module=module, group=group)
    for group, module, names in _FEATURE_GROUP_DEFINITIONS
    for name in names
)

FEATURE_NAMES: tuple[str, ...] = tuple(spec.name for spec in FEATURE_SPECS)
FEATURE_CATALOG_VERSION = "wp08-feature-catalog-v1"


def _catalog_payload() -> list[dict[str, Any]]:
    return [asdict(spec) for spec in FEATURE_SPECS]


FEATURE_CATALOG_SHA256 = hashlib.sha256(
    json.dumps(_catalog_payload(), sort_keys=True, separators=(",", ":")).encode("utf-8")
).hexdigest()


def validate_feature_catalog() -> None:
    """Fail closed if the catalog contains duplicates or drift."""

    names = list(FEATURE_NAMES)
    if len(names) != len(set(names)):
        raise ValueError("Feature catalog contains duplicate feature names")


def feature_catalog_metadata() -> dict[str, Any]:
    """Return canonical feature catalog metadata for provenance and manifests."""

    return {
        "feature_catalog_version": FEATURE_CATALOG_VERSION,
        "feature_catalog_sha256": FEATURE_CATALOG_SHA256,
        "feature_catalog_feature_names": list(FEATURE_NAMES),
        "feature_catalog_feature_count": len(FEATURE_NAMES),
    }


def validate_feature_columns(columns: Sequence[str]) -> None:
    """Ensure the observed frame contains the full canonical feature catalog."""

    present = set(columns)
    missing = [name for name in FEATURE_NAMES if name not in present]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")


def attach_feature_catalog(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach catalog metadata to a feature frame."""

    frame = frame.copy()
    frame.attrs.update(feature_catalog_metadata())
    frame.attrs["feature_missing_data_policy"] = "drop-core-and-derived-missing-v1"
    return frame
