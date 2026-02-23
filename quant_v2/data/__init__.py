"""Data platform modules for v2 multi-symbol research."""

from quant_v2.data.multi_symbol_dataset import fetch_symbol_dataset, fetch_universe_dataset
from quant_v2.data.storage import (
    DataQualityError,
    MultiSymbolSnapshot,
    load_multi_symbol_snapshot,
    save_multi_symbol_snapshot,
    validate_multi_symbol_ohlcv,
)

__all__ = [
    "DataQualityError",
    "MultiSymbolSnapshot",
    "fetch_symbol_dataset",
    "fetch_universe_dataset",
    "load_multi_symbol_snapshot",
    "save_multi_symbol_snapshot",
    "validate_multi_symbol_ohlcv",
]
