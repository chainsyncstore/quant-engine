"""Tests for Fix 1: Decimal-safe quantity & price rounding.

Validates that _normalize_quantity_with_filters uses Decimal(ROUND_DOWN)
and _quantize_price uses Decimal(ROUND_HALF_UP) for tick-size compliance.
"""

from __future__ import annotations

from quant_v2.contracts import OrderPlan
from quant_v2.execution.binance_adapter import BinanceExecutionAdapter


class FakeBinanceClientDecimal:
    """Fake client with configurable symbol filters."""

    def __init__(self, filters: dict | None = None) -> None:
        self._filters = filters or {}
        self.place_calls: list[tuple] = []

    def get_symbol_filters(self, symbol: str) -> dict:
        return dict(self._filters)

    def place_order(self, symbol, side, qty, order_type="MARKET"):
        self.place_calls.append((symbol, side, qty, order_type))
        return {
            "orderId": 100,
            "status": "FILLED",
            "avgPrice": "50000.0",
            "executedQty": str(qty),
        }

    def place_limit_order(self, symbol, side, qty, price, post_only=False, **kw):
        self.place_calls.append((symbol, side, qty, price))
        return {
            "orderId": 101,
            "status": "FILLED",
            "avgPrice": str(price),
            "executedQty": str(qty),
        }

    def get_positions(self, symbol=None):
        return [{"symbol": "BTCUSDT", "positionAmt": "0.5"}]

    def get_open_orders(self, symbol=None):
        return []

    def cancel_order(self, symbol, order_id):
        return {"status": "CANCELED"}

    def get_orderbook(self, symbol, limit=5):
        return {
            "bids": [["50000.0", "1.0"]],
            "asks": [["50010.0", "1.0"]],
        }


class TestQuantityDecimalPrecision:
    """Verify quantity rounding uses Decimal(ROUND_DOWN), not float math."""

    def test_quantity_quantized_to_step_size_decimal(self):
        """0.123 with step_size=0.01 must produce exactly 0.12, not 0.11999..."""
        client = FakeBinanceClientDecimal(
            filters={"step_size": "0.01", "min_qty": "0.01", "min_notional": "0.0"}
        )
        adapter = BinanceExecutionAdapter(client)

        normalized, skip = adapter._normalize_quantity_with_filters(
            "BTCUSDT", quantity=0.123, mark_price=50000.0
        )
        assert skip is None
        assert normalized == 0.12
        # Ensure exact float representation (no trailing imprecision)
        assert str(normalized) == "0.12"

    def test_quantity_rounds_down_not_up(self):
        """0.019 with step_size=0.01 must produce 0.01, not 0.02."""
        client = FakeBinanceClientDecimal(
            filters={"step_size": "0.01", "min_qty": "0.001", "min_notional": "0.0"}
        )
        adapter = BinanceExecutionAdapter(client)

        normalized, skip = adapter._normalize_quantity_with_filters(
            "BTCUSDT", quantity=0.019, mark_price=50000.0
        )
        assert skip is None
        assert normalized == 0.01

    def test_quantity_fine_step_size(self):
        """0.12345 with step_size=0.001 must produce 0.123."""
        client = FakeBinanceClientDecimal(
            filters={"step_size": "0.001", "min_qty": "0.001", "min_notional": "0.0"}
        )
        adapter = BinanceExecutionAdapter(client)

        normalized, skip = adapter._normalize_quantity_with_filters(
            "BTCUSDT", quantity=0.12345, mark_price=50000.0
        )
        assert skip is None
        assert normalized == 0.123


class TestPriceDecimalPrecision:
    """Verify _quantize_price rounds to tick_size via Decimal(ROUND_HALF_UP)."""

    def test_price_quantized_to_tick_size(self):
        client = FakeBinanceClientDecimal(
            filters={"tick_size": "0.01"}
        )
        adapter = BinanceExecutionAdapter(client)

        # 49750.005 with tick=0.01 should round to 49750.01 (ROUND_HALF_UP)
        result = adapter._quantize_price("BTCUSDT", 49750.005)
        assert result == 49750.01

    def test_price_already_aligned(self):
        client = FakeBinanceClientDecimal(
            filters={"tick_size": "0.10"}
        )
        adapter = BinanceExecutionAdapter(client)

        result = adapter._quantize_price("BTCUSDT", 50000.0)
        assert result == 50000.0

    def test_price_no_filter_returns_raw(self):
        """If no tick_size filter exists, price is returned unchanged."""
        client = FakeBinanceClientDecimal(filters={})
        adapter = BinanceExecutionAdapter(client)

        result = adapter._quantize_price("BTCUSDT", 49999.123456)
        assert result == 49999.123456


class TestCloseBoundedPriceQuantized:
    """Verify close_position_bounded applies tick-size quantization to limit price."""

    def test_close_bounded_price_is_tick_quantized(self):
        client = FakeBinanceClientDecimal(
            filters={"tick_size": "0.10", "step_size": "0.001"}
        )
        adapter = BinanceExecutionAdapter(client)

        result = adapter.close_position_bounded("BTCUSDT", max_slippage_bps=50)

        assert result.accepted is True
        # The limit price should be tick-quantized to multiples of 0.10
        _, _, _, limit_price = client.place_calls[0]
        remainder = round(limit_price % 0.10, 10)
        assert remainder == 0.0 or remainder == 0.10, (
            f"Price {limit_price} not quantized to tick_size=0.10"
        )
