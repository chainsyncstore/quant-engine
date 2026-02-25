"""Tests for Fix 4: Slippage-bounded emergency liquidation."""

from __future__ import annotations

from quant_v2.contracts import OrderPlan
from quant_v2.execution.binance_adapter import BinanceExecutionAdapter


class FakeBinanceClientForBounded:
    """Fake Binance client for testing close_position_bounded."""

    def __init__(self) -> None:
        self.limit_calls: list[tuple[str, str, float, float]] = []
        self.market_calls: list[tuple[str, str, float]] = []
        self.cancel_calls: list[tuple[str, str | int]] = []
        self._positions: list[dict] = [
            {"symbol": "BTCUSDT", "positionAmt": "0.5"},
        ]
        self._open_orders: list[dict] = []

    def get_positions(self, symbol=None):
        if symbol:
            return [p for p in self._positions if p["symbol"] == symbol]
        return self._positions

    def place_limit_order(self, symbol, side, qty, price, post_only=False, **kwargs):
        self.limit_calls.append((symbol, side, qty, price))
        return {
            "orderId": 1001,
            "status": "FILLED",
            "avgPrice": str(price),
            "executedQty": str(qty),
        }

    def place_order(self, symbol, side, qty, order_type="MARKET"):
        self.market_calls.append((symbol, side, qty))
        return {
            "orderId": 2001,
            "status": "FILLED",
            "avgPrice": "49000.0",
            "executedQty": str(qty),
        }

    def get_open_orders(self, symbol=None):
        return list(self._open_orders)

    def cancel_order(self, symbol, order_id):
        self.cancel_calls.append((symbol, order_id))
        return {}

    def get_symbol_filters(self, symbol):
        return {}

    def get_orderbook(self, symbol, limit=5):
        return {
            "bids": [["50000.0", "1.0"]],
            "asks": [["50010.0", "1.0"]],
        }


class TestCloseBoundedUsesAggressiveLimit:
    """Verify close_position_bounded uses LIMIT, not MARKET."""

    def test_close_position_bounded_uses_aggressive_limit(self):
        client = FakeBinanceClientForBounded()
        adapter = BinanceExecutionAdapter(client)

        result = adapter.close_position_bounded("BTCUSDT", max_slippage_bps=50)

        assert result.accepted is True
        assert result.reason == "bounded_limit_exit"
        assert len(client.limit_calls) == 1
        assert len(client.market_calls) == 0

        # Check the limit price is below bid for a SELL (long exit)
        _, side, qty, price = client.limit_calls[0]
        assert side == "SELL"
        assert qty == 0.5
        assert price < 50000.0  # Should be bid * (1 - 50bps)
        assert price > 49000.0  # Shouldn't be absurdly low

    def test_close_position_bounded_no_position(self):
        client = FakeBinanceClientForBounded()
        client._positions = []
        adapter = BinanceExecutionAdapter(client)

        result = adapter.close_position_bounded("BTCUSDT")

        assert result.accepted is False
        assert result.reason == "reduce_only_no_position"
        assert len(client.limit_calls) == 0
        assert len(client.market_calls) == 0


class TestCloseBoundedEscalation:
    """Verify chase logic when limit doesn't fill (MARKET removed)."""

    def test_close_position_bounded_chase_exhausted_on_timeout(self):
        client = FakeBinanceClientForBounded()
        # Make the limit order return as NEW (not filled)
        original_place_limit = client.place_limit_order

        def unfilled_limit(symbol, side, qty, price, post_only=False, **kwargs):
            client.limit_calls.append((symbol, side, qty, price))
            order_id = 3001 + len(client.limit_calls) - 1
            client._open_orders = [{"orderId": order_id, "symbol": symbol}]
            return {
                "orderId": order_id,
                "status": "NEW",
                "avgPrice": "0.0",
                "executedQty": "0.0",
            }

        client.place_limit_order = unfilled_limit
        # cancel_order clears open orders so chase can proceed
        original_cancel = client.cancel_order
        def verified_cancel(symbol, order_id):
            client.cancel_calls.append((symbol, order_id))
            client._open_orders = []
            return {"status": "CANCELED"}
        client.cancel_order = verified_cancel

        adapter = BinanceExecutionAdapter(client)

        result = adapter.close_position_bounded(
            "BTCUSDT",
            max_slippage_bps=50,
            fill_check_timeout_seconds=0.05,
        )

        # Chase exhausted — no MARKET fallback
        assert result.accepted is False
        assert "chase" in result.reason or "exhausted" in result.reason
        assert len(client.limit_calls) == 3  # 3 chase attempts
        assert len(client.market_calls) == 0  # No MARKET orders ever
        assert len(client.cancel_calls) == 3  # Cancelled each unfilled attempt


class TestReduceOnlyPathUsesBoundedExit:
    """Verify that place_order(reduce_only=True) routes through close_position_bounded."""

    def test_reduce_only_uses_bounded_exit(self):
        client = FakeBinanceClientForBounded()
        adapter = BinanceExecutionAdapter(client)

        plan = OrderPlan(symbol="BTCUSDT", side="SELL", quantity=0.5, reduce_only=True)
        result = adapter.place_order(plan, idempotency_key="ro_test")

        assert result.accepted is True
        assert "bounded" in result.reason or result.reason == "bounded_limit_exit"
        assert len(client.limit_calls) == 1
        assert len(client.market_calls) == 0
