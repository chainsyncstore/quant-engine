"""Tests for Fix 2: Zombie order prevention & bounded chase logic.

Validates that close_position_bounded uses a multi-attempt chase loop,
never calls MARKET orders, and aborts cleanly on cancel failures.
"""

from __future__ import annotations

from quant_v2.execution.binance_adapter import BinanceExecutionAdapter


class FakeBinanceClientChase:
    """Fake client whose limit orders never fill, requiring chase logic."""

    def __init__(self) -> None:
        self.limit_calls: list[tuple[str, str, float, float]] = []
        self.market_calls: list[tuple] = []
        self.cancel_calls: list[tuple[str, int]] = []
        self._open_orders: list[dict] = []
        self._cancel_raises: bool = False
        self._cancel_returns_bad_status: bool = False

    def get_positions(self, symbol=None):
        return [{"symbol": "BTCUSDT", "positionAmt": "0.5"}]

    def place_limit_order(self, symbol, side, qty, price, post_only=False, **kw):
        order_id = 3001 + len(self.limit_calls)
        self.limit_calls.append((symbol, side, qty, price))
        self._open_orders = [{"orderId": order_id, "symbol": symbol}]
        return {
            "orderId": order_id,
            "status": "NEW",
            "avgPrice": "0.0",
            "executedQty": "0.0",
        }

    def place_order(self, symbol, side, qty, order_type="MARKET"):
        self.market_calls.append((symbol, side, qty, order_type))
        return {
            "orderId": 9999,
            "status": "FILLED",
            "avgPrice": "49000.0",
            "executedQty": str(qty),
        }

    def get_open_orders(self, symbol=None):
        return list(self._open_orders)

    def cancel_order(self, symbol, order_id):
        self.cancel_calls.append((symbol, order_id))
        if self._cancel_raises:
            raise ConnectionError("Binance API 500")
        if self._cancel_returns_bad_status:
            return {"status": "PENDING_CANCEL"}
        self._open_orders = []
        return {"status": "CANCELED"}

    def get_symbol_filters(self, symbol):
        return {"tick_size": "0.01", "step_size": "0.001"}

    def get_orderbook(self, symbol, limit=5):
        return {
            "bids": [["50000.0", "1.0"]],
            "asks": [["50010.0", "1.0"]],
        }


class TestChaseRetriesOnUnfilled:
    """Bounded chase must retry up to 3 times, never using MARKET."""

    def test_chase_retries_on_unfilled_order(self):
        client = FakeBinanceClientChase()
        adapter = BinanceExecutionAdapter(client)

        result = adapter.close_position_bounded(
            "BTCUSDT",
            max_slippage_bps=50,
            fill_check_timeout_seconds=0.05,
        )

        # Chase exhausted — all 3 attempts failed to fill
        assert result.accepted is False
        assert "chase" in result.reason or "exhausted" in result.reason
        assert len(client.limit_calls) == 3
        assert len(client.market_calls) == 0, "MARKET orders must never be used"

    def test_no_market_orders_ever(self):
        """Regardless of chase outcome, place_order(MARKET) is never called."""
        client = FakeBinanceClientChase()
        adapter = BinanceExecutionAdapter(client)

        adapter.close_position_bounded(
            "BTCUSDT",
            max_slippage_bps=50,
            fill_check_timeout_seconds=0.05,
        )

        assert len(client.market_calls) == 0


class TestChaseAbortsOnCancelFailure:
    """If cancel_order raises or returns bad status, chase aborts to prevent zombie orders."""

    def test_chase_aborts_on_cancel_exception(self):
        client = FakeBinanceClientChase()
        client._cancel_raises = True
        adapter = BinanceExecutionAdapter(client)

        result = adapter.close_position_bounded(
            "BTCUSDT",
            max_slippage_bps=50,
            fill_check_timeout_seconds=0.05,
        )

        assert result.accepted is False
        assert "cancel_verification_failed" in result.reason
        # Should abort after 1st attempt's cancel fails
        assert len(client.limit_calls) == 1
        assert len(client.market_calls) == 0

    def test_chase_aborts_on_bad_cancel_status(self):
        client = FakeBinanceClientChase()
        client._cancel_returns_bad_status = True
        adapter = BinanceExecutionAdapter(client)

        result = adapter.close_position_bounded(
            "BTCUSDT",
            max_slippage_bps=50,
            fill_check_timeout_seconds=0.05,
        )

        assert result.accepted is False
        assert "cancel_verification_failed" in result.reason
        assert len(client.limit_calls) == 1
        assert len(client.market_calls) == 0


class TestChaseWidensSlippage:
    """Each chase attempt should use a wider slippage to increase fill probability."""

    def test_chase_widens_slippage_per_attempt(self):
        client = FakeBinanceClientChase()
        adapter = BinanceExecutionAdapter(client)

        adapter.close_position_bounded(
            "BTCUSDT",
            max_slippage_bps=50,
            fill_check_timeout_seconds=0.05,
        )

        assert len(client.limit_calls) == 3
        prices = [call[3] for call in client.limit_calls]

        # For a SELL (long exit), each chase price should be lower (more aggressive)
        for i in range(1, len(prices)):
            assert prices[i] < prices[i - 1], (
                f"Chase attempt {i+1} price {prices[i]} must be lower than "
                f"attempt {i} price {prices[i-1]}"
            )
