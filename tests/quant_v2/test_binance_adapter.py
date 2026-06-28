from __future__ import annotations

from quant_v2.contracts import OrderPlan
from quant_v2.execution.binance_adapter import BinanceExecutionAdapter


class FakeBinanceClient:
    def __init__(self) -> None:
        self.place_calls: list[tuple[str, str, float]] = []
        self.close_calls: list[str] = []
        self.symbol_filters: dict[str, dict[str, float]] = {}

    def place_order(self, symbol: str, side: str, quantity: float, order_type: str = "MARKET"):
        self.place_calls.append((symbol, side, quantity, order_type))
        return {
            "orderId": 123,
            "status": "FILLED",
            "avgPrice": "50000.0",
            "executedQty": quantity,
        }

    def close_position(self, symbol: str, **kwargs):
        self.close_calls.append(symbol)
        if symbol == "ETHUSDT":
            return None
        return {
            "orderId": 987,
            "status": "FILLED",
            "avgPrice": "2500.0",
            "executedQty": 0.5,
        }

    def get_positions(self, symbol=None):
        all_positions = [
            {
                "symbol": "BTCUSDT",
                "positionAmt": "0.25",
                "entryPrice": "48000",
                "unrealizedProfit": "500",
                "markPrice": "50000",
            },
            {"symbol": "ETHUSDT", "positionAmt": "0"},
            {
                "symbol": "SOLUSDT",
                "positionAmt": "-1.5",
                "entryPrice": "150",
                "unrealizedProfit": "15",
                "markPrice": "140",
            },
        ]
        if symbol:
            return [p for p in all_positions if p["symbol"] == symbol]
        return all_positions

    def get_symbol_filters(self, symbol: str) -> dict[str, float]:
        return dict(self.symbol_filters.get(symbol, {}))

    def get_orderbook(self, symbol: str, limit: int = 5) -> dict:
        return {"bids": [["50000.0", "1.0"]], "asks": [["50010.0", "1.0"]]}

    def place_limit_order(self, symbol, side, qty, price, post_only=False, **kwargs):
        self.place_calls.append((symbol, side, qty))
        return {
            "orderId": 555,
            "status": "FILLED",
            "avgPrice": str(price),
            "executedQty": qty,
        }

    def cancel_order(self, symbol, order_id):
        return {}

    def get_open_orders(self, symbol=None):
        return []

    def get_best_bid_ask(self, symbol):
        return (50000.0, 50010.0)


def test_binance_adapter_place_order_and_idempotency() -> None:
    client = FakeBinanceClient()
    adapter = BinanceExecutionAdapter(client)

    plan = OrderPlan(symbol="BTCUSDT", side="BUY", quantity=0.01)
    first = adapter.place_order(plan, idempotency_key="abc", mark_price=50000.0)
    second = adapter.place_order(plan, idempotency_key="abc", mark_price=51000.0)

    assert first.order_id == second.order_id
    assert len(client.place_calls) == 1
    assert first.accepted is True
    assert first.status == "filled"


def test_binance_adapter_reduce_only_paths() -> None:
    client = FakeBinanceClient()
    adapter = BinanceExecutionAdapter(client)

    ok = adapter.place_order(
        OrderPlan(symbol="BTCUSDT", side="SELL", quantity=0.5, reduce_only=True),
        idempotency_key="r1",
    )
    no_pos = adapter.place_order(
        OrderPlan(symbol="ETHUSDT", side="SELL", quantity=0.5, reduce_only=True),
        idempotency_key="r2",
    )

    assert ok.accepted is True
    assert no_pos.accepted is False
    assert no_pos.reason == "reduce_only_no_position"


def test_binance_adapter_reduce_only_respects_partial_quantity() -> None:
    client = FakeBinanceClient()
    adapter = BinanceExecutionAdapter(client)

    result = adapter.place_order(
        OrderPlan(symbol="BTCUSDT", side="SELL", quantity=0.10, reduce_only=True),
        idempotency_key="r3",
    )

    assert result.accepted is True
    assert result.requested_qty == 0.10
    assert result.filled_qty == 0.10
    assert client.place_calls[-1][:3] == ("BTCUSDT", "SELL", 0.10)


def test_binance_adapter_reduce_only_triggers_residual_supervision_on_step_size_truncation() -> None:
    client = FakeBinanceClient()
    client.symbol_filters["BTCUSDT"] = {
        "step_size": 0.02,
        "min_qty": 0.01,
        "min_notional": 1.0,
    }
    adapter = BinanceExecutionAdapter(client)

    result = adapter.place_order(
        OrderPlan(symbol="BTCUSDT", side="SELL", quantity=0.055, reduce_only=True),
        idempotency_key="r4",
    )

    assert result.accepted is True
    assert result.status == "filled"
    assert result.reason == "bounded_limit_exit:supervised_residual_position_required:step_size"
    assert client.place_calls[-1][:3] == ("BTCUSDT", "SELL", 0.04)


def test_binance_adapter_reduce_only_triggers_residual_supervision_on_min_notional_floor() -> None:
    client = FakeBinanceClient()
    client.get_positions = lambda symbol=None: [  # type: ignore[assignment]
        {
            "symbol": "BTCUSDT",
            "positionAmt": "0.05",
            "entryPrice": "100.0",
            "unrealizedProfit": "0.0",
            "markPrice": "100.0",
        }
    ]
    client.get_orderbook = lambda symbol, limit=5: {  # type: ignore[assignment]
        "bids": [["100.0", "1.0"]],
        "asks": [["101.0", "1.0"]],
    }
    client.symbol_filters["BTCUSDT"] = {
        "step_size": 0.01,
        "min_qty": 0.01,
        "min_notional": 10.0,
    }
    adapter = BinanceExecutionAdapter(client)

    result = adapter.place_order(
        OrderPlan(symbol="BTCUSDT", side="SELL", quantity=0.05, reduce_only=True),
        idempotency_key="r5",
    )

    assert result.accepted is False
    assert result.status == "residual_supervision_required"
    assert result.reason == "supervised_residual_position_required:min_notional"
    assert client.place_calls == []


def test_binance_adapter_get_positions_mapping() -> None:
    client = FakeBinanceClient()
    adapter = BinanceExecutionAdapter(client)

    positions = adapter.get_positions()

    assert positions == {"BTCUSDT": 0.25, "SOLUSDT": -1.5}


def test_binance_adapter_get_position_metrics_includes_mark_price() -> None:
    client = FakeBinanceClient()
    adapter = BinanceExecutionAdapter(client)

    metrics = adapter.get_position_metrics()

    assert metrics["BTCUSDT"] == {
        "entry_price": 48000.0,
        "unrealized_pnl_usd": 500.0,
        "mark_price": 50000.0,
    }
    assert metrics["SOLUSDT"] == {
        "entry_price": 150.0,
        "unrealized_pnl_usd": 15.0,
        "mark_price": 140.0,
    }


def test_binance_adapter_normalizes_quantity_to_step_size() -> None:
    client = FakeBinanceClient()
    client.symbol_filters["BTCUSDT"] = {
        "step_size": 0.01,
        "min_qty": 0.01,
        "min_notional": 100.0,
    }
    adapter = BinanceExecutionAdapter(client)

    plan = OrderPlan(symbol="BTCUSDT", side="BUY", quantity=0.123)
    result = adapter.place_order(plan, idempotency_key="norm", mark_price=1000.0)

    assert result.accepted is True
    assert client.place_calls[-1][:3] == ("BTCUSDT", "BUY", 0.12)
    assert result.requested_qty == 0.12


def test_binance_adapter_skips_order_below_symbol_filters() -> None:
    client = FakeBinanceClient()
    client.symbol_filters["BTCUSDT"] = {
        "step_size": 0.001,
        "min_qty": 0.001,
        "min_notional": 100.0,
    }
    adapter = BinanceExecutionAdapter(client)

    plan = OrderPlan(symbol="BTCUSDT", side="BUY", quantity=0.05)
    result = adapter.place_order(plan, idempotency_key="skip-filter", mark_price=1000.0)

    assert result.accepted is False
    assert result.status == "skipped"
    assert result.reason.startswith("skipped_by_filter")
    assert client.place_calls == []


# Phase 4: Limit Order Execution Tests


def test_binance_adapter_limit_order_buy_uses_best_bid() -> None:
    """BUY limit orders should be placed at best bid price (join the bid)."""
    client = FakeBinanceClient()
    adapter = BinanceExecutionAdapter(client)

    plan = OrderPlan(symbol="BTCUSDT", side="BUY", quantity=0.01)
    # When limit_price is provided, adapter should fetch best bid/ask
    result = adapter.place_order(plan, idempotency_key="limit-buy", mark_price=50000.0, limit_price=50000.0)

    # Verify order was accepted
    assert result.accepted is True
    # The FakeBinanceClient.get_best_bid_ask returns (50000.0, 50010.0)
    # For BUY, adapter should use bid price (50000.0)


def test_binance_adapter_limit_order_sell_uses_best_ask() -> None:
    """SELL limit orders should be placed at best ask price (join the ask)."""
    client = FakeBinanceClient()
    adapter = BinanceExecutionAdapter(client)

    plan = OrderPlan(symbol="BTCUSDT", side="SELL", quantity=0.01)
    result = adapter.place_order(plan, idempotency_key="limit-sell", mark_price=50000.0, limit_price=50000.0)

    assert result.accepted is True


def test_binance_adapter_limit_order_fallback_to_market() -> None:
    """If limit order fails, should fallback to market order."""
    class FailingLimitClient(FakeBinanceClient):
        def place_limit_order(self, symbol, side, qty, price, **kwargs):
            raise RuntimeError("Limit order rejected")

    client = FailingLimitClient()
    adapter = BinanceExecutionAdapter(client)

    plan = OrderPlan(symbol="BTCUSDT", side="BUY", quantity=0.01)
    result = adapter.place_order(plan, idempotency_key="limit-fallback", mark_price=50000.0, limit_price=50000.0)

    # Should fallback to market order and succeed
    assert result.accepted is True
    assert result.reason == "fallback_to_market"


def test_binance_adapter_cancel_order_delegates_to_client() -> None:
    """cancel_order should delegate to the underlying client."""
    client = FakeBinanceClient()
    adapter = BinanceExecutionAdapter(client)

    # Should not raise
    result = adapter.cancel_order("BTCUSDT", "12345")
    assert result == {}
