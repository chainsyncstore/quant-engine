from __future__ import annotations

from quant_v2.contracts import OrderPlan
from quant_v2.execution.binance_adapter import BinanceExecutionAdapter


class FakeBinanceClient:
    def __init__(self) -> None:
        self.place_calls: list[tuple[str, str, float]] = []
        self.close_calls: list[str] = []

    def place_order(self, symbol: str, side: str, quantity: float):
        self.place_calls.append((symbol, side, quantity))
        return {
            "orderId": 123,
            "status": "FILLED",
            "avgPrice": "50000.0",
            "executedQty": quantity,
        }

    def close_position(self, symbol: str):
        self.close_calls.append(symbol)
        if symbol == "ETHUSDT":
            return None
        return {
            "orderId": 987,
            "status": "FILLED",
            "avgPrice": "2500.0",
            "executedQty": 0.5,
        }

    def get_positions(self):
        return [
            {"symbol": "BTCUSDT", "positionAmt": "0.25"},
            {"symbol": "ETHUSDT", "positionAmt": "0"},
            {"symbol": "SOLUSDT", "positionAmt": "-1.5"},
        ]


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
    assert client.close_calls == ["BTCUSDT", "ETHUSDT"]


def test_binance_adapter_get_positions_mapping() -> None:
    client = FakeBinanceClient()
    adapter = BinanceExecutionAdapter(client)

    positions = adapter.get_positions()

    assert positions == {"BTCUSDT": 0.25, "SOLUSDT": -1.5}
