import pandas as pd
import numpy as np
from quant.risk.cost_model import PercentageCostModel
from quant.validation.metrics import compute_trade_pnl
from quant.selection.threshold_optimizer import optimize_threshold

def test_cost_model_logic():
    # Setup data: avg vol = 1.0
    df = pd.DataFrame({'realized_vol_5': [1.0, 1.0, 1.0]})
    model = PercentageCostModel(fee_rate=0.001, vol_col='realized_vol_5', power=1.0)
    model.fit(df)

    assert model.avg_vol == 1.0

    # Base round-trip fee at close=100, fee_rate=0.1% = 0.2
    cost = model.estimate_cost(pd.Series({'close': 100.0, 'realized_vol_5': 0.5}))
    assert np.isclose(cost, 0.2)

    # Avg vol -> base fee
    cost = model.estimate_cost(pd.Series({'close': 100.0, 'realized_vol_5': 1.0}))
    assert np.isclose(cost, 0.2)

    # High vol (2x) -> 2x fee (power=1)
    cost = model.estimate_cost(pd.Series({'close': 100.0, 'realized_vol_5': 2.0}))
    assert np.isclose(cost, 0.4)

def test_vectorized_pnl_calculation():
    # Predictions: 3 trades, 2 valid (above 0.5)
    preds = np.array([0.9, 0.4, 0.8])
    moves = np.array([10.0, -5.0, 5.0])
    threshold = 0.5
    actuals = np.zeros(3) # unused by pnl calc directly
    
    # Case A: Static Spread = 1.0
    # Trade 0: 10 - 1 = 9
    # Trade 2: 5 - 1 = 4
    # Total = 13
    pnl_static = compute_trade_pnl(preds, actuals, moves, threshold, spread=1.0)
    assert len(pnl_static) == 2
    assert pnl_static.sum() == 13.0
    
    # Case B: Dynamic Spread
    # Trade 0 cost = 2.0 -> PnL = 10 - 2 = 8
    # Trade 2 cost = 3.0 -> PnL = 5 - 3 = 2
    costs = np.array([2.0, 1.0, 3.0])
    pnl_dyn = compute_trade_pnl(preds, actuals, moves, threshold, spread=costs)
    assert len(pnl_dyn) == 2
    assert pnl_dyn.sum() == 10.0


def test_bidirectional_pnl_calculation():
    preds = np.array([0.9, 0.1, 0.6, 0.4])
    moves = np.array([5.0, -4.0, 3.0, -2.0])
    actuals = np.zeros(4)

    pnl = compute_trade_pnl(
        predictions=preds,
        actuals=actuals,
        price_moves=moves,
        threshold=0.6,
        spread=1.0,
        allow_short=True,
    )

    # LONG: 5-1=4, 3-1=2 | SHORT: 4-1=3, 2-1=1
    assert len(pnl) == 4
    assert pnl.sum() == 10.0

def test_optimizer_with_costs():
    # 3 scenario trades repeated 4 times = 12 trades (> 10 min)
    # Trade A: P=0.9, Move=10, Cost=1.0 -> PnL=9
    # Trade B: P=0.8, Move=10, Cost=100.0 -> PnL=-90
    # Trade C: P=0.7, Move=10, Cost=1.0 -> PnL=9
    
    # 3 scenario trades repeated 10 times = 30 trades.
    # Trade A (P=0.9) will have 10 instances.
    
    preds = np.tile(np.array([0.9, 0.8, 0.7]), 10)
    moves = np.tile(np.array([10., 10., 10.]), 10)
    costs = np.tile(np.array([1., 100., 1.]), 10)
    
    # Threshold=0.85 -> Trade A only (10 trades) -> EV = 9.0 (Best)
    # Threshold=0.75 -> Trade A+B -> EV = (9-90)/2 = -40.5
    
    best_t, best_ev = optimize_threshold(
        preds, moves, costs, 
        threshold_min=0.6, threshold_max=0.9, threshold_step=0.05
    )
    
    # Check that optimizer picks the high threshold to avoid the expensive trade
    assert best_t >= 0.85
    assert best_ev == 9.0


def test_optimizer_with_costs_bidirectional():
    # Long-only prefers high thresholds due weak mid-confidence longs,
    # while bidirectional mode can exploit profitable short-side signals.
    preds = np.tile(np.array([0.9, 0.7, 0.3, 0.1]), 10)
    moves = np.tile(np.array([2.0, 0.0, -5.0, 1.0]), 10)
    costs = np.ones_like(moves)

    best_t_long, best_ev_long = optimize_threshold(
        preds,
        moves,
        costs,
        threshold_min=0.6,
        threshold_max=0.9,
        threshold_step=0.1,
        allow_short=False,
    )
    best_t_bi, best_ev_bi = optimize_threshold(
        preds,
        moves,
        costs,
        threshold_min=0.6,
        threshold_max=0.9,
        threshold_step=0.1,
        allow_short=True,
    )

    assert best_t_bi >= best_t_long
    assert best_ev_bi >= best_ev_long
    assert best_ev_bi > 0.0
