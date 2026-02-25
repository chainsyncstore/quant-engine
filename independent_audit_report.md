# Independent System Audit Report: Hypothesis Research Engine (v2)

## Executive Summary
**Real-World Profitability & Performance Score:** **35 / 100**

While the overall architectural skeleton, risk management constraints, and backtesting pipelines are exceptionally well-designed (institutional grade), there is a **fatal logic oversight** in the live execution layer that functionally bypasses the advanced Machine Learning stack entirely. If deployed in the real world *as-is*, the system would abandon its statistically defensible models and trade using a naive, hardcoded moving average crossover, rendering it highly vulnerable to chop and spread decay. If the ML pipeline integration flaw is fixed, the score would easily jump to **80+/100**, as the risk architecture itself is excellent.

---

## 🛑 Critical Failures & Logic Issues

### 1. ML Stack Bypass in Live Native Loop (Fatal Oversight)
**Location:** `quant_v2/telebot/signal_manager.py` (`_build_signal_payload` method)
**Issue:** The system features an advanced model trainer (`trainer.py`) supporting LightGBM, fold-local sigmoid calibration, logistic meta-models, and uncertainty quantification. However, the native telegram signal loop completely ignores these models. Instead, it computes primitive indicators in-line (`ema_fast`, `ema_slow`, `momentum`) and produces a hardcoded score:
```python
score = (momentum * 450.0) + (short_return * 120.0)
proba_up = min(max(0.5 + score, 0.01), 0.99)
```
**Impact:** Live trading is currently operating on an exceptionally basic, non-optimized heuristic rather than your thoroughly researched LightGBM predictors.

### 2. Adverse Selection & Slippage in Order Routing
**Location:** `quant_v2/execution/binance_adapter.py` (`place_order`)
**Issue:** The adapter attempts to capture the spread by submitting POST_ONLY Limit orders precisely at the mark price. If the Binance engine rejects it with a `-2010` error (meaning it would immediately match/take liquidity), you gracefully catch the error and immediately fall back to a `MARKET` order.
**Impact:** 
- Submitting a limit order exactly at the mark implies a high risk of adverse selection (it only gets filled if the market moves against you). 
- Instantly falling back to a `MARKET` order guarantees execution but also guarantees paying taker fees and swallowing slippage, completely negating the original maker-fee routing strategy. This will erode profitability, especially for a high-turnover strategy.

### 3. Vague Rebalance Deadbands
**Location:** `quant_v2/execution/service.py` (`route_signals`)
**Issue:** The deadband checks for minimum rebalance drift (`min_weight_drift`) are functional, but `min_rebalance_weight_drift` limits are compared directly against Notional USD changes scaled by equity. On crypto pairs with high short-term volatility, tiny fractional drifts may constantly trigger rebalances, resulting in fee bleed.

---

## ✅ System Strengths (What is done right)

1. **Portfolio Risk Constraints (`risk_policy.py`)** 
   - State-of-the-art exposure controls. The fact that the system restricts `max_symbol_exposure_frac`, `max_gross_exposure_frac`, and elegantly auto-balances the dominant long/short side to adhere to the `max_net_exposure_frac` is brilliant.
2. **Purged Group Cross-Validation (`stage1_pipeline.py`)**
   - Utilizing embargoes and group purge CV drastically reduces data leakage in crypto time-series data, ensuring the backtests (if used!) are actually reliable.
3. **Execution Safety & Killswitches (`service.py`)**
   - The separation between planning, reconciling, and routing allows cleanly intercepting execution anomalies and pausing via the `KillSwitchEvaluation`. 
4. **Idempotency Execution**
   - Implementing `InMemoryIdempotencyJournal` to prevent double-spending/double-ordering on network retries is a crucial safety mechanism often overlooked by retail bots.

---

## 🛠️ Recommendations for Refinement

1. **Integrate Predictor into Signal Manager:**
   Rewrite `signal_manager.py` (or inject the predictor directly) so it calls `quant_v2.models.predictor.predict_proba()` loading the active model from `model_registry.py` instead of calculating hardcoded `ema` scores.
2. **Revise Execution Logic:**
   Instead of falling back to a `MARKET` order immediately upon a limit rejection, either:
   - Adjust the limit price sequentially (chasing) within an acceptable slippage bound.
   - Use smart algorithmic execution (like TWAP) for large exposures instead of dumping at market.
3. **Fee Accounting:**
   Ensure the portfolio snapshots and PnL trackers deduct standard execution fees (0.05% maker / 0.10% taker), as directional models with tight horizons (e.g. 4-bar) can easily get chopped to death by Binance margin fees alone.
