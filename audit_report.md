# Hypothesis Research Engine - Full System Audit

## Real-World Performance & Profitability Score: **65 / 100**

**TL;DR:** The engineering, infrastructure, and control-plane architecture of this system are exceptionally well-built for a retail/independent quant setup (easily an **85-90/100** on software engineering alone). Features like the Telegram bot control, execution killswitches, SQLite WAL persistence, AWS deployment tooling, and rigid cross-margin risk controls (Gross/Net/Symbol buckets) are production-grade. 

However, the **Quant/Trading Strategy and Execution** bring the overall real-world profitability expectation down significantly. Relying purely on **Long-Only** signals, **Market Orders (Taker Fees)**, and traditional ML over basic OHLCV/OI features makes it extremely difficult to outpace slippage and market regimes in crypto futures.

Below is a deep dive into the failures, logic issues, and oversights found across the codebase.

---

### 1. Market Execution & Slippage (Critical Profitability Drag)
* **Taker-Only Execution:** In `quant/data/binance_client.py`, the `place_order` wrapper defaults to `order_type="MARKET"`. The Binance Futures taker fee is typically `0.04%` to `0.05%` per trade. A round-trip costs `~0.10%` purely in fixed fees, entirely excluding bid-ask spread and slippage. For lower-timeframe models (like the 3m/5m/10m horizons defined in `ensemble.py`), the expected alpha per trade is usually too small to consistently beat a `0.10%` hurdle.
* **No Maker/Limit Order Implementation:** A highly profitable real-world system on short-to-medium horizons actively captures the spread or rests limit orders (Maker fees are `0.01%` or `0.02%`). Market-only execution is a severe oversight that bleeds capital.

### 2. Strategy Logic & "Long-Only" Constraint
* **Missed Alpha in Bear Regimes:** In `quant/models/ensemble.py`, the consensus logic enforces `signal_val = 1` if the required votes are met, noting *"Currently only Long strategies implemented"*.
* **Crypto Markets are Cyclical:** Crypto futures derive massive utility from shorting during market downturns. By restricting the bot to Long-Only, the system relies entirely on flat allocation (holding cash) during bearish regimes. This drastically reduces the Sharpe ratio and overall capital efficiency.

### 3. Feature Leakage Risk in Pipeline
* **Blacklist vs. Whitelist:** In `quant/features/pipeline.py`, non-feature columns are filtered out using a hardcoded `_NON_FEATURE_COLS` blacklist (open, high, EMA cols, etc.). 
  * **Oversight:** If a researcher adds a new intermediate variable or a forward-looking label inside a feature module (like `order_flow.py` or `liquidation.py`) but forgets to explicitly add the column name to `_NON_FEATURE_COLS` or prefix it with `label_`, that column will silently leak into the training dataset. Over time, this guarantees Look-Ahead Bias. 
  * **Fix:** You should use an explicit **Whitelist** of expected feature columns, rather than a blacklist of known non-features.

### 4. Model Training Redundancies
* **CalibratedClassifierCV Double-Fitting:** In `quant/models/trainer.py`, the LightGBM classifier `lgbm_raw` is manually `.fit()` to extract feature importances and perform pruning. Immediately afterward, it is passed into `CalibratedClassifierCV(estimator=lgbm_raw, cv=ps)`, which internally clones the base estimator and fits it **again** on the exact same training split. While this does not break the logic, it effectively doubles the training time for the LightGBM tree building phase.

### 5. API Chunking and Hardcoded Limits
* **Binance Open Interest Endpoint:** In `binance_client.py#fetch_open_interest`, the chunking uses `CHUNK_MS = 29 * 24 * 3600 * 1000`. Binance occasionally enforces strict month-boundary or 30-day limits. The 29-day fetch loop safely works, but any disconnect could cause silent data holes because missing periods are just concatenated out without strict validation on continuity.
* **Rebalance Deadbands vs. Portfolio Drift:** In `service.py`, `BOT_V2_MIN_REBALANCE_NOTIONAL_USD` defaults to $10.0. While useful for avoiding micro-trades, this static dollar threshold can be problematic depending on the portfolio equity. A percentage-based deadband (e.g., `rebalance if weight drift > 1%`) is significantly safer for dynamic portfolios than a hardcoded minimal dollar value.

### 6. Risk Policy Adjustments (Mathematical Observation)
* In `quant_v2/portfolio/risk_policy.py#apply`, the sequence of caps is: `Symbol -> Bucket -> Gross -> Net`.
* This order of operations is mathematically sound and safe because all scales are `<= 1.0` (it never scales a position *up*). If the Gross Cap scales everything down, and the resulting absolute sum of longs/shorts violates the Net Cap, the Net Cap applies a further fractional reduction to the dominant side. This ensures strict adherence to all risk boundaries simultaneously without infinite loops. **Great engineering here.**

### Summary Recommendations for Real-World Deployment
1. **Implement Limit Orders (Post-Only):** Revamp `BinanceClient` and `RoutedExecutionService` to submit Limit orders at the bid/ask and manage their lifecycle (chasers, timeouts).
2. **Train Short Models:** The ensemble architecture easily supports it. Introduce `-1` labels to trade bearish regimes.
3. **Migrate to Feature Whitelists:** Eradicate any risk of lookahead bias by defining exactly which features the model is allowed to see during `.predict()`.
4. **Percentage-based Rebalance Deadbands:** Avoid fixed `$10` limits; switch to relative portfolio weight tolerances.
