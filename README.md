# Hypothesis Research Engine (HRE v0)

A research-grade, deterministic market replay engine designed to evaluate trading hypotheses offline without look-ahead bias.

## Overview

The Hypothesis Research Engine (HRE) replays historical market data bar-by-bar, advancing a single global clock and invoking user-defined hypotheses at each step. Hypotheses observe only past and present state and emit trade intents; they do not execute trades, store data, or access future information.

All execution, cost modeling, evaluation, and persistence are handled outside the hypothesis layer and occur strictly after decisions are made.

## Key Features

- **No Look-Ahead Bias**: Decisions made using only information available up to the current bar
- **Deterministic**: Same inputs always produce identical outputs
- **Separation of Concerns**: Hypothesis logic isolated from execution, storage, and evaluation
- **Replay-First Architecture**: Time advances only through replayed market bars
- **Write-Once Results**: Evaluation outputs are immutable

## Quick Start

### Installation

```bash
# Install dependencies
pip install -e .

# Install development dependencies
pip install -e ".[dev]"
```

### Running an Evaluation

```bash
# Preferred wrapper (thin CLI that calls orchestrator.run_evaluation.main)
python scripts/run_evaluation.py \
    --hypothesis always_long \
    --policy WF_V1 \
    --data-path data/sample_market_data.csv \
    --symbol SAMPLE \
    --start-date 2020-01-01 \
    --end-date 2023-12-31

# Direct module execution is still available if you prefer:
# python -m orchestrator.run_evaluation ...
```

### Other CLI Entry Points

All operational CLIs now live under `scripts/` as thin wrappers around their respective package modules:

```bash
python scripts/run_batch.py --help         # orchestrator.run_batch
python scripts/run_meta.py --help          # orchestrator.run_meta
python scripts/run_portfolio.py --help     # orchestrator.run_portfolio
python scripts/check_decay.py --help       # orchestrator.check_decay
python scripts/run_batch_runner.py --help  # batch.run_batch (legacy batch runner)
```

### Creating a Custom Hypothesis

```python
from hypotheses.base import Hypothesis, TradeIntent, IntentType
from state.market_state import MarketState
from state.position_state import PositionState
from clock.clock import Clock

class MyHypothesis(Hypothesis):
    @property
    def hypothesis_id(self) -> str:
        return "my_hypothesis"
    
    @property
    def parameters(self) -> dict:
        return {"version": "1.0"}
    
    def on_bar(
        self, 
        market_state: MarketState, 
        position_state: PositionState, 
        clock: Clock
    ) -> TradeIntent | None:
        # Your logic here
        if not position_state.has_position:
            return TradeIntent(type=IntentType.BUY, size=1.0)
        return None
```

Register your hypothesis in `hypotheses/registry.py`.

## Architecture

The system follows a strict layered architecture:

1. **Data Layer**: Loads and validates OHLCV market data
2. **Clock Module**: Single source of time truth
3. **State Management**: Tracks market history and positions
4. **Hypothesis Layer**: User-defined trading logic (isolated)
5. **Engine Core**: Main replay loop and decision queue
6. **Execution Layer**: Simulates trade execution with costs
7. **Evaluation Layer**: Computes metrics (Sharpe, drawdown, etc.)
8. **Storage Layer**: Immutable persistence of results

## Testing

```bash
pytest tests/ -v
```

## Documentation

- **PRD.md**: Complete product requirements
- **ARCHITECTURE.md**: System boundaries and contracts

## Non-Goals

This system does NOT:
- Connect to live broker APIs or place real trades
- Optimize for execution latency
- Perform portfolio-level capital allocation
- Use ML/RL techniques
- Manage multiple concurrent positions (v0)

## License

Research use only.
