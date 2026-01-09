"""
Database repositories for immutable result persistence.

Uses raw SQL (no ORM) for append-only writes.
"""

import hashlib
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

from execution.simulator import CompletedTrade
from evaluation.policy import ResearchPolicy
from portfolio.models import PortfolioState

class EvaluationRepository:
    """
    Repository for storing evaluation results.
    
    All operations are append-only - no updates or deletes.
    """
    
    def __init__(self, db_path: str):
        """
        Initialize repository.
        
        Args:
            db_path: Path to SQLite database file
        """
        self._db_path = Path(db_path)
        self._ensure_schema()
    
    def _ensure_schema(self) -> None:
        """Ensure database schema exists."""
        schema_path = Path(__file__).parent / "schema.sql"
        
        with sqlite3.connect(self._db_path) as conn:
            with open(schema_path, 'r') as f:
                conn.executescript(f.read())
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection."""
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        return conn
    
    def store_hypothesis(
        self,
        hypothesis_id: str,
        parameters: dict,
        description: Optional[str] = None
    ) -> None:
        """
        Store hypothesis metadata (if not already stored).
        
        Args:
            hypothesis_id: Unique hypothesis identifier
            parameters: Hypothesis parameters
            description: Optional description
        """
        with self._get_connection() as conn:
            try:
                conn.execute(
                    """
                    INSERT INTO hypotheses (hypothesis_id, parameters_json, description)
                    VALUES (?, ?, ?)
                    """,
                    (hypothesis_id, json.dumps(parameters), description)
                )
                conn.commit()
            except sqlite3.IntegrityError:
                # Already exists, skip
                pass

    def store_policy(self, policy: ResearchPolicy) -> None:
        """
        Store a research policy.
        
        Args:
            policy: ResearchPolicy object
        """
        policy_hash = policy.compute_hash()
        policy_json = policy.model_dump_json()
        
        with self._get_connection() as conn:
            try:
                conn.execute(
                    """
                    INSERT INTO policies (policy_id, policy_hash, definition_json)
                    VALUES (?, ?, ?)
                    """,
                    (policy.policy_id, policy_hash, policy_json)
                )
                conn.commit()
            except sqlite3.IntegrityError:
                # Already exists. Check if hash matches?
                # For now, assume ID collision means same policy or we skip.
                pass
    
    def store_evaluation(
        self,
        hypothesis_id: str,
        parameters: dict,
        market_symbol: str,
        test_start_timestamp: datetime,
        test_end_timestamp: datetime,
        metrics: dict,
        benchmark_metrics: dict,
        assumed_costs_bps: float,
        initial_capital: float,
        final_equity: float,
        bars_processed: int,
        result_tag: Optional[str] = None,
        window_index: Optional[int] = None,
        window_start: Optional[datetime] = None,
        window_end: Optional[datetime] = None,
        window_type: Optional[str] = None,
        market_regime: Optional[str] = None,
        sample_type: Optional[str] = None,
        policy_id: Optional[str] = None,
        policy_hash: Optional[str] = None
    ) -> int:
        """
        Store evaluation results.
        
        Args:
            hypothesis_id: Hypothesis ID
            parameters: Hypothesis parameters
            market_symbol: Market symbol tested
            test_start_timestamp: Test period start
            test_end_timestamp: Test period end
            metrics: Evaluation metrics dictionary
            benchmark_metrics: Benchmark metrics dictionary
            assumed_costs_bps: Transaction costs used
            initial_capital: Starting capital
            final_equity: Final equity
            bars_processed: Number of bars processed
            result_tag: Optional tag for categorization
            window_index: Walk-forward window index
            window_start: Window start timestamp
            window_end: Window end timestamp
            window_type: Window type (TRAIN/TEST)
            market_regime: Market regime tag
            sample_type: Sample type tag
            policy_id: Research Policy ID
            policy_hash: Research Policy Hash
            
        Returns:
            Evaluation ID
        """
        # Create parameters hash
        params_str = json.dumps(parameters, sort_keys=True)
        params_hash = hashlib.sha256(params_str.encode()).hexdigest()[:16]
        
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO evaluations (
                    hypothesis_id,
                    parameters_hash,
                    market_symbol,
                    test_start_timestamp,
                    test_end_timestamp,
                    trade_count,
                    entry_count,
                    exit_count,
                    mean_return_per_trade,
                    sharpe_ratio,
                    max_drawdown,
                    profit_factor,
                    win_rate,
                    total_return,
                    total_pnl,
                    beta,
                    alpha,
                    information_ratio,
                    cagr,
                    benchmark_return_pct,
                    benchmark_pnl,
                    assumed_costs_bps,
                    initial_capital,
                    final_equity,
                    result_tag,
                    bars_processed,
                    average_trade_duration_days,
                    window_index,
                    window_start,
                    window_end,
                    window_type,
                    sample_type,
                    market_regime,
                    policy_id,
                    policy_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    hypothesis_id,
                    params_hash,
                    market_symbol,
                    test_start_timestamp,
                    test_end_timestamp,
                    metrics.get("trade_count", 0),
                    metrics.get("entry_count", 0),
                    metrics.get("exit_count", 0),
                    metrics.get("mean_return_per_trade"),
                    metrics.get("sharpe_ratio"),
                    metrics.get("max_drawdown"),
                    metrics.get("profit_factor"),
                    metrics.get("win_rate"),
                    metrics.get("total_return"),
                    metrics.get("total_pnl"),
                    metrics.get("beta"),
                    metrics.get("alpha"),
                    metrics.get("information_ratio"),
                    metrics.get("cagr"),
                    benchmark_metrics.get("benchmark_return_pct"),
                    benchmark_metrics.get("benchmark_pnl"),
                    assumed_costs_bps,
                    initial_capital,
                    final_equity,
                    result_tag,
                    bars_processed,
                    metrics.get("average_trade_duration_days"),
                    window_index,
                    window_start,
                    window_end,
                    window_type,
                    sample_type,
                    market_regime,
                    policy_id,
                    policy_hash
                )
            )
            conn.commit()
            last_row_id = cursor.lastrowid
            if last_row_id is None:
                raise RuntimeError("Failed to insert evaluation row.")
            return int(last_row_id)
    
    def store_trades(
        self,
        evaluation_id: int,
        trades: List[CompletedTrade]
    ) -> None:
        """
        Store trade records.
        
        Args:
            evaluation_id: Evaluation ID to associate trades with
            trades: List of completed trades
        """
        with self._get_connection() as conn:
            for trade in trades:
                conn.execute(
                    """
                    INSERT INTO trades (
                        evaluation_id,
                        trade_type,
                        side,
                        execution_price,
                        size,
                        execution_timestamp,
                        decision_timestamp,
                        cost_bps,
                        total_cost,
                        entry_price,
                        entry_timestamp,
                        realized_pnl,
                        trade_duration_days
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        evaluation_id,
                        trade.trade_type,
                        trade.side,
                        trade.execution_price,
                        trade.size,
                        trade.execution_timestamp,
                        trade.decision_timestamp,
                        trade.cost_bps,
                        trade.total_cost,
                        trade.entry_price,
                        trade.entry_timestamp,
                        trade.realized_pnl,
                        trade.trade_duration_days
                    )
                )
            conn.commit()
    
    def get_evaluations(
        self,
        hypothesis_id: Optional[str] = None,
        market_symbol: Optional[str] = None,
        limit: int = 100
    ) -> List[dict]:
        """
        Query evaluation results.
        
        Args:
            hypothesis_id: Filter by hypothesis ID
            market_symbol: Filter by market symbol
            limit: Maximum results to return
            
        Returns:
            List of evaluation records as dictionaries
        """
        query = "SELECT * FROM evaluations WHERE 1=1"
        params: List[Any] = []
        
        if hypothesis_id:
            query += " AND hypothesis_id = ?"
            params.append(hypothesis_id)
        
        if market_symbol:
            query += " AND market_symbol = ?"
            params.append(market_symbol)
        
        query += " ORDER BY evaluation_run_timestamp DESC LIMIT ?"
        params.append(limit)
        
        with self._get_connection() as conn:
            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def store_batch(self, batch_id: str, market_symbol: str, config_hash: str, policy_id: Optional[str] = None) -> None:
        """Store batch configuration metadata."""
        with self._get_connection() as conn:
            try:
                conn.execute(
                    """
                    INSERT INTO batches (batch_id, market_symbol, config_hash, policy_id)
                    VALUES (?, ?, ?, ?)
                    """,
                    (batch_id, market_symbol, config_hash, policy_id)
                )
                conn.commit()
            except sqlite3.IntegrityError:
                pass

    def store_batch_ranking(self, ranking: dict) -> None:
        """Store a single hypothesis ranking."""
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO batch_rankings (
                    batch_id, hypothesis_id, research_score, rank
                )
                VALUES (?, ?, ?, ?)
                """,
                (
                    ranking['batch_id'], 
                    ranking['hypothesis_id'], 
                    ranking['research_score'], 
                    ranking['rank']
                )
            )
            conn.commit()

    def store_hypothesis_status(
        self,
        hypothesis_id: str,
        status: str,
        batch_id: Optional[str] = None,
        policy_id: Optional[str] = None,
        rationale: Optional[List[str]] = None
    ) -> None:
        """Log a hypothesis status change."""
        rationale_json = json.dumps(rationale) if rationale else None
        
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO hypothesis_status_history (
                    hypothesis_id, batch_id, policy_id, status, rationale_json
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (hypothesis_id, batch_id, policy_id, status, rationale_json)
            )
            conn.commit()

    def get_hypotheses_by_status(self, status: str, policy_id: Optional[str] = None) -> List[str]:
        """
        Get all hypotheses with a specific current status.
        
        Args:
            status: HypothesisStatus string (e.g., "PROMOTED")
            policy_id: Optional filter by policy context
            
        Returns:
            List of hypothesis IDs
        """
        query = """
            SELECT h1.hypothesis_id
            FROM hypothesis_status_history h1
            WHERE h1.status = ?
            AND h1.rowid = (
                SELECT MAX(h2.rowid)
                FROM hypothesis_status_history h2
                WHERE h2.hypothesis_id = h1.hypothesis_id
            )
        """
        params = [status]
        
        if policy_id:
            query += " AND h1.policy_id = ?"
            params.append(policy_id)
            
        with self._get_connection() as conn:
            cursor = conn.execute(query, params)
            return [row['hypothesis_id'] for row in cursor.fetchall()]

    def get_latest_evaluation(self, hypothesis_id: str, policy_id: Optional[str] = None) -> Optional[dict]:
        """
        Get the most recent evaluation for a hypothesis.
        
        Args:
            hypothesis_id: Hypothesis ID
            policy_id: Optional filter by policy ID
            
        Returns:
            Dictionary with evaluation details or None
        """
        query = """
            SELECT *
            FROM evaluations
            WHERE hypothesis_id = ?
        """
        params = [hypothesis_id]
        
        if policy_id:
            query += " AND policy_id = ?"
            params.append(policy_id)
            
        query += " ORDER BY test_end_timestamp DESC LIMIT 1"
        
        with self._get_connection() as conn:
            cursor = conn.execute(query, params)
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_hypothesis_details(self, hypothesis_id: str) -> Optional[dict]:
        """
        Get hypothesis details including parameters.
        
        Args:
            hypothesis_id: Hypothesis ID
            
        Returns:
            Dictionary with hypothesis details (including parameters_json)
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT * FROM hypotheses WHERE hypothesis_id = ?
                """,
                (hypothesis_id,)
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    def store_portfolio_evaluation(self, state: PortfolioState, portfolio_tag: str, policy_id: str) -> None:
        """
        Store a portfolio state snapshot.
        
        Args:
            state: PortfolioState object
            portfolio_tag: Tag for grouping (e.g. run ID)
            policy_id: Policy ID
        """
        import json
        
        allocations_snapshot = {
            hid: {
                "capital": alloc.allocated_capital,
                "unrealized_pnl": alloc.unrealized_pnl,
                "realized_pnl": alloc.realized_pnl,
                "position_size": alloc.current_position.size if alloc.current_position else 0
            } 
            for hid, alloc in state.allocations.items()
        }
        
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO portfolio_evaluations (
                    portfolio_tag, timestamp, total_capital, cash,
                    realized_pnl, unrealized_pnl, drawdown_pct,
                    allocations_json, policy_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    portfolio_tag,
                    state.timestamp,
                    state.total_capital,
                    state.cash,
                    state.total_realized_pnl,
                    state.total_unrealized_pnl,
                    state.drawdown_pct,
                    json.dumps(allocations_snapshot),
                    policy_id
                )
            )
            conn.commit()
