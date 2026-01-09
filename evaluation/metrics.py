"""
Evaluation metrics computation.

Computes performance metrics from completed trades.
"""

import numpy as np
import pandas as pd
from typing import List, Optional

from execution.simulator import CompletedTrade


class EvaluationMetrics:
    """
    Computes evaluation metrics from trade history.
    
    All metrics are computed after replay completes.
    """
    
    def __init__(
        self,
        completed_trades: List[CompletedTrade],
        initial_capital: float,
        final_capital: float,
        equity_curve: Optional[List[float]] = None,
        benchmark_curve: Optional[List[float]] = None
    ):
        """
        Initialize metrics calculator.
        
        Args:
            completed_trades: List of all completed trades
            initial_capital: Starting capital
            final_capital: Final capital (including unrealized PnL if needed)
            equity_curve: Optional list of daily total equity values
            benchmark_curve: Optional list of daily benchmark equity values
        """
        self._trades = completed_trades
        self._initial_capital = initial_capital
        self._final_capital = final_capital
        self._equity_curve = equity_curve if equity_curve else []
        self._benchmark_curve = benchmark_curve if benchmark_curve else []
        self._sample_type: Optional[str] = None

    def beta(self) -> float:
        """Calculate Beta relative to benchmark."""
        if not self._equity_curve or not self._benchmark_curve:
            return 0.0
            
        returns_strat = pd.Series(self._equity_curve).pct_change().dropna()
        returns_bench = pd.Series(self._benchmark_curve).pct_change().dropna()
        
        # Align lengths
        min_len = min(len(returns_strat), len(returns_bench))
        if min_len < 2:
            return 0.0
            
        returns_strat = returns_strat.iloc[:min_len]
        returns_bench = returns_bench.iloc[:min_len]
        
        cov = np.cov(returns_strat, returns_bench)[0, 1]
        var = np.var(returns_bench)
        
        if var == 0:
            return 0.0
            
        return cov / var

    def alpha(self, risk_free_rate: float = 0.0) -> float:
        """Calculate Alpha (Annualized) assuming 252 days."""
        if not self._equity_curve or not self._benchmark_curve:
            return 0.0
            
        beta = self.beta()
        
        # Annualized Returns
        strat_cagr = self.cagr()
        bench_cagr = self._calculate_cagr(self._benchmark_curve)
        
        # Jensen's Alpha: R_p - (R_f + Beta * (R_m - R_f))
        return strat_cagr - (risk_free_rate + beta * (bench_cagr - risk_free_rate))

    def information_ratio(self) -> float:
        """Calculate Information Ratio."""
        if not self._equity_curve or not self._benchmark_curve:
            return 0.0
            
        returns_strat = pd.Series(self._equity_curve).pct_change().dropna()
        returns_bench = pd.Series(self._benchmark_curve).pct_change().dropna()
        
        min_len = min(len(returns_strat), len(returns_bench))
        if min_len < 2:
            return 0.0
            
        active_returns = returns_strat.iloc[:min_len] - returns_bench.iloc[:min_len]
        
        mean_active = np.mean(active_returns)
        std_active = np.std(active_returns)
        
        if std_active == 0:
            return 0.0
            
        # Annualize (assuming daily)
        return (mean_active / std_active) * np.sqrt(252)

    def cagr(self) -> float:
        """Calculate CAGR."""
        return self._calculate_cagr(self._equity_curve)

    def _calculate_cagr(self, curve: List[float]) -> float:
        if not curve:
            return 0.0
        start = curve[0]
        end = curve[-1]
        bars = len(curve)
        years = bars / 252.0
        if start == 0 or years == 0:
            return 0.0
        return (end / start) ** (1 / years) - 1

    def set_sample_type(self, sample_type: str):
        """Set sample type (IN_SAMPLE | OUT_OF_SAMPLE | MONITORING)."""
        if sample_type not in ["IN_SAMPLE", "OUT_OF_SAMPLE", "MONITORING"]:
            raise ValueError(f"Invalid sample type: {sample_type}")
        self._sample_type = sample_type
        
    @property
    def sample_type(self) -> Optional[str]:
        """Get sample type."""
        return self._sample_type
    
    def trade_count(self) -> int:
        """
        Count total number of trades (entries + exits).
        
        Returns:
            Number of trades
        """
        return len(self._trades)
    
    def entry_count(self) -> int:
        """Count number of entries."""
        return len([t for t in self._trades if t.trade_type == "ENTRY"])
    
    def exit_count(self) -> int:
        """Count number of exits."""
        return len([t for t in self._trades if t.trade_type == "EXIT"])
    
    def get_closed_trades(self) -> List[CompletedTrade]:
        """Get only exit trades (which have realized PnL)."""
        return [t for t in self._trades if t.trade_type == "EXIT"]
    
    def mean_return_per_trade(self) -> float:
        """
        Calculate mean return per closed trade.
        
        Returns:
            Mean realized PnL, or 0.0 if no closed trades
        """
        closed_trades = self.get_closed_trades()
        
        if not closed_trades:
            return 0.0
        
        total_pnl = sum(t.realized_pnl for t in closed_trades if t.realized_pnl is not None)
        return total_pnl / len(closed_trades)
    
    def sharpe_ratio(self, risk_free_rate: float = 0.0) -> float:
        """
        Calculate Sharpe ratio from trade returns.
        
        Args:
            risk_free_rate: Annual risk-free rate (default 0.0)
            
        Returns:
            Sharpe ratio, or 0.0 if not enough trades
        """
        closed_trades = self.get_closed_trades()
        
        if len(closed_trades) < 2:
            return 0.0
        
        # Calculate return percentages
        returns = [
            (t.realized_pnl / t.entry_price / t.size) * 100.0
            for t in closed_trades
            if t.realized_pnl is not None and t.entry_price is not None and t.size > 0
        ]
        
        if not returns:
            return 0.0
        
        mean_return = float(np.mean(returns))
        std_return = float(np.std(returns, ddof=1))
        
        if std_return == 0:
            return 0.0
        
        # Simple Sharpe (not annualized)
        return (mean_return - risk_free_rate) / std_return
    
    def max_drawdown(self) -> float:
        """
        Calculate maximum drawdown from equity curve.
        
        Returns:
            Maximum drawdown as a percentage
        """
        closed_trades = self.get_closed_trades()
        
        if not closed_trades:
            return 0.0
        
        # Build equity curve
        equity = self._initial_capital
        equity_curve = [equity]
        
        for trade in closed_trades:
            if trade.realized_pnl is not None:
                equity += trade.realized_pnl
                equity_curve.append(equity)
        
        # Calculate drawdown
        peak = float(equity_curve[0])
        max_dd = 0.0
        
        for raw_value in equity_curve:
            value = float(raw_value)
            if value > peak:
                peak = value
            
            dd = ((peak - value) / peak) * 100.0 if peak > 0 else 0.0
            max_dd = max(max_dd, dd)
        
        return max_dd
    
    def profit_factor(self) -> float:
        """
        Calculate profit factor (gross profit / gross loss).
        
        Returns:
            Profit factor, or 0.0 if no losing trades
        """
        closed_trades = self.get_closed_trades()
        
        if not closed_trades:
            return 0.0
        
        gross_profit = sum(
            t.realized_pnl for t in closed_trades
            if t.realized_pnl is not None and t.realized_pnl > 0
        )
        
        gross_loss = abs(sum(
            t.realized_pnl for t in closed_trades
            if t.realized_pnl is not None and t.realized_pnl < 0
        ))
        
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0
        
        return gross_profit / gross_loss
    
    def win_rate(self) -> float:
        """
        Calculate win rate (percentage of profitable trades).
        
        Returns:
            Win rate as percentage, or 0.0 if no trades
        """
        closed_trades = self.get_closed_trades()
        
        if not closed_trades:
            return 0.0
        
        winning_trades = sum(
            1 for t in closed_trades
            if t.realized_pnl is not None and t.realized_pnl > 0
        )
        
        return (winning_trades / len(closed_trades)) * 100.0
    
    def total_return(self) -> float:
        """
        Calculate total return as percentage.
        
        Returns:
            Total return percentage
        """
        if self._initial_capital == 0:
            return 0.0
        
        return ((self._final_capital - self._initial_capital) / self._initial_capital) * 100.0
    
    def final_equity(self) -> float:
        """Get final equity."""
        return self._final_capital
    
    def total_pnl(self) -> float:
        """Get total realized PnL from all closed trades."""
        closed_trades = self.get_closed_trades()
        return sum(t.realized_pnl for t in closed_trades if t.realized_pnl is not None)
    
    def average_trade_duration_days(self) -> float:
        """
        Calculate average trade duration in days.
        
        Returns:
            Average duration, or 0.0 if no closed trades
        """
        closed_trades = self.get_closed_trades()
        
        if not closed_trades:
            return 0.0
        
        durations = [
            t.trade_duration_days for t in closed_trades
            if t.trade_duration_days is not None
        ]
        
        if not durations:
            return 0.0
        
        return float(np.mean(durations))
    
    def to_dict(self) -> dict:
        """
        Get all metrics as a dictionary.
        
        Returns:
            Dictionary of metric names and values
        """
        return {
            "trade_count": self.trade_count(),
            "entry_count": self.entry_count(),
            "exit_count": self.exit_count(),
            "mean_return_per_trade": self.mean_return_per_trade(),
            "sharpe_ratio": self.sharpe_ratio(),
            "max_drawdown": self.max_drawdown(),
            "profit_factor": self.profit_factor(),
            "win_rate": self.win_rate(),
            "total_return": self.total_return(),
            "final_equity": self.final_equity(),
            "total_pnl": self.total_pnl(),
            "average_trade_duration_days": self.average_trade_duration_days(),
            "sample_type": self.sample_type,
            "beta": self.beta(),
            "alpha": self.alpha(),
            "information_ratio": self.information_ratio(),
            "cagr": self.cagr()
        }
