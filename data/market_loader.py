"""
Market data loader for OHLCV time-series data.

Loads historical market data from CSV files and validates chronological ordering.
No transformations or indicators are applied - raw data only.
"""

from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pd

from data.schemas import Bar


class MarketDataLoader:
    """
    Loads OHLCV market data from CSV files.
    
    Expected CSV format:
    - timestamp (ISO format or parseable date string)
    - open (float)
    - high (float)
    - low (float)
    - close (float)
    - volume (float)
    """
    
    REQUIRED_COLUMNS = {'timestamp', 'open', 'high', 'low', 'close', 'volume'}

    @classmethod
    def load_from_csv(cls, file_path: str | Path, symbol: str | None = None) -> List[Bar]:
        """
        Load market data from a CSV file.
        
        Args:
            file_path: Path to CSV file
            symbol: Optional symbol/ticker to attach to bars
            
        Returns:
            List of validated Bar objects in chronological order
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If data is invalid or not chronologically ordered
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Market data file not found: {file_path}")
        
        df = pd.read_csv(file_path)
        return cls.load_from_dataframe(df, symbol=symbol)

    @classmethod
    def load_from_dataframe(cls, df: pd.DataFrame, symbol: str | None = None) -> List[Bar]:
        """
        Load market data from an in-memory dataframe.

        Args:
            df: DataFrame containing OHLCV data
            symbol: Optional symbol/ticker to attach to bars
        """
        prepared_df = cls._prepare_dataframe(df)
        bars = cls._dataframe_to_bars(prepared_df, symbol)

        if not bars:
            raise ValueError("No valid bars found in dataframe")

        return bars

    @classmethod
    def _prepare_dataframe(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and normalize a dataframe prior to conversion."""
        missing_cols = cls.REQUIRED_COLUMNS - set(df.columns)

        if missing_cols:
            raise ValueError(
                f"Missing required columns in data: {missing_cols}. "
                f"Found columns: {list(df.columns)}"
            )

        normalized = df.copy()
        normalized['timestamp'] = pd.to_datetime(normalized['timestamp'])
        normalized = normalized.sort_values('timestamp').reset_index(drop=True)

        # Handle duplicate timestamps - drop duplicates, keeping the last occurrence
        if 'symbol' in normalized.columns:
            # For multi-symbol data, drop duplicates within each symbol
            before_count = len(normalized)
            normalized = normalized.drop_duplicates(
                subset=['symbol', 'timestamp'], 
                keep='last'
            ).reset_index(drop=True)
            dropped = before_count - len(normalized)
            if dropped > 0:
                import logging
                logging.getLogger(__name__).warning(
                    "Dropped %d duplicate timestamp rows (multi-symbol)", dropped
                )
        else:
            before_count = len(normalized)
            normalized = normalized.drop_duplicates(
                subset=['timestamp'], 
                keep='last'
            ).reset_index(drop=True)
            dropped = before_count - len(normalized)
            if dropped > 0:
                import logging
                logging.getLogger(__name__).warning(
                    "Dropped %d duplicate timestamp rows", dropped
                )

        return normalized

    @staticmethod
    def _dataframe_to_bars(df: pd.DataFrame, symbol: str | None) -> List[Bar]:
        """Convert a validated dataframe into Bar objects."""
        # Filter by symbol if multi-symbol data and symbol is specified
        if symbol and 'symbol' in df.columns:
            df = df[df['symbol'] == symbol].reset_index(drop=True)
        
        bars: List[Bar] = []

        for idx, row in df.iterrows():
            try:
                timestamp = row['timestamp']
                if hasattr(timestamp, 'to_pydatetime'):
                    timestamp = timestamp.to_pydatetime()

                bar = Bar(
                    timestamp=timestamp,
                    open=float(row['open']),
                    high=float(row['high']),
                    low=float(row['low']),
                    close=float(row['close']),
                    volume=float(row['volume']),
                    symbol=symbol or row.get('symbol')
                )
                bars.append(bar)
            except Exception as e:
                raise ValueError(
                    f"Invalid bar data at row {idx}: {e}"
                ) from e
        
        # Validate chronological ordering per symbol (multi-symbol data may have same timestamps)
        symbol_last_ts: dict = {}
        for i, bar in enumerate(bars):
            sym = bar.symbol or "__DEFAULT__"
            if sym in symbol_last_ts:
                if bar.timestamp < symbol_last_ts[sym]:
                    raise ValueError(
                        f"Data for {sym} is not in chronological order at index {i}: "
                        f"{symbol_last_ts[sym]} -> {bar.timestamp}"
                    )
            symbol_last_ts[sym] = bar.timestamp
        
        if not bars:
            raise ValueError("No valid bars found in CSV file")
        
        return bars
    
    @staticmethod
    def create_synthetic_data(
        symbol: str,
        start_date: datetime,
        num_bars: int,
        initial_price: float = 100.0,
        volatility: float = 0.02,
        trend: float = 0.0001,
        seed: int = 42
    ) -> List[Bar]:
        """
        Create synthetic market data for testing.
        
        Args:
            symbol: Symbol/ticker
            start_date: Start date for data
            num_bars: Number of bars to generate
            initial_price: Starting price
            volatility: Daily volatility (std dev of returns)
            trend: Daily trend (mean return)
            seed: Random seed for reproducibility
            
        Returns:
            List of synthetic Bar objects
        """
        import numpy as np
        
        np.random.seed(seed)
        
        bars: List[Bar] = []
        current_price = initial_price
        
        for i in range(num_bars):
            # Generate random return
            return_pct = np.random.normal(trend, volatility)
            
            # Calculate OHLC
            open_price = current_price
            close_price = open_price * (1 + return_pct)
            
            # High and low with some randomness
            intrabar_range = abs(return_pct) * 1.5
            high_price = max(open_price, close_price) * (1 + np.random.uniform(0, intrabar_range))
            low_price = min(open_price, close_price) * (1 - np.random.uniform(0, intrabar_range))
            
            # Ensure high/low consistency
            high_price = max(high_price, open_price, close_price)
            low_price = min(low_price, open_price, close_price)
            
            # Random volume
            volume = np.random.uniform(1_000_000, 5_000_000)
            
            # Timestamp (daily bars)
            timestamp = start_date + pd.Timedelta(days=i)
            
            bar = Bar(
                timestamp=timestamp,
                open=round(open_price, 2),
                high=round(high_price, 2),
                low=round(low_price, 2),
                close=round(close_price, 2),
                volume=round(volume, 0),
                symbol=symbol
            )
            
            bars.append(bar)
            current_price = close_price
        
        return bars
    
    @staticmethod
    def create_synthetic(num_bars: int, symbol: str = "SYNTHETIC") -> List[Bar]:
        """Alias for create_synthetic_data with defaults."""
        return MarketDataLoader.create_synthetic_data(
            symbol=symbol,
            start_date=datetime(2020, 1, 1),
            num_bars=num_bars
        )
