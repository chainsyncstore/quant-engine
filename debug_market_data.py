
from datetime import datetime, timedelta, timezone

from quant.data.binance_client import BinanceClient


def main() -> None:
    client = BinanceClient()
    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=48)

    print(f"Fetching Binance BTCUSDT klines from {start} to {end}...")
    df = client.fetch_historical(start, end)

    print(f"Bars: {len(df)}")
    if not df.empty:
        print("Columns:", list(df.columns))
        print("Last row:")
        print(df.tail(1).to_string())


if __name__ == "__main__":
    main()
