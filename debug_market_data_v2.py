
from datetime import datetime, timedelta
from datetime import timezone

from quant.data.binance_client import BinanceClient


def main() -> None:
    client = BinanceClient()
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=7)

    print(f"Fetching Binance supplementary data from {start} to {end}...")

    ohlcv = client.fetch_historical(start, end)
    funding = client.fetch_funding_rates(start, end)
    oi = client.fetch_open_interest(start, end)
    merged = BinanceClient.merge_supplementary(ohlcv, funding, oi)

    print(f"OHLCV bars: {len(ohlcv)}")
    print(f"Funding rows: {len(funding)}")
    print(f"Open interest rows: {len(oi)}")
    print(f"Merged rows: {len(merged)}")

    if not merged.empty:
        print("Merged columns:", list(merged.columns))
        print("Sample tail:")
        print(merged.tail(3).to_string())


if __name__ == "__main__":
    main()
