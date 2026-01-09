from __future__ import annotations

import csv
from datetime import datetime, timedelta
from pathlib import Path


def generate(path: Path, bars: int = 250) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    start = datetime(2024, 1, 1)
    price = 100.0
    rows = []

    for i in range(bars):
        timestamp = start + timedelta(days=i)
        drift = 0.001 * ((i % 5) - 2)
        open_price = price
        close_price = open_price * (1 + drift)
        high_price = max(open_price, close_price) * 1.002
        low_price = min(open_price, close_price) * 0.998
        volume = 1_000 + (10 * i)

        rows.append(
            (
                timestamp.isoformat(),
                round(open_price, 2),
                round(high_price, 2),
                round(low_price, 2),
                round(close_price, 2),
                round(volume, 2),
            )
        )
        price = close_price

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", "open", "high", "low", "close", "volume"])
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {path}")


if __name__ == "__main__":
    target_path = Path(r"c:\\Users\\HP\\Downloads\\hypothesis-research-engine\\results\\synthetic_meta.csv")
    generate(target_path)
