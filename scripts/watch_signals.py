"""Lightweight watcher that surfaces new trade signals for manual execution.

Usage (from repo root):
    python -m scripts.watch_signals \
        --log results/competition_live_log.jsonl \
        --beep

The script tails the specified JSONL log (default: results/competition_live_log.jsonl)
and prints a one-line alert whenever a new execution report is recorded. These alerts
include the timestamp, symbol, action, and suggested lot size so you can execute the
trade manually inside MT5 when EAs are forbidden.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional
from urllib import request as urllib_request
from urllib.error import URLError
import ssl

TELEGRAM_BOT_TOKEN = "8569598946:AAH09JpNS1BUciwreXjloZwHwQQIfVn-unE"
TELEGRAM_CHAT_ID = "6268794073"


try:  # Windows-only optional beep
    import winsound  # type: ignore
except ImportError:  # pragma: no cover
    winsound = None  # type: ignore


def _maybe_beep() -> None:
    if winsound is None:
        return
    winsound.Beep(880, 200)
    winsound.Beep(1320, 200)


def _format_alert(payload: dict) -> str:
    ts = payload.get("timestamp") or datetime.now(timezone.utc).isoformat()
    symbol = payload.get("symbol", "?")
    action = payload.get("action", "?")
    qty = payload.get("filled_quantity", 0.0)
    price = payload.get("avg_fill_price")
    return (
        f"{ts} | {symbol} | {action.upper()} | qty={qty:.2f} | "
        + (f"price={price:.5f}" if price is not None else "price=?")
    )


def _should_alert(symbol: str, action: str, active_positions: Dict[str, str]) -> bool:
    action = action.upper()
    if action == "CLOSE":
        active_positions.pop(symbol, None)
        return True

    if action in {"BUY", "SELL"}:
        last = active_positions.get(symbol)
        if last == action:
            # Already told trader to enter in this direction; wait for CLOSE before repeating
            return False
        active_positions[symbol] = action
        return True

    return True


def _send_telegram_message(text: str, verify_tls: bool) -> None:
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "disable_web_page_preview": True,
    }
    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib_request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        if verify_tls:
            context = ssl.create_default_context()
        else:
            context = ssl._create_unverified_context()  # pragma: no cover - user opt-in

        with urllib_request.urlopen(req, timeout=5, context=context):
            pass
    except (URLError, TimeoutError, ValueError) as exc:  # pragma: no cover
        print(f"[WARN] Failed to send Telegram message: {exc}")


def _tail_file(
    path: Path, poll_seconds: float, beep: bool, telegram: bool, telegram_verify: bool, replay_last: int = 0
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Watching {path} (Ctrl+C to exit)...")

    active_positions: Dict[str, str] = {}

    with path.open("a+", encoding="utf-8") as handle:
        if replay_last > 0:
            # Replay last N execution_report lines (console only, no telegram/beep)
            handle.seek(0)
            all_lines = handle.readlines()
            exec_lines = [l for l in all_lines if '"execution_report"' in l and '"FILLED"' in l]
            for line in exec_lines[-replay_last:]:
                try:
                    event = json.loads(line.strip())
                    payload = event.get("payload", {})
                    alert = _format_alert(payload)
                    print(f"[REPLAY] {alert}")
                except:
                    pass
            if exec_lines:
                print("--- End of replay (console only), watching for new signals ---")
            else:
                print("--- No previous trades found, watching for new signals ---")
        handle.seek(0, os.SEEK_END)
        while True:
            line = handle.readline()
            if not line:
                time.sleep(poll_seconds)
                continue

            line = line.strip()
            if not line:
                continue

            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                print(f"[WARN] Skipping non-JSON line: {line[:80]}")
                continue

            event_type = event.get("event_type")
            payload = event.get("payload") or {}

            if event_type != "execution_report":
                continue

            status = payload.get("status", "").upper()
            if status != "FILLED":
                continue

            symbol = payload.get("symbol", "?")
            action = payload.get("action", "").upper()
            if not _should_alert(symbol, action, active_positions):
                continue

            alert = _format_alert(payload)
            print(f"[SIGNAL] {alert}")
            if beep:
                _maybe_beep()
            if telegram:
                _send_telegram_message(f"📈 {alert}", verify_tls=telegram_verify)


def main() -> None:
    parser = argparse.ArgumentParser(description="Tail execution log and surface signals")
    parser.add_argument(
        "--log",
        default="results/competition_live_log.jsonl",
        help="Path to the JSONL execution log produced by run_meta.py",
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=1.0,
        help="How often to poll for new lines (seconds)",
    )
    parser.add_argument(
        "--beep",
        action="store_true",
        help="Play a short sound whenever a new signal arrives (Windows only)",
    )
    parser.add_argument(
        "--telegram",
        action="store_true",
        help="Forward alerts to the configured Telegram bot",
    )
    parser.add_argument(
        "--insecure-telegram",
        action="store_true",
        help="Disable TLS verification for Telegram webhook (use only if cert errors appear)",
    )
    parser.add_argument(
        "--replay",
        type=int,
        default=0,
        help="Replay the last N filled trades on startup before watching for new ones",
    )

    args = parser.parse_args()
    log_path = Path(args.log)
    if not log_path.exists():
        print(f"[INFO] Log {log_path} does not exist yet. Waiting for it to be created...")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.touch()

    try:
        _tail_file(
            log_path,
            args.poll_seconds,
            args.beep,
            args.telegram,
            not args.insecure_telegram,
            args.replay,
        )
    except KeyboardInterrupt:
        print("\nWatcher stopped.")


if __name__ == "__main__":
    main()
