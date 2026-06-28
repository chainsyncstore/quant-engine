from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant.features.pipeline import build_features, get_feature_columns
from quant_v2.models.trainer import load_model, save_model_bundle, train
from quant_v2.research.portfolio_replay import (
    ReplayActorConfig,
    ReplayScenario,
    run_portfolio_replay,
    write_replay_artifact,
)


def _single_symbol_bars(
    *,
    symbol: str,
    rows: int = 144,
    trend: float = 0.0,
    phase: float = 0.0,
) -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=rows, freq="h", tz="UTC")
    wave = np.sin(np.linspace(phase, phase + 10.0, rows))
    base = 100.0 + (wave * 4.0) + np.linspace(0.0, trend, rows)
    frame = pd.DataFrame(
        {
            "open": base * 0.99,
            "high": base * 1.01,
            "low": base * 0.98,
            "close": base,
            "volume": 1_000.0 + np.abs(wave) * 50.0,
            "taker_buy_volume": (1_000.0 + np.abs(wave) * 50.0) * 0.55,
            "taker_sell_volume": (1_000.0 + np.abs(wave) * 50.0) * 0.45,
            "funding_rate": np.sin(np.linspace(phase, phase + 4.0, rows)) * 0.0001,
            "open_interest": 10_000.0 + np.linspace(0.0, 100.0, rows),
            "open_interest_value": 1_000_000.0 + np.linspace(0.0, 10_000.0, rows),
        },
        index=idx,
    )
    frame.index.name = "timestamp"
    frame["symbol"] = symbol
    return frame


def _multi_symbol_bars() -> pd.DataFrame:
    pieces = []
    for symbol, trend, phase in (
        ("BTCUSDT", 8.0, 0.25),
        ("ETHUSDT", -4.0, 1.25),
    ):
        frame = _single_symbol_bars(symbol=symbol, trend=trend, phase=phase)
        frame = frame.reset_index().set_index(["timestamp", "symbol"]).sort_index()
        pieces.append(frame)
    return pd.concat(pieces).sort_index()


def _trained_model(tmp_path: Path, monkeypatch) -> object:
    raw = _single_symbol_bars(symbol="BTCUSDT", rows=180, trend=6.0, phase=0.2)
    featured = build_features(raw.drop(columns=["symbol"]).copy())
    feature_cols = get_feature_columns(featured)
    close = pd.to_numeric(featured["close"], errors="coerce")
    future_up = (close.shift(-1) > close).astype(int)
    mask = featured[feature_cols].replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
    mask &= future_up.notna()
    X = featured.loc[mask, feature_cols]
    y = future_up.loc[mask].astype(int)

    monkeypatch.setenv("QUANT_IMAGE", "registry.example/quant-bot@sha256:" + "a" * 64)
    trained = train(X, y, horizon=1, calibration_frac=0.2)
    artifact = tmp_path / "model_1m.pkl"
    save_model_bundle(
        trained,
        artifact,
        metadata={
            "threshold": 0.60,
            "threshold_policy": {
                "source": "oof_dev_predictions",
                "selected_threshold": 0.60,
                "selected_accuracy": 0.63,
            },
        },
    )
    return load_model(artifact)


def test_portfolio_replay_is_deterministic_and_content_addressed(
    tmp_path: Path,
    monkeypatch,
) -> None:
    model = _trained_model(tmp_path, monkeypatch)
    dataset = _multi_symbol_bars()

    actors = {
        "candidate": ReplayActorConfig(
            name="candidate",
            kind="model",
            model=model,
            threshold=0.52,
            min_confidence=0.50,
            horizon_bars=1,
            metadata={"role": "candidate"},
        ),
        "incumbent": ReplayActorConfig(
            name="incumbent",
            kind="model",
            model=model,
            threshold=0.84,
            min_confidence=0.50,
            horizon_bars=1,
            metadata={"role": "incumbent"},
        ),
        "benchmark": ReplayActorConfig(
            name="benchmark",
            kind="baseline",
            min_confidence=0.50,
            baseline_lookback=3,
            baseline_deadband=0.0005,
            horizon_bars=1,
            metadata={"role": "benchmark"},
        ),
    }

    manifest = {
        "dataset_name": "unit_replay",
        "symbols": ["BTCUSDT", "ETHUSDT"],
        "interval": "1h",
    }
    scenario = ReplayScenario(name="base")

    replay_1 = run_portfolio_replay(
        dataset,
        actors,
        initial_equity=1_000.0,
        scenario=scenario,
        dataset_manifest=manifest,
    )
    replay_2 = run_portfolio_replay(
        dataset,
        actors,
        initial_equity=1_000.0,
        scenario=scenario,
        dataset_manifest=manifest,
    )

    assert replay_1.replay_digest == replay_2.replay_digest
    assert replay_1.actors["candidate"].state_digest == replay_2.actors["candidate"].state_digest
    assert replay_1.actors["candidate"].metrics["fill_count"] > 0
    assert replay_1.actors["candidate"].state_digest != replay_1.actors["incumbent"].state_digest
    assert replay_1.actors["benchmark"].metrics["fill_count"] > 0

    artifact_path = write_replay_artifact(replay_1, tmp_path)
    assert artifact_path.name == f"portfolio_replay_{replay_1.replay_digest}.json"
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert payload["replay_digest"] == replay_1.replay_digest
    assert payload["manifest"]["dataset_digest"] == replay_1.manifest["dataset_digest"]
    assert payload["actors"]["candidate"]["state_digest"] == replay_1.actors["candidate"].state_digest


def test_replay_actor_defaults_to_model_threshold_floor(tmp_path: Path, monkeypatch) -> None:
    model = _trained_model(tmp_path, monkeypatch)
    actor = ReplayActorConfig(
        name="candidate",
        kind="model",
        model=model,
    )

    assert actor.threshold == pytest.approx(0.60)


def test_portfolio_replay_scenario_injection_changes_costs_and_open_orders(
    tmp_path: Path,
    monkeypatch,
) -> None:
    model = _trained_model(tmp_path, monkeypatch)
    dataset = _multi_symbol_bars()

    actors = {
        "candidate": ReplayActorConfig(
            name="candidate",
            kind="model",
            model=model,
            threshold=0.52,
            min_confidence=0.50,
            horizon_bars=1,
        ),
    }

    base = run_portfolio_replay(
        dataset,
        actors,
        initial_equity=1_000.0,
        scenario=ReplayScenario(name="base"),
        dataset_manifest={"dataset_name": "unit_replay"},
    )
    adverse = run_portfolio_replay(
        dataset,
        actors,
        initial_equity=1_000.0,
        scenario=ReplayScenario(
            name="adverse",
            spread_multiplier=1.5,
            fill_ratio=0.5,
            mark_jump_bps=75.0,
            reject_symbols=("ETHUSDT",),
            restart_after_bars=18,
        ),
        dataset_manifest={"dataset_name": "unit_replay"},
    )

    base_actor = base.actors["candidate"]
    adverse_actor = adverse.actors["candidate"]
    base_filled_qty = sum(fill.filled_qty for fill in base_actor.fills)
    adverse_filled_qty = sum(fill.filled_qty for fill in adverse_actor.fills)

    assert base.replay_digest != adverse.replay_digest
    assert adverse_actor.metrics["blocked_intents"] > base_actor.metrics["blocked_intents"]
    assert adverse_actor.metrics["fill_count"] >= base_actor.metrics["fill_count"]
    assert adverse_filled_qty < base_filled_qty
    assert adverse_actor.metrics["open_order_count"] >= base_actor.metrics["open_order_count"]
    assert adverse_actor.reconciliation["status"] in {"OK", "BLOCKED"}
