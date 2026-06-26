from __future__ import annotations

import builtins
import importlib
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from quant_v2.models.confirmation import (
    ConfirmationModelError,
    NativeTorchConfirmationModel,
    build_confirmation_model_for_config,
    load_confirmation_model,
)


def _fixture_config() -> dict:
    return {
        "model_type": "native_torch_confirmation",
        "model_version": "fixture-v1",
        "feature_schema": ["momentum", "volatility", "liquidity"],
        "architecture": {
            "name": "mlp",
            "input_dim": 3,
            "hidden_dims": [],
            "activation": "relu",
        },
    }


def _write_fixture_artifact(tmp_path: Path, config: dict | None = None) -> Path:
    artifact_dir = tmp_path / "confirmation"
    artifact_dir.mkdir(parents=True)
    config = config or _fixture_config()
    (artifact_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")

    model = build_confirmation_model_for_config(config)
    with torch.no_grad():
        model.network[0].weight.copy_(torch.tensor([[0.25, -0.5, 0.75]], dtype=torch.float32))
        model.network[0].bias.copy_(torch.tensor([0.1], dtype=torch.float32))
    torch.save(model.state_dict(), artifact_dir / "state_dict.pt")
    return artifact_dir


def test_loads_tiny_fixture_model_from_state_dict_artifact(tmp_path: Path) -> None:
    artifact_dir = _write_fixture_artifact(tmp_path)

    model = load_confirmation_model(artifact_dir)

    assert isinstance(model, NativeTorchConfirmationModel)
    assert model.model_version == "fixture-v1"
    assert model.feature_schema == ("momentum", "volatility", "liquidity")
    assert model.model.training is False
    assert all(not parameter.requires_grad for parameter in model.model.parameters())


def test_feature_order_is_schema_driven_for_mapping_and_dataframe(tmp_path: Path) -> None:
    model = load_confirmation_model(_write_fixture_artifact(tmp_path))
    ordered = {"momentum": 1.0, "volatility": 2.0, "liquidity": 3.0}
    shuffled = {"liquidity": 3.0, "momentum": 1.0, "volatility": 2.0}
    frame = pd.DataFrame([{"liquidity": 3.0, "volatility": 2.0, "momentum": 1.0}])

    expected = model.predict_proba(ordered)

    assert model.predict_proba(shuffled) == pytest.approx(expected)
    assert model.predict_proba(frame) == pytest.approx(expected)


def test_feature_schema_validation_rejects_missing_and_unexpected_features(tmp_path: Path) -> None:
    model = load_confirmation_model(_write_fixture_artifact(tmp_path))

    with pytest.raises(ConfirmationModelError, match="missing=\\['liquidity'\\]"):
        model.predict_proba({"momentum": 1.0, "volatility": 2.0})

    with pytest.raises(ConfirmationModelError, match="unexpected=\\['extra'\\]"):
        model.predict_proba(
            {"momentum": 1.0, "volatility": 2.0, "liquidity": 3.0, "extra": 4.0}
        )


def test_prediction_is_bounded_versioned_deterministic_and_grad_free(tmp_path: Path) -> None:
    model = load_confirmation_model(_write_fixture_artifact(tmp_path))
    features = {"momentum": 1.0, "volatility": 2.0, "liquidity": 3.0}

    first = model.predict(features)
    second = model.predict(features)

    assert first == second
    assert 0.0 <= first.probability <= 1.0
    assert 0.0 <= first.uncertainty <= 1.0
    assert first.model_version == "fixture-v1"
    assert first.direction in {"BUY", "SELL", "HOLD"}
    assert all(parameter.grad is None for parameter in model.model.parameters())


def test_missing_artifacts_fail_closed_with_explicit_error(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "missing"
    artifact_dir.mkdir()

    with pytest.raises(ConfirmationModelError, match="missing confirmation config"):
        load_confirmation_model(artifact_dir)

    (artifact_dir / "config.json").write_text(json.dumps(_fixture_config()), encoding="utf-8")
    with pytest.raises(ConfirmationModelError, match="missing confirmation state_dict"):
        load_confirmation_model(artifact_dir)


def test_corrupt_or_incompatible_artifacts_raise_explicit_errors(tmp_path: Path) -> None:
    corrupt_config_dir = tmp_path / "corrupt-config"
    corrupt_config_dir.mkdir()
    (corrupt_config_dir / "config.json").write_text("{not-json", encoding="utf-8")
    torch.save({}, corrupt_config_dir / "state_dict.pt")

    with pytest.raises(ConfirmationModelError, match="invalid confirmation config JSON"):
        load_confirmation_model(corrupt_config_dir)

    corrupt_state_dir = tmp_path / "corrupt-state"
    corrupt_state_dir.mkdir()
    (corrupt_state_dir / "config.json").write_text(json.dumps(_fixture_config()), encoding="utf-8")
    (corrupt_state_dir / "state_dict.pt").write_text("not a torch state dict", encoding="utf-8")

    with pytest.raises(ConfirmationModelError, match="invalid confirmation state_dict file"):
        load_confirmation_model(corrupt_state_dir)

    incompatible_dir = _write_fixture_artifact(tmp_path / "incompatible")
    torch.save({"wrong.weight": torch.tensor([1.0])}, incompatible_dir / "state_dict.pt")

    with pytest.raises(ConfirmationModelError, match="invalid confirmation state_dict"):
        load_confirmation_model(incompatible_dir)


def test_config_validation_rejects_schema_architecture_mismatch() -> None:
    config = _fixture_config()
    config["architecture"] = dict(config["architecture"], input_dim=99)
    model = build_confirmation_model_for_config

    with pytest.raises(ConfirmationModelError, match="input_dim"):
        model(config)


def test_import_does_not_attempt_chronos_import(monkeypatch: pytest.MonkeyPatch) -> None:
    sys.modules.pop("quant_v2.models.confirmation", None)
    original_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith("chronos"):
            raise AssertionError(f"unexpected Chronos import: {name}")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    module = importlib.import_module("quant_v2.models.confirmation")

    assert module.CONFIRMATION_MODEL_TYPE == "native_torch_confirmation"
