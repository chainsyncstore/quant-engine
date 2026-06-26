"""Native Torch confirmation model interface.

This module is intentionally standalone for Phase 2. It loads a compact CPU
classifier from an explicit config dictionary plus a Torch ``state_dict`` and
does not wire predictions into routing, allocation, or model registry behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn


CONFIRMATION_MODEL_TYPE = "native_torch_confirmation"
DEFAULT_CONFIG_NAME = "config.json"
DEFAULT_STATE_DICT_NAME = "state_dict.pt"


class ConfirmationModelError(ValueError):
    """Raised when confirmation model artifacts or inputs are invalid."""


@dataclass(frozen=True)
class ConfirmationPrediction:
    """Source-level confirmation prediction for one feature row."""

    probability: float
    direction: str
    uncertainty: float
    model_version: str
    feature_schema: tuple[str, ...]


@dataclass(frozen=True)
class ConfirmationModelConfig:
    """Validated architecture and schema metadata for a confirmation model."""

    model_version: str
    feature_schema: tuple[str, ...]
    architecture: dict[str, Any]
    model_type: str = CONFIRMATION_MODEL_TYPE

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ConfirmationModelConfig":
        if not isinstance(raw, Mapping):
            raise ConfirmationModelError("confirmation config must be a mapping")

        model_type = raw.get("model_type", CONFIRMATION_MODEL_TYPE)
        if model_type != CONFIRMATION_MODEL_TYPE:
            raise ConfirmationModelError(f"unsupported confirmation model_type: {model_type!r}")

        model_version = raw.get("model_version")
        if not isinstance(model_version, str) or not model_version.strip():
            raise ConfirmationModelError("confirmation config requires non-empty model_version")

        feature_schema = raw.get("feature_schema")
        if not isinstance(feature_schema, Sequence) or isinstance(feature_schema, (str, bytes)):
            raise ConfirmationModelError("confirmation config requires feature_schema list")
        schema = tuple(feature_schema)
        if not schema or not all(isinstance(name, str) and name for name in schema):
            raise ConfirmationModelError("feature_schema must contain non-empty string names")
        if len(set(schema)) != len(schema):
            raise ConfirmationModelError("feature_schema contains duplicate names")

        architecture = raw.get("architecture")
        if not isinstance(architecture, Mapping):
            raise ConfirmationModelError("confirmation config requires architecture mapping")
        architecture_dict = dict(architecture)
        name = architecture_dict.get("name")
        if name != "mlp":
            raise ConfirmationModelError(f"unsupported confirmation architecture: {name!r}")

        input_dim = architecture_dict.get("input_dim")
        if input_dim != len(schema):
            raise ConfirmationModelError(
                "architecture input_dim must match feature_schema length"
            )

        hidden_dims = architecture_dict.get("hidden_dims", [])
        if not isinstance(hidden_dims, Sequence) or isinstance(hidden_dims, (str, bytes)):
            raise ConfirmationModelError("architecture hidden_dims must be a list")
        normalized_hidden_dims: list[int] = []
        for dim in hidden_dims:
            if not isinstance(dim, int) or dim <= 0:
                raise ConfirmationModelError("architecture hidden_dims must contain positive ints")
            normalized_hidden_dims.append(dim)

        activation = architecture_dict.get("activation", "relu")
        if activation not in {"relu", "tanh"}:
            raise ConfirmationModelError("architecture activation must be 'relu' or 'tanh'")

        architecture_dict["hidden_dims"] = normalized_hidden_dims
        architecture_dict["activation"] = activation
        return cls(
            model_version=model_version,
            feature_schema=schema,
            architecture=architecture_dict,
            model_type=model_type,
        )


class _ConfirmationMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        activation: str = "relu",
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous_dim = input_dim
        activation_layer: type[nn.Module] = nn.ReLU if activation == "relu" else nn.Tanh
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(previous_dim, hidden_dim))
            layers.append(activation_layer())
            previous_dim = hidden_dim
        layers.append(nn.Linear(previous_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


class NativeTorchConfirmationModel:
    """CPU-only deterministic confirmation classifier wrapper."""

    def __init__(self, config: ConfirmationModelConfig, model: nn.Module) -> None:
        self.config = config
        self.model = model.to(torch.device("cpu"))
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @property
    def model_version(self) -> str:
        return self.config.model_version

    @property
    def feature_schema(self) -> tuple[str, ...]:
        return self.config.feature_schema

    @classmethod
    def from_state_dict(
        cls,
        *,
        config: Mapping[str, Any] | ConfirmationModelConfig,
        state_dict: Mapping[str, torch.Tensor],
    ) -> "NativeTorchConfirmationModel":
        parsed_config = (
            config
            if isinstance(config, ConfirmationModelConfig)
            else ConfirmationModelConfig.from_dict(config)
        )
        model = _build_model(parsed_config)
        try:
            load_result = model.load_state_dict(dict(state_dict), strict=True)
        except Exception as exc:
            raise ConfirmationModelError(f"invalid confirmation state_dict: {exc}") from exc
        if load_result.missing_keys or load_result.unexpected_keys:
            raise ConfirmationModelError(
                "confirmation state_dict does not match architecture: "
                f"missing={load_result.missing_keys}, unexpected={load_result.unexpected_keys}"
            )
        return cls(parsed_config, model)

    @classmethod
    def from_files(
        cls,
        *,
        config_path: str | Path,
        state_dict_path: str | Path,
    ) -> "NativeTorchConfirmationModel":
        config_file = Path(config_path)
        state_file = Path(state_dict_path)
        if not config_file.is_file():
            raise ConfirmationModelError(f"missing confirmation config: {config_file}")
        if not state_file.is_file():
            raise ConfirmationModelError(f"missing confirmation state_dict: {state_file}")

        try:
            raw_config = json.loads(config_file.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ConfirmationModelError(f"invalid confirmation config JSON: {exc}") from exc

        try:
            state_dict = torch.load(state_file, map_location="cpu", weights_only=True)
        except Exception as exc:
            raise ConfirmationModelError(f"invalid confirmation state_dict file: {exc}") from exc
        if not isinstance(state_dict, Mapping):
            raise ConfirmationModelError("confirmation state_dict artifact must contain a mapping")

        return cls.from_state_dict(config=raw_config, state_dict=state_dict)

    @classmethod
    def from_artifact_dir(
        cls,
        artifact_dir: str | Path,
        *,
        config_name: str = DEFAULT_CONFIG_NAME,
        state_dict_name: str = DEFAULT_STATE_DICT_NAME,
    ) -> "NativeTorchConfirmationModel":
        artifact_path = Path(artifact_dir)
        return cls.from_files(
            config_path=artifact_path / config_name,
            state_dict_path=artifact_path / state_dict_name,
        )

    def predict_proba(self, features: Mapping[str, float] | pd.Series | pd.DataFrame) -> float:
        tensor = self._features_to_tensor(features)
        with torch.no_grad():
            logits = self.model(tensor)
            probability = torch.sigmoid(logits).detach().cpu().item()
        return float(np.clip(probability, 0.0, 1.0))

    def predict(
        self,
        features: Mapping[str, float] | pd.Series | pd.DataFrame,
    ) -> ConfirmationPrediction:
        probability = self.predict_proba(features)
        uncertainty = _bounded_binary_uncertainty(probability)
        if probability > 0.5:
            direction = "BUY"
        elif probability < 0.5:
            direction = "SELL"
        else:
            direction = "HOLD"
        return ConfirmationPrediction(
            probability=probability,
            direction=direction,
            uncertainty=uncertainty,
            model_version=self.model_version,
            feature_schema=self.feature_schema,
        )

    def _features_to_tensor(
        self,
        features: Mapping[str, float] | pd.Series | pd.DataFrame,
    ) -> torch.Tensor:
        values = _ordered_feature_values(features, self.feature_schema)
        array = np.asarray(values, dtype=np.float32).reshape(1, -1)
        return torch.as_tensor(array, dtype=torch.float32, device=torch.device("cpu"))


def load_confirmation_model(
    artifact_dir: str | Path,
    *,
    config_name: str = DEFAULT_CONFIG_NAME,
    state_dict_name: str = DEFAULT_STATE_DICT_NAME,
) -> NativeTorchConfirmationModel:
    """Load a confirmation model from an artifact directory.

    Expected files are ``config.json`` and ``state_dict.pt`` by default. The
    state artifact is loaded with ``weights_only=True`` and mapped to CPU.
    """

    return NativeTorchConfirmationModel.from_artifact_dir(
        artifact_dir,
        config_name=config_name,
        state_dict_name=state_dict_name,
    )


def build_confirmation_model_for_config(config: Mapping[str, Any]) -> nn.Module:
    """Build an uninitialized model for tests and offline artifact creation."""

    return _build_model(ConfirmationModelConfig.from_dict(config))


def _build_model(config: ConfirmationModelConfig) -> nn.Module:
    architecture = config.architecture
    return _ConfirmationMLP(
        input_dim=int(architecture["input_dim"]),
        hidden_dims=architecture["hidden_dims"],
        activation=architecture["activation"],
    )


def _ordered_feature_values(
    features: Mapping[str, float] | pd.Series | pd.DataFrame,
    feature_schema: tuple[str, ...],
) -> list[float]:
    if isinstance(features, pd.DataFrame):
        if len(features) != 1:
            raise ConfirmationModelError("confirmation inference requires exactly one feature row")
        incoming_names = tuple(str(column) for column in features.columns)
        _validate_feature_names(incoming_names, feature_schema)
        row = features.iloc[0]
        values = [row[name] for name in feature_schema]
    elif isinstance(features, pd.Series):
        incoming_names = tuple(str(index) for index in features.index)
        _validate_feature_names(incoming_names, feature_schema)
        values = [features[name] for name in feature_schema]
    elif isinstance(features, Mapping):
        incoming_names = tuple(str(key) for key in features.keys())
        _validate_feature_names(incoming_names, feature_schema)
        values = [features[name] for name in feature_schema]
    else:
        raise ConfirmationModelError(
            "features must be a mapping, pandas Series, or one-row DataFrame"
        )

    numeric_values: list[float] = []
    for value in values:
        try:
            numeric_value = float(value)
        except (TypeError, ValueError) as exc:
            raise ConfirmationModelError("feature values must be numeric") from exc
        if not np.isfinite(numeric_value):
            raise ConfirmationModelError("feature values must be finite")
        numeric_values.append(numeric_value)
    return numeric_values


def _validate_feature_names(
    incoming_names: Sequence[str],
    feature_schema: tuple[str, ...],
) -> None:
    incoming_set = set(incoming_names)
    expected_set = set(feature_schema)
    if incoming_set != expected_set:
        missing = sorted(expected_set - incoming_set)
        unexpected = sorted(incoming_set - expected_set)
        raise ConfirmationModelError(
            "feature schema mismatch: "
            f"missing={missing or []}, unexpected={unexpected or []}"
        )
    if len(incoming_names) != len(incoming_set):
        raise ConfirmationModelError("feature input contains duplicate names")


def _bounded_binary_uncertainty(probability: float) -> float:
    safe = float(np.clip(probability, 1e-9, 1.0 - 1e-9))
    entropy = -(safe * np.log2(safe) + (1.0 - safe) * np.log2(1.0 - safe))
    margin_uncertainty = 1.0 - abs(2.0 * safe - 1.0)
    return float(np.clip(0.5 * entropy + 0.5 * margin_uncertainty, 0.0, 1.0))
