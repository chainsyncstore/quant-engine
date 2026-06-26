"""Training/export utilities for native Torch confirmation models."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from quant_v2.models.confirmation import (
    CONFIRMATION_MODEL_TYPE,
    DEFAULT_CONFIG_NAME,
    DEFAULT_STATE_DICT_NAME,
    build_confirmation_model_for_config,
    load_confirmation_model,
)


@dataclass(frozen=True)
class ConfirmationTrainingResult:
    """Serializable training/export evidence for one confirmation model."""

    available: bool
    model_type: str
    model_version: str
    horizon: int
    feature_schema: list[str]
    train_rows: int
    validation_rows: int
    validation_accuracy: float
    shadow_only: bool
    torch_version: str
    config_path: str
    state_dict_path: str
    profitability: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def train_and_export_confirmation_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
    artifact_dir: Path | str,
    *,
    version_id: str,
    horizon: int = 4,
    hidden_dims: Sequence[int] = (16,),
    max_epochs: int = 40,
    learning_rate: float = 0.01,
    batch_size: int = 64,
    seed: int = 17,
    primary_validation_probabilities: Sequence[float] | None = None,
    validation_forward_returns: Sequence[float] | pd.Series | None = None,
    round_trip_cost_bps: float = 20.0,
    min_profitability_samples: int = 200,
    min_agreement_edge_bps: float = 2.0,
    min_agreement_win_rate: float = 0.52,
    min_agreement_coverage: float = 0.10,
) -> ConfirmationTrainingResult:
    """Train and export a small CPU MLP confirmation classifier.

    The model consumes the exact engineered feature row schema used at runtime.
    No scaler is fitted here; adding normalization requires a matching runtime
    loader change and should be handled as a separate reviewed pass.
    """

    feature_schema = _validate_training_frames(X_train, y_train, X_validation, y_validation)
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))

    config = {
        "model_type": CONFIRMATION_MODEL_TYPE,
        "model_version": f"{version_id}-confirmation-{int(horizon)}m",
        "feature_schema": list(feature_schema),
        "architecture": {
            "name": "mlp",
            "input_dim": len(feature_schema),
            "hidden_dims": [int(dim) for dim in hidden_dims],
            "activation": "relu",
        },
        "horizon": int(horizon),
        "shadow_only": True,
        "torch_version": torch.__version__,
    }
    model = build_confirmation_model_for_config(config).to(torch.device("cpu"))
    model.train()

    train_x = _frame_to_tensor(X_train, feature_schema)
    train_y = _labels_to_tensor(y_train)
    loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=max(1, int(batch_size)),
        shuffle=True,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    loss_fn = nn.BCEWithLogitsLoss()

    for _epoch in range(max(1, int(max_epochs))):
        for batch_x, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = loss_fn(logits, batch_y)
            loss.backward()
            optimizer.step()

    validation_probabilities = _validation_probabilities(model, X_validation, feature_schema)
    validation_accuracy = _validation_accuracy_from_probabilities(
        validation_probabilities,
        y_validation,
    )
    profitability = None
    if (
        primary_validation_probabilities is not None
        and validation_forward_returns is not None
    ):
        profitability = evaluate_confirmation_profitability(
            primary_probabilities=primary_validation_probabilities,
            confirmation_probabilities=validation_probabilities,
            forward_returns=validation_forward_returns,
            round_trip_cost_bps=round_trip_cost_bps,
            min_profitability_samples=min_profitability_samples,
            min_agreement_edge_bps=min_agreement_edge_bps,
            min_agreement_win_rate=min_agreement_win_rate,
            min_agreement_coverage=min_agreement_coverage,
        )
    confirmation_dir = Path(artifact_dir).expanduser() / "confirmation"
    confirmation_dir.mkdir(parents=True, exist_ok=True)
    config_path = confirmation_dir / DEFAULT_CONFIG_NAME
    state_dict_path = confirmation_dir / DEFAULT_STATE_DICT_NAME
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    torch.save(model.state_dict(), state_dict_path)

    # Prove exported artifact loads through the production inference path.
    load_confirmation_model(confirmation_dir)
    return ConfirmationTrainingResult(
        available=True,
        model_type=CONFIRMATION_MODEL_TYPE,
        model_version=str(config["model_version"]),
        horizon=int(horizon),
        feature_schema=list(feature_schema),
        train_rows=int(len(X_train)),
        validation_rows=int(len(X_validation)),
        validation_accuracy=float(validation_accuracy),
        shadow_only=True,
        torch_version=torch.__version__,
        config_path=str(config_path.relative_to(Path(artifact_dir).expanduser())),
        state_dict_path=str(state_dict_path.relative_to(Path(artifact_dir).expanduser())),
        profitability=profitability,
    )


def confirmation_unavailable(reason: str, *, horizon: int = 4) -> dict[str, Any]:
    return {
        "available": False,
        "horizon": int(horizon),
        "reason": str(reason),
        "shadow_only": True,
    }


def evaluate_confirmation_profitability(
    *,
    primary_probabilities: Sequence[float],
    confirmation_probabilities: Sequence[float],
    forward_returns: Sequence[float] | pd.Series,
    round_trip_cost_bps: float = 20.0,
    min_profitability_samples: int = 200,
    min_agreement_edge_bps: float = 2.0,
    min_agreement_win_rate: float = 0.52,
    min_agreement_coverage: float = 0.10,
) -> dict[str, Any]:
    """Compare baseline-vs-confirmation agreement on net holdout returns.

    The simulated trade follows the baseline model's direction; confirmation is
    only used to split the holdout into agreement and disagreement cohorts.
    Costs are modeled as a flat round-trip bps haircut so the evidence is about
    net profitability, not just directional accuracy.
    """

    primary = _finite_array(primary_probabilities, name="primary_probabilities")
    confirmation = _finite_array(
        confirmation_probabilities,
        name="confirmation_probabilities",
    )
    returns = _finite_array(forward_returns, name="forward_returns")
    if len(primary) != len(confirmation) or len(primary) != len(returns):
        raise ValueError("profitability inputs must have matching lengths")
    if len(primary) == 0:
        raise ValueError("profitability validation requires at least one row")

    primary_direction = np.where(primary >= 0.5, 1.0, -1.0)
    confirmation_direction = np.where(confirmation >= 0.5, 1.0, -1.0)
    agreement_mask = primary_direction == confirmation_direction
    net_returns = primary_direction * returns - (float(round_trip_cost_bps) / 10_000.0)

    agreement = _cohort_profitability(net_returns[agreement_mask])
    disagreement = _cohort_profitability(net_returns[~agreement_mask])
    sample_count = int(len(net_returns))
    agreement_count = int(agreement["sample_count"])
    disagreement_count = int(disagreement["sample_count"])
    agreement_coverage = agreement_count / sample_count
    disagreement_coverage = disagreement_count / sample_count
    edge_bps = float(agreement["mean_net_return_bps"] - disagreement["mean_net_return_bps"])
    gate_eligible = (
        sample_count >= int(min_profitability_samples)
        and agreement_count > 0
        and disagreement_count > 0
        and agreement_coverage >= float(min_agreement_coverage)
        and float(agreement["mean_net_return_bps"]) > 0.0
        and edge_bps >= float(min_agreement_edge_bps)
        and float(agreement["win_rate"]) >= float(min_agreement_win_rate)
    )

    return {
        "sample_count": sample_count,
        "round_trip_cost_bps": float(round_trip_cost_bps),
        "agreement": {
            **agreement,
            "coverage": float(agreement_coverage),
        },
        "disagreement": {
            **disagreement,
            "coverage": float(disagreement_coverage),
        },
        "agreement_edge_bps": edge_bps,
        "gate_eligible": bool(gate_eligible),
        "gate_eligibility_criteria": {
            "min_profitability_samples": int(min_profitability_samples),
            "min_agreement_edge_bps": float(min_agreement_edge_bps),
            "min_agreement_win_rate": float(min_agreement_win_rate),
            "min_agreement_coverage": float(min_agreement_coverage),
        },
        "mode": "offline_holdout_shadow_evidence",
    }


def _validate_training_frames(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
) -> tuple[str, ...]:
    if X_train.empty or X_validation.empty:
        raise ValueError("confirmation training requires non-empty train and validation frames")
    if len(X_train) != len(y_train):
        raise ValueError("X_train and y_train length mismatch")
    if len(X_validation) != len(y_validation):
        raise ValueError("X_validation and y_validation length mismatch")
    feature_schema = tuple(str(column) for column in X_train.columns)
    if not feature_schema:
        raise ValueError("confirmation training requires at least one feature")
    if len(set(feature_schema)) != len(feature_schema):
        raise ValueError("confirmation feature schema contains duplicates")
    if tuple(str(column) for column in X_validation.columns) != feature_schema:
        raise ValueError("confirmation validation features must match training schema")
    for frame in (X_train, X_validation):
        values = frame.to_numpy(dtype=np.float32, copy=True)
        if not np.isfinite(values).all():
            raise ValueError("confirmation features must be finite")
    for labels in (y_train, y_validation):
        values = pd.to_numeric(labels, errors="coerce").to_numpy(dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError("confirmation labels must be finite")
        unique = set(float(value) for value in values)
        if not unique.issubset({0.0, 1.0}):
            raise ValueError("confirmation labels must be binary 0/1")
    return feature_schema


def _frame_to_tensor(frame: pd.DataFrame, feature_schema: tuple[str, ...]) -> torch.Tensor:
    values = frame.loc[:, list(feature_schema)].to_numpy(dtype=np.float32, copy=True)
    return torch.as_tensor(values, dtype=torch.float32, device=torch.device("cpu"))


def _labels_to_tensor(labels: pd.Series) -> torch.Tensor:
    values = pd.to_numeric(labels, errors="raise").to_numpy(dtype=np.float32, copy=True)
    return torch.as_tensor(values, dtype=torch.float32, device=torch.device("cpu"))


def _validation_probabilities(
    model: nn.Module,
    X_validation: pd.DataFrame,
    feature_schema: tuple[str, ...],
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(_frame_to_tensor(X_validation, feature_schema))
        probabilities = torch.sigmoid(logits).detach().cpu().numpy()
    return np.asarray(probabilities, dtype=float)


def _validation_accuracy(
    model: nn.Module,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
    feature_schema: tuple[str, ...],
) -> float:
    return _validation_accuracy_from_probabilities(
        _validation_probabilities(model, X_validation, feature_schema),
        y_validation,
    )


def _validation_accuracy_from_probabilities(
    probabilities: Sequence[float],
    y_validation: pd.Series,
) -> float:
    predictions = (np.asarray(probabilities, dtype=float) > 0.5).astype(int)
    labels = pd.to_numeric(y_validation, errors="raise").to_numpy(dtype=int)
    return float(np.mean(predictions == labels))


def _finite_array(values: Sequence[float] | pd.Series, *, name: str) -> np.ndarray:
    array = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _cohort_profitability(net_returns: np.ndarray) -> dict[str, Any]:
    if len(net_returns) == 0:
        return {
            "sample_count": 0,
            "win_rate": 0.0,
            "mean_net_return_bps": 0.0,
            "median_net_return_bps": 0.0,
            "total_net_return_bps": 0.0,
            "max_drawdown_bps": 0.0,
        }
    equity = np.cumprod(1.0 + net_returns)
    running_max = np.maximum.accumulate(equity)
    drawdown = np.where(running_max > 0, equity / running_max - 1.0, 0.0)
    return {
        "sample_count": int(len(net_returns)),
        "win_rate": float(np.mean(net_returns > 0.0)),
        "mean_net_return_bps": float(np.mean(net_returns) * 10_000.0),
        "median_net_return_bps": float(np.median(net_returns) * 10_000.0),
        "total_net_return_bps": float(np.sum(net_returns) * 10_000.0),
        "max_drawdown_bps": float(abs(np.min(drawdown)) * 10_000.0),
    }
