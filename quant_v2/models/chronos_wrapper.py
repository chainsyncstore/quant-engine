"""Chronos time-series foundation model wrapper for next-bar prediction."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Lazy-load to avoid import overhead when Chronos is not used
_pipeline: Any = None


def _get_pipeline() -> Any:
    """Lazy-load the Chronos pipeline (downloads model on first use)."""
    global _pipeline
    if _pipeline is None:
        import torch
        from chronos import ChronosPipeline

        _pipeline = ChronosPipeline.from_pretrained(
            "amazon/chronos-t5-small",  # 20M params, fast inference
            device_map="cpu",
            torch_dtype=torch.float32,
        )
    return _pipeline


def predict_next_bar_direction(
    close_series: pd.Series,
    prediction_length: int = 4,
) -> tuple[float, float]:
    """Predict probability of price going up over next ``prediction_length`` bars.

    Parameters
    ----------
    close_series : pd.Series
        Historical close prices (at least 64 bars recommended).
    prediction_length : int
        Number of bars to forecast ahead.

    Returns
    -------
    tuple[float, float]
        (probability_up, uncertainty) both in [0.0, 1.0]
    """
    if len(close_series) < 32:
        return 0.5, 1.0  # insufficient data

    import torch

    pipeline = _get_pipeline()

    # Chronos expects a torch tensor of shape (1, context_length)
    context = torch.tensor(
        close_series.values[-256:],
        dtype=torch.float32,
    ).unsqueeze(0)

    # Generate probabilistic forecast (multiple samples)
    with torch.no_grad():
        forecast = pipeline.predict(
            context,
            prediction_length=prediction_length,
            num_samples=50,
        )
    # forecast shape: (1, num_samples, prediction_length)
    samples = forecast[0].numpy()  # (num_samples, prediction_length)

    # Direction: compare final forecasted price to current price
    current_price = float(close_series.iloc[-1])
    final_prices = samples[:, -1]  # last bar of each sample
    prob_up = float(np.mean(final_prices > current_price))

    # Uncertainty from spread of samples
    std_ratio = float(np.std(final_prices) / max(abs(current_price), 1e-8))
    uncertainty = float(np.clip(std_ratio * 10.0, 0.0, 1.0))  # Scale to [0,1]

    return (
        float(np.clip(prob_up, 0.0, 1.0)),
        float(np.clip(uncertainty, 0.0, 1.0)),
    )
