"""Test Phase 1+3 improvements: dead-zone labels + tuned hyperparameters.

Quick test with 3 months of data to verify accuracy improvement.
"""
import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

from quant_v2.research.scheduled_retrain import retrain_and_promote

model_root = Path(os.getenv("BOT_MODEL_ROOT", "/app/models/production"))
registry_root = Path(os.getenv("BOT_MODEL_REGISTRY_ROOT", str(model_root / "registry")))

print("=" * 60)
print("TEST: Phase 1+3 (dead-zone labels + tuned hyperparams)")
print("=" * 60)
print("Training with 3 months BTC+ETH only (fast test)...")
print(f"Model root: {model_root}")
print(f"Registry: {registry_root}")

# Quick test: 3 months, 2 symbols, strict accuracy gate
version_id = retrain_and_promote(
    model_root=model_root,
    registry_root=registry_root,
    train_months=3,
    min_accuracy=0.525,
    extra_symbols=["ETHUSDT"],  # Just 2 symbols for speed
)

if version_id:
    print(f"\n✅ SUCCESS: Promoted {version_id}")
    print("Phase 1+3 changes are working - model passed accuracy gate")
else:
    print("\n❌ FAILED: No model passed the 0.525 accuracy gate")
    print("Check logs for details")
